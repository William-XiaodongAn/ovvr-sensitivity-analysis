"""Standalone 2D ten Tusscher-Panfilov (2006) monodomain PDE solver.

Time integration matches the reference (Chaos Lab WebGL) solver:
  * 12 gating variables : Rush-Larsen (exact exponential, unconditionally stable)
  * concentrations + V  : forward Euler
  * voltage diffusion   : explicit 9-point Laplacian
PyTorch backend (CUDA/MPS/CPU). Self-contained: no dependency on simulation.py
or drug_dict.pkl.

Monodomain equation solved per grid node:

    dV/dt   = -(I_ion + I_stim) / C_m + D * lap(V)
    dy_k/dt = f_k(state)                              (k = the other 18 vars)

Only V (index 17) is spatially coupled; every other variable is a local ODE.

Spatial scaling matters: the grid covers a physical domain of size `domain_size`
(default 12), so dx = domain_size / grid_size. This keeps dx ~ 0.02-0.05 and lets
dt = 0.1 ms be stable -- exactly the reference setting (ds=12, dt=0.1, D=0.001).
Using dx = 1/grid_size instead would force a microscopic dt.

Usage
-----
    python tnnp_pde.py --grid-size 256 --domain-size 12 --total-time 3000 \
        --pacing-period 1000 --snapshot-interval 1000 --device cuda
"""

import argparse
import os
import sys
import time

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit('This solver requires PyTorch (pip install torch).') from exc


# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

# Order of the 19 state variables (index == position in the state vector).
STATE_NAMES = [
    'Xr1', 'Xr2', 'Xs', 'm', 'h', 'j', 'd', 'f', 'f2', 'fCass',
    's', 'r', 'Ca_i', 'R_prime', 'Ca_SR', 'Ca_ss', 'Na_i', 'V', 'K_i',
]
V_INDEX = 17

# Equilibrated resting state (ten Tusscher-Panfilov 2006), one value per var.
Y0 = np.array([
    0.000206466599002,   # 0  Xr1
    0.473142295376,      # 1  Xr2
    0.00322170878365,    # 2  Xs
    0.00164959539136,    # 3  m
    0.750058790356,      # 4  h
    0.749677890643,      # 5  j
    3.28817457136e-05,   # 6  d
    0.977249616825,      # 7  f
    0.999410380975,      # 8  f2
    0.999972733836,      # 9  fCass
    0.999997921133,      # 10 s
    2.34167244267e-08,   # 11 r
    0.000104622621021,   # 12 Ca_i    [mM]
    0.989176724937,      # 13 R_prime
    3.50685455674,       # 14 Ca_SR   [mM]
    0.000212791419736,   # 15 Ca_ss   [mM]
    9.79776167522,       # 16 Na_i    [mM]
    -85.4001369649,      # 17 V       [mV]
    135.733295852,       # 18 K_i     [mM]
], dtype=float)

PARAMS = {
    'P_kna': 0.03, 'g_K1': 5.405, 'g_Kr': 0.153, 'g_Ks': 0.392, 'g_Na': 14.838,
    'g_bna': 0.00029, 'g_CaL': 3.98e-05, 'g_bca': 0.000592, 'g_to': 0.294,
    'K_mNa': 40.0, 'K_mk': 1.0, 'P_NaK': 2.724, 'K_NaCa': 1000.0, 'K_sat': 0.1,
    'Km_Ca': 1.38, 'Km_Nai': 87.5, 'alpha': 2.5, 'gamma': 0.35, 'K_pCa': 0.0005,
    'g_pCa': 0.1238, 'g_pK': 0.0146, 'Buf_c': 0.2, 'Buf_sr': 10.0, 'Buf_ss': 0.4,
    'Ca_o': 2.0, 'EC': 1.5, 'K_buf_c': 0.001, 'K_buf_sr': 0.3, 'K_buf_ss': 0.00025,
    'K_up': 0.00025, 'V_leak': 0.00036, 'V_rel': 0.102, 'V_sr': 0.001094,
    'V_ss': 5.468e-05, 'V_xfer': 0.0038, 'Vmax_up': 0.006375, 'k1_prime': 0.15,
    'k2_prime': 0.045, 'k3': 0.06, 'k4': 0.005, 'max_sr': 2.5, 'min_sr': 1.0,
    'Na_o': 140.0, 'Cm': 0.185, 'F': 96485.3415, 'R': 8314.472, 'T': 310.0,
    'V_c': 0.016404, 'stim_amplitude': 40.0, 'stim_duration': 1.0, 'stim_start': 0.0,
    'K_o': 5.4,
}


def _exp(x):
    """Overflow-guarded exp; float32 saturates far sooner than float64."""
    limit = 80.0 if x.dtype == torch.float32 else 700.0
    return torch.exp(torch.clamp(x, -limit, limit))


# ---------------------------------------------------------------------------
# Ionic currents and right-hand side
# ---------------------------------------------------------------------------

def currents(state, p, g):
    """Return a dict of ionic currents for a state field shaped [..., 19].

    ``g`` scales channel conductances (all 1.0 = drug-free baseline).
    """
    Xr1, Xr2, Xs = state[..., 0], state[..., 1], state[..., 2]
    m, h, j = state[..., 3], state[..., 4], state[..., 5]
    d, f, f2, fCass = state[..., 6], state[..., 7], state[..., 8], state[..., 9]
    s, r = state[..., 10], state[..., 11]
    Ca_i, R_prime, Ca_SR, Ca_ss = state[..., 12], state[..., 13], state[..., 14], state[..., 15]
    Na_i, V, K_i = state[..., 16], state[..., 17], state[..., 18]

    RTF = p['R'] * p['T'] / p['F']
    E_Na = RTF * torch.log(p['Na_o'] / Na_i)
    E_K = RTF * torch.log(p['K_o'] / K_i)
    E_Ks = RTF * torch.log((p['K_o'] + p['Na_o'] * p['P_kna']) / (p['P_kna'] * Na_i + K_i))
    E_Ca = 0.5 * RTF * torch.log(p['Ca_o'] / Ca_i)

    alpha_K1 = 0.1 / (1 + 6.14421235333e-06 * _exp(0.06 * V - 0.06 * E_K))
    beta_K1 = (
        0.367879441171 * _exp(0.1 * V - 0.1 * E_K)
        + 3.06060402008 * _exp(0.0002 * V - 0.0002 * E_K)
    ) / (1 + _exp(0.5 * E_K - 0.5 * V))
    i_K1 = 0.430331482912 * p['g_K1'] * g['IK1'] * (p['K_o'] ** 0.5) * (V - E_K) * alpha_K1 / (alpha_K1 + beta_K1)
    i_Kr = 0.430331482912 * p['g_Kr'] * g['IKr'] * (p['K_o'] ** 0.5) * (V - E_K) * Xr1 * Xr2
    i_Ks = p['g_Ks'] * g['IKs'] * Xs ** 2 * (V - E_Ks)
    i_Na = p['g_Na'] * g['INa'] * m ** 3 * (V - E_Na) * h * j
    i_b_Na = p['g_bna'] * g['IbNa'] * (V - E_Na)

    # I_CaL has a removable singularity at V = 15 mV; blend to a local limit.
    exp_term = _exp(p['F'] * (-30 + 2 * V) / (p['R'] * p['T']))
    i_CaL_reg = (
        4 * p['g_CaL'] * g['ICaL'] * p['F'] ** 2 * (V - 15)
        * (-p['Ca_o'] + 0.25 * Ca_ss * exp_term) * d * f * f2 * fCass
        / (p['R'] * p['T'] * (-1 + exp_term))
    )
    diff = 1e-7
    exp_near = _exp(torch.as_tensor(2 * p['F'] * diff / (p['R'] * p['T']), dtype=V.dtype, device=V.device))
    i_CaL_near = (
        p['g_CaL'] * g['ICaL'] * 4 * diff * p['F'] ** 2
        * (-p['Ca_o'] + 0.25 * Ca_ss * exp_near) * d * f * f2 * fCass
        / (p['R'] * p['T'] * (-1 + exp_near))
    )
    i_CaL = torch.where(torch.abs(V - 15.0) < 1e-7, i_CaL_near, i_CaL_reg)

    i_b_Ca = p['g_bca'] * g['IbCa'] * (V - E_Ca)
    i_to = p['g_to'] * g['Ito'] * (V - E_K) * r * s
    i_NaK = (
        p['K_o'] * p['P_NaK'] * g['INaK'] * Na_i
        / ((p['K_mNa'] + Na_i) * (p['K_mk'] + p['K_o'])
           * (1 + 0.0353 * _exp(-p['F'] * V / (p['R'] * p['T']))
              + 0.1245 * _exp(-0.1 * p['F'] * V / (p['R'] * p['T']))))
    )
    i_NaCa = (
        p['K_NaCa'] * g['INaCa']
        * (p['Ca_o'] * Na_i ** 3 * _exp(p['F'] * p['gamma'] * V / (p['R'] * p['T']))
           - p['alpha'] * p['Na_o'] ** 3 * Ca_i * _exp(p['F'] * (-1 + p['gamma']) * V / (p['R'] * p['T'])))
        / ((1 + p['K_sat'] * _exp(p['F'] * (-1 + p['gamma']) * V / (p['R'] * p['T'])))
           * (p['Ca_o'] + p['Km_Ca']) * (p['Km_Nai'] ** 3 + p['Na_o'] ** 3))
    )
    i_p_Ca = p['g_pCa'] * g['IpCa'] * Ca_i / (p['K_pCa'] + Ca_i)
    i_p_K = p['g_pK'] * g['IpK'] * (V - E_K) / (1 + 65.4052157419 * _exp(-0.167224080268 * V))

    i_up = p['Vmax_up'] / (1 + p['K_up'] ** 2 / Ca_i ** 2)
    i_leak = p['V_leak'] * (Ca_SR - Ca_i)
    i_xfer = p['V_xfer'] * (Ca_ss - Ca_i)
    kcasr = p['max_sr'] - (p['max_sr'] - p['min_sr']) / (1 + p['EC'] ** 2 / Ca_SR ** 2)
    k1 = p['k1_prime'] / kcasr
    O = Ca_ss ** 2 * R_prime * k1 / (p['k3'] + Ca_ss ** 2 * k1)
    i_rel = p['V_rel'] * (Ca_SR - Ca_ss) * O

    return {
        'INa': i_Na, 'Ito': i_to, 'ICaL': i_CaL, 'IKs': i_Ks, 'IpK': i_p_K,
        'INaK': i_NaK, 'IKr': i_Kr, 'INaCa': i_NaCa, 'IK1': i_K1, 'IbCa': i_b_Ca,
        'IpCa': i_p_Ca, 'IbNa': i_b_Na, 'Iup': i_up, 'Ileak': i_leak,
        'Ixfer': i_xfer, 'Irel': i_rel,
    }


def rhs(state, p, g, i_stim):
    """Full dY/dt for the 19 state variables (no diffusion term here).

    ``i_stim`` is the stimulus current field broadcast over the grid.
    """
    Xr1, Xr2, Xs = state[..., 0], state[..., 1], state[..., 2]
    m, h, j = state[..., 3], state[..., 4], state[..., 5]
    d, f, f2, fCass = state[..., 6], state[..., 7], state[..., 8], state[..., 9]
    s, r = state[..., 10], state[..., 11]
    Ca_i, R_prime, Ca_SR, Ca_ss = state[..., 12], state[..., 13], state[..., 14], state[..., 15]
    V = state[..., 17]

    dY = torch.zeros_like(state)
    c = currents(state, p, g)

    # --- gating variables (steady-state / time-constant form) ---
    xr1_inf = 1.0 / (1 + _exp(-26 / 7 - V / 7))
    a_xr1 = 450 / (1 + _exp(-9 / 2 - V / 10))
    b_xr1 = 6 / (1 + 13.5813245226 * _exp(0.0869565217391 * V))
    dY[..., 0] = (xr1_inf - Xr1) / (a_xr1 * b_xr1)

    xr2_inf = 1.0 / (1 + _exp(11 / 3 + V / 24))
    a_xr2 = 3 / (1 + _exp(-3 - V / 20))
    b_xr2 = 1.12 / (1 + _exp(-3 + V / 20))
    dY[..., 1] = (xr2_inf - Xr2) / (a_xr2 * b_xr2)

    xs_inf = 1.0 / (1 + _exp(-5 / 14 - V / 14))
    a_xs = 1400 / torch.sqrt(1 + _exp(5 / 6 - V / 6))
    b_xs = 1.0 / (1 + _exp(-7 / 3 + V / 15))
    dY[..., 2] = (xs_inf - Xs) / (80 + a_xs * b_xs)

    m_inf = (1 + 0.00184221158117 * _exp(-0.110741971207 * V)) ** -2
    a_m = 1.0 / (1 + _exp(-12 - V / 5))
    b_m = 0.1 / (1 + _exp(7 + V / 5)) + 0.1 / (1 + _exp(-1 / 4 + V / 200))
    dY[..., 3] = (m_inf - m) / (a_m * b_m)

    h_inf = (1 + 15212.5932857 * _exp(0.134589502019 * V)) ** -2
    a_h_lo = 4.43126792958e-07 * _exp(-0.147058823529 * V)
    b_h_lo = 310000 * _exp(0.3485 * V) + 2.7 * _exp(0.079 * V)
    a_j_lo = ((37.78 + V) * (-25428 * _exp(0.2444 * V) - 6.948e-06 * _exp(-0.04391 * V))
              / (1 + 50262745826.0 * _exp(0.311 * V)))
    b_j_lo = 0.02424 * _exp(-0.01052 * V) / (1 + 0.0039608683399 * _exp(-0.1378 * V))
    b_h_hi = 0.77 / (0.13 + 0.0497581410839 * _exp(-0.0900900900901 * V))
    b_j_hi = 0.6 * _exp(0.057 * V) / (1 + 0.0407622039784 * _exp(-0.1 * V))
    low_v = V < -40
    zero = torch.zeros_like(V)
    a_h = torch.where(low_v, a_h_lo, zero)
    b_h = torch.where(low_v, b_h_lo, b_h_hi)
    a_j = torch.where(low_v, a_j_lo, zero)
    b_j = torch.where(low_v, b_j_lo, b_j_hi)
    dY[..., 4] = (h_inf - h) * (a_h + b_h)
    dY[..., 5] = (h_inf - j) * (a_j + b_j)

    d_inf = 1.0 / (1 + 0.344153786865 * _exp(-0.133333333333 * V))
    a_d = 0.25 + 1.4 / (1 + _exp(-35 / 13 - V / 13))
    b_d = 1.4 / (1 + _exp(1 + V / 5))
    g_d = 1.0 / (1 + _exp(5 / 2 - V / 20))
    dY[..., 6] = (d_inf - d) / (a_d * b_d + g_d)

    f_inf = 1.0 / (1 + _exp(20 / 7 + V / 7))
    tau_f = 20 + 180 / (1 + _exp(3 + V / 10)) + 200 / (1 + _exp(13 / 10 - V / 10)) + 1102.5 * _exp(-(27 + V) ** 2 / 225)
    dY[..., 7] = (f_inf - f) / tau_f

    f2_inf = 0.33 + 0.67 / (1 + _exp(5 + V / 7))
    tau_f2 = 31 / (1 + _exp(5 / 2 - V / 10)) + 80 / (1 + _exp(3 + V / 10)) + 562 * _exp(-(27 + V) ** 2 / 240)
    dY[..., 8] = (f2_inf - f2) / tau_f2

    fCass_inf = 0.4 + 0.6 / (1 + 400.0 * Ca_ss ** 2)
    tau_fCass = 2 + 80 / (1 + 400.0 * Ca_ss ** 2)
    dY[..., 9] = (fCass_inf - fCass) / tau_fCass

    dY[..., 10] = (1.0 / (1 + _exp(4 + V / 5)) - s) / (3 + 5 / (1 + _exp(-4 + V / 5)) + 85 * _exp(-(45 + V) ** 2 / 320))
    dY[..., 11] = (1.0 / (1 + _exp(10 / 3 - V / 6)) - r) / (0.8 + 9.5 * _exp(-(40 + V) ** 2 / 1800))

    # --- concentrations ---
    Ca_i_bufc = 1.0 / (1 + p['Buf_c'] * p['K_buf_c'] / (p['K_buf_c'] + Ca_i) ** 2)
    Ca_sr_bufsr = 1.0 / (1 + p['Buf_sr'] * p['K_buf_sr'] / (p['K_buf_sr'] + Ca_SR) ** 2)
    Ca_ss_bufss = 1.0 / (1 + p['Buf_ss'] * p['K_buf_ss'] / (p['K_buf_ss'] + Ca_ss) ** 2)
    kcasr = p['max_sr'] - (p['max_sr'] - p['min_sr']) / (1 + p['EC'] ** 2 / Ca_SR ** 2)
    k2 = p['k2_prime'] * kcasr

    dY[..., 12] = (
        p['V_sr'] * (-c['Iup'] + c['Ileak']) / p['V_c']
        - p['Cm'] * (-2 * c['INaCa'] + c['IbCa'] + c['IpCa']) / (2 * p['F'] * p['V_c'])
        + c['Ixfer']
    ) * Ca_i_bufc
    dY[..., 13] = p['k4'] * (1 - R_prime) - Ca_ss * R_prime * k2
    dY[..., 14] = (-c['Ileak'] - c['Irel'] + c['Iup']) * Ca_sr_bufsr
    dY[..., 15] = (
        p['V_sr'] * c['Irel'] / p['V_ss'] - p['V_c'] * c['Ixfer'] / p['V_ss']
        - p['Cm'] * c['ICaL'] / (2 * p['F'] * p['V_ss'])
    ) * Ca_ss_bufss
    dY[..., 16] = p['Cm'] * (-c['INa'] - c['IbNa'] - 3 * c['INaCa'] - 3 * c['INaK']) / (p['F'] * p['V_c'])

    # --- transmembrane potential (ionic part only; diffusion added by caller) ---
    dY[..., 17] = -(
        c['ICaL'] + c['IK1'] + c['IKr'] + c['IKs'] + c['INa'] + c['INaCa']
        + c['INaK'] + i_stim + c['IbCa'] + c['IbNa'] + c['IpCa'] + c['IpK'] + c['Ito']
    )
    dY[..., 18] = p['Cm'] * (
        -c['IK1'] - c['IKr'] - c['IKs'] - i_stim - c['IpK'] - c['Ito'] + 2 * c['INaK']
    ) / (p['F'] * p['V_c'])
    return dY


def rush_larsen_gates(state, dt):
    """Exact-exponential (Rush-Larsen) update for the 12 gating variables.

        g_new = g_inf - (g_inf - g) * exp(-dt / tau)

    Unconditionally stable regardless of dt, which is exactly what lets the
    reference (WebGL) solver march at dt = 0.1 ms despite the very fast INa
    activation (tau_m can be < 0.1 ms). Concentrations and V are integrated
    separately by forward Euler in the caller.
    """
    V = state[..., 17]
    Ca_ss = state[..., 15]
    new = state.clone()
    specs = []  # (state index, g_inf, tau)

    xr1_inf = 1.0 / (1 + _exp(-26 / 7 - V / 7))
    specs.append((0, xr1_inf, (450 / (1 + _exp(-9 / 2 - V / 10)))
                              * (6 / (1 + 13.5813245226 * _exp(0.0869565217391 * V)))))
    xr2_inf = 1.0 / (1 + _exp(11 / 3 + V / 24))
    specs.append((1, xr2_inf, (3 / (1 + _exp(-3 - V / 20)))
                              * (1.12 / (1 + _exp(-3 + V / 20)))))
    xs_inf = 1.0 / (1 + _exp(-5 / 14 - V / 14))
    a_xs = 1400 / torch.sqrt(1 + _exp(5 / 6 - V / 6))
    b_xs = 1.0 / (1 + _exp(-7 / 3 + V / 15))
    specs.append((2, xs_inf, 80 + a_xs * b_xs))
    m_inf = (1 + 0.00184221158117 * _exp(-0.110741971207 * V)) ** -2
    a_m = 1.0 / (1 + _exp(-12 - V / 5))
    b_m = 0.1 / (1 + _exp(7 + V / 5)) + 0.1 / (1 + _exp(-1 / 4 + V / 200))
    specs.append((3, m_inf, a_m * b_m))

    h_inf = (1 + 15212.5932857 * _exp(0.134589502019 * V)) ** -2
    a_h_lo = 4.43126792958e-07 * _exp(-0.147058823529 * V)
    b_h_lo = 310000 * _exp(0.3485 * V) + 2.7 * _exp(0.079 * V)
    a_j_lo = ((37.78 + V) * (-25428 * _exp(0.2444 * V) - 6.948e-06 * _exp(-0.04391 * V))
              / (1 + 50262745826.0 * _exp(0.311 * V)))
    b_j_lo = 0.02424 * _exp(-0.01052 * V) / (1 + 0.0039608683399 * _exp(-0.1378 * V))
    b_h_hi = 0.77 / (0.13 + 0.0497581410839 * _exp(-0.0900900900901 * V))
    b_j_hi = 0.6 * _exp(0.057 * V) / (1 + 0.0407622039784 * _exp(-0.1 * V))
    low_v = V < -40
    zero = torch.zeros_like(V)
    a_h = torch.where(low_v, a_h_lo, zero)
    b_h = torch.where(low_v, b_h_lo, b_h_hi)
    a_j = torch.where(low_v, a_j_lo, zero)
    b_j = torch.where(low_v, b_j_lo, b_j_hi)
    specs.append((4, h_inf, 1.0 / (a_h + b_h)))
    specs.append((5, h_inf, 1.0 / (a_j + b_j)))

    d_inf = 1.0 / (1 + 0.344153786865 * _exp(-0.133333333333 * V))
    a_d = 0.25 + 1.4 / (1 + _exp(-35 / 13 - V / 13))
    b_d = 1.4 / (1 + _exp(1 + V / 5))
    g_d = 1.0 / (1 + _exp(5 / 2 - V / 20))
    specs.append((6, d_inf, a_d * b_d + g_d))
    f_inf = 1.0 / (1 + _exp(20 / 7 + V / 7))
    tau_f = 20 + 180 / (1 + _exp(3 + V / 10)) + 200 / (1 + _exp(13 / 10 - V / 10)) + 1102.5 * _exp(-(27 + V) ** 2 / 225)
    specs.append((7, f_inf, tau_f))
    f2_inf = 0.33 + 0.67 / (1 + _exp(5 + V / 7))
    tau_f2 = 31 / (1 + _exp(5 / 2 - V / 10)) + 80 / (1 + _exp(3 + V / 10)) + 562 * _exp(-(27 + V) ** 2 / 240)
    specs.append((8, f2_inf, tau_f2))
    fCass_inf = 0.4 + 0.6 / (1 + 400.0 * Ca_ss ** 2)
    specs.append((9, fCass_inf, 2 + 80 / (1 + 400.0 * Ca_ss ** 2)))
    specs.append((10, 1.0 / (1 + _exp(4 + V / 5)),
                  3 + 5 / (1 + _exp(-4 + V / 5)) + 85 * _exp(-(45 + V) ** 2 / 320)))
    specs.append((11, 1.0 / (1 + _exp(10 / 3 - V / 6)),
                  0.8 + 9.5 * _exp(-(40 + V) ** 2 / 1800)))

    for idx, g_inf, tau in specs:
        new[..., idx] = g_inf - (g_inf - state[..., idx]) * torch.exp(-dt / tau)
    return new


# ---------------------------------------------------------------------------
# Spatial diffusion (explicit 9-point stencil, replicate/no-flux boundary)
# ---------------------------------------------------------------------------

def laplacian_kernel(dx, dy, dtype, device):
    cddx, cddy = 1.0 / dx ** 2, 1.0 / dy ** 2
    gamma = 1.0 / 3.0
    diag = gamma * 0.5 * (cddx + cddy)
    horiz = (1.0 - gamma) * cddx
    vert = (1.0 - gamma) * cddy
    center = -2.0 * (cddx + cddy)
    k = torch.tensor([[diag, vert, diag], [horiz, center, horiz], [diag, vert, diag]],
                     dtype=dtype, device=device)
    return k.view(1, 1, 3, 3)


def laplacian(V, kernel):
    padded = F.pad(V.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
    return F.conv2d(padded, kernel).squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# Stimulus
# ---------------------------------------------------------------------------

def pacemaker_mask(point, radius, nx, ny, dtype, device):
    px, py = point
    cx, cy = px * (nx - 1), py * (ny - 1)
    ys = torch.arange(ny, dtype=dtype, device=device).view(ny, 1)
    xs = torch.arange(nx, dtype=dtype, device=device).view(1, nx)
    if radius <= 0:
        mask = torch.zeros(ny, nx, dtype=dtype, device=device)
        mask[int(round(cy)), int(round(cx))] = 1.0
        return mask
    r_pix = radius * max(nx - 1, ny - 1)
    dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    return (dist2 <= r_pix ** 2).to(dtype)


def stimulus_value(t, pacing_period, p):
    phase = t - pacing_period * np.floor(t / pacing_period)
    if p['stim_start'] <= phase <= p['stim_start'] + p['stim_duration']:
        return -p['stim_amplitude']
    return 0.0


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve(nx=256, ny=256, dt=0.1, total_time=500.0, pacing_period=1000.0,
          diff_coef=0.001, c_m=1.0, domain_size=12.0, dx=None, dy=None,
          pacemaker_point=(0.05, 0.5), pacemaker_radius=0.03,
          measurement_point=(0.5, 0.5), record_interval=0.1,
          conductance_scale=None, device=None, progress_interval=5.0,
          snapshot_interval=1000.0, snapshot_dir=None,
          finite_check_interval=1000):
    """Forward-Euler monodomain solve.

    Returns (times, voltage_trace, final_state, snapshots), where snapshots is a
    list of (sim_time_ms, voltage_grid) taken every ``snapshot_interval`` ms.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available()
                              else ('mps' if getattr(torch.backends, 'mps', None)
                                    and torch.backends.mps.is_available() else 'cpu'))
    device = torch.device(device)
    dtype = torch.float32 if device.type in {'cuda', 'mps'} else torch.float64

    # Physical grid spacing: domain of size `domain_size` covered by nx cells.
    # (Using dx = 1/nx would shrink dx as the grid refines and force a tiny dt.)
    dx = domain_size / nx if dx is None else dx
    dy = domain_size / ny if dy is None else dy
    g = {name: 1.0 for name in (
        'INa', 'IKr', 'ICaL', 'IKs', 'Ito', 'IK1', 'IpK', 'INaK',
        'INaCa', 'IbCa', 'IpCa', 'IbNa')}
    if conductance_scale:
        g.update(conductance_scale)

    p = PARAMS
    kernel = laplacian_kernel(dx, dy, dtype, device)
    pace_mask = pacemaker_mask(pacemaker_point, pacemaker_radius, nx, ny, dtype, device)
    my = int(round(measurement_point[1] * (ny - 1)))
    mx = int(round(measurement_point[0] * (nx - 1)))

    # Broadcast the equilibrated resting state to every node -> [ny, nx, 19].
    y0 = torch.as_tensor(Y0, dtype=dtype, device=device)
    state = y0.view(1, 1, 19).expand(ny, nx, 19).clone()

    num_steps = int(np.ceil(total_time / dt))
    rec_v, rec_t, next_rec = [], [], 0.0
    snapshots, snap_count = [], 0
    snap_enabled = snapshot_interval and snapshot_interval > 0
    next_snap = 0.0 if snap_enabled else None
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
    wall0 = time.perf_counter()
    last_report = wall0

    with torch.no_grad():
        for step in range(num_steps + 1):
            t = min(step * dt, total_time)
            while t + 1e-9 >= next_rec:
                rec_v.append(state[my, mx, V_INDEX].clone())
                rec_t.append(next_rec)
                next_rec += record_interval

            # Voltage-field snapshot + simulation-time report every snapshot_interval ms.
            if snap_enabled and t + 1e-9 >= next_snap:
                grid = state[..., V_INDEX].detach().cpu().numpy().copy()
                snapshots.append((t, grid))
                sys.stderr.write(
                    f'\n[snapshot {snap_count:03d}] sim t={t:8.1f} ms  wall={time.perf_counter() - wall0:6.1f}s  '
                    f'V min/mean/max = {grid.min():7.2f} /{grid.mean():7.2f} /{grid.max():7.2f} mV\n')
                sys.stderr.flush()
                if snapshot_dir:
                    np.save(os.path.join(snapshot_dir, f'V_{snap_count:03d}_t{t:.0f}ms.npy'), grid)
                snap_count += 1
                while next_snap <= t + 1e-9:
                    next_snap += snapshot_interval

            now = time.perf_counter()
            if progress_interval and now - last_report >= progress_interval:
                sys.stderr.write(f'\rsim t={t:.1f}/{total_time:.1f} ms  wall={now - wall0:.1f}s')
                sys.stderr.flush()
                last_report = now

            if t >= total_time:
                break

            h = min(dt, total_time - t)
            i_stim = pace_mask * stimulus_value(t, pacing_period, p)
            lap = laplacian(state[..., V_INDEX], kernel)

            # 1) gates: exact-exponential (Rush-Larsen); 2) concentrations + V: forward Euler.
            new_state = rush_larsen_gates(state, h)
            rates = rhs(new_state, p, g, i_stim)             # currents use the updated gates
            new_state[..., 12:19] = new_state[..., 12:19] + h * rates[..., 12:19]
            new_state[..., V_INDEX] = state[..., V_INDEX] + h * (rates[..., V_INDEX] / c_m + diff_coef * lap)

            if finite_check_interval and (step % finite_check_interval == 0):
                if not bool(torch.isfinite(new_state).all()):
                    raise RuntimeError(f'Non-finite state at t={t + h:g} ms (dt likely too large)')
            state = new_state

    sys.stderr.write('\n')
    times = np.array(rec_t)
    voltage = torch.stack(rec_v).cpu().numpy().astype(float) if rec_v else np.array([])
    return times, voltage, state, snapshots


def _stable_dt(nx, ny, diff_coef, domain_size):
    """Reference dt = 0.1 ms (Rush-Larsen gates make the reaction unconditionally
    stable); capped only by the explicit-diffusion CFL limit for safety."""
    dx, dy = domain_size / nx, domain_size / ny
    diff_limit = 0.2 * min(dx, dy) ** 2 / diff_coef if diff_coef > 0 else 0.1
    return min(0.1, diff_limit)


def write_gif(snapshots, path, vmin=-90.0, vmax=40.0, cmap='turbo', fps=10, upscale=1):
    """Assemble the recorded voltage snapshots into an animated GIF."""
    if not snapshots:
        print('No snapshots recorded; GIF not written.')
        return
    try:
        import matplotlib
        from matplotlib.colors import Normalize
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        print(f'GIF output needs matplotlib + Pillow ({exc}); skipping GIF.')
        return
    try:
        mapper = matplotlib.colormaps[cmap]
    except (AttributeError, KeyError):
        import matplotlib.cm as cm
        mapper = cm.get_cmap(cmap)
    norm = Normalize(vmin=vmin, vmax=vmax)
    frames = []
    for _, grid in snapshots:
        rgb = (mapper(norm(grid))[..., :3] * 255).astype(np.uint8)
        img = Image.fromarray(rgb)
        if upscale > 1:
            img = img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)
    print(f'Wrote {len(frames)}-frame GIF -> {path}')


def main():
    ap = argparse.ArgumentParser(description='Forward-Euler 2D TNNP monodomain PDE solver (PyTorch).')
    ap.add_argument('--grid-size', type=int, default=256, help='Square grid n (nx=ny=n)')
    ap.add_argument('--domain-size', type=float, default=12.0,
                    help='Physical domain size (dx = domain_size/grid_size); reference uses 12')
    ap.add_argument('--dt', type=float, default=None, help='Time step in ms (default: reference 0.1, CFL-capped)')
    ap.add_argument('--total-time', type=float, default=500.0, help='Simulated time in ms')
    ap.add_argument('--pacing-period', type=float, default=1000.0, help='Pace period: pacing cycle length in ms')
    ap.add_argument('--pace-point', type=lambda s: tuple(map(float, s.split(','))), default=(0.05, 0.5),
                    help='Pacemaker position as x,y in [0,1]^2 (fraction of the grid)')
    ap.add_argument('--diff-coef', type=float, default=0.001, help='Voltage diffusion coefficient')
    ap.add_argument('--pace-radius', type=float, default=0.03)
    ap.add_argument('--measure-point', type=lambda s: tuple(map(float, s.split(','))), default=(0.5, 0.5))
    ap.add_argument('--record-interval', type=float, default=0.1, help='AP sampling interval in ms')
    ap.add_argument('--device', type=str, default=None, help='cuda, cuda:0, mps, or cpu')
    ap.add_argument('--progress-interval', type=float, default=5.0, help='Wall-clock seconds between progress prints')
    ap.add_argument('--snapshot-interval', type=float, default=1000.0,
                    help='Sim-time ms between voltage snapshots + time reports (default 1000 = every 1 s)')
    ap.add_argument('--snapshot-dir', type=str, default=None,
                    help='If set, also save each snapshot as a .npy voltage grid in this directory')
    ap.add_argument('--gif', type=str, default='tnnp_2d.gif', help='Output animated GIF path (default on)')
    ap.add_argument('--no-gif', action='store_true', help='Disable GIF output')
    ap.add_argument('--fps', type=int, default=10, help='GIF frames per second')
    args = ap.parse_args()

    n = args.grid_size
    dt = args.dt if args.dt is not None else _stable_dt(n, n, args.diff_coef, args.domain_size)
    dev = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    dx = args.domain_size / n
    print(f'TNNP monodomain: {n}x{n}  domain={args.domain_size:g} (dx={dx:g})  dt={dt:g} ms  '
          f'total={args.total_time:g} ms  D={args.diff_coef}  device={dev}')

    t0 = time.perf_counter()
    times, voltage, _, snapshots = solve(
        nx=n, ny=n, dt=dt, total_time=args.total_time, pacing_period=args.pacing_period,
        diff_coef=args.diff_coef, domain_size=args.domain_size, pacemaker_point=args.pace_point,
        pacemaker_radius=args.pace_radius, measurement_point=args.measure_point,
        record_interval=args.record_interval, device=args.device,
        progress_interval=args.progress_interval,
        snapshot_interval=args.snapshot_interval, snapshot_dir=args.snapshot_dir,
    )
    elapsed = time.perf_counter() - t0
    if voltage.size:
        print(f'Done in {elapsed:.2f}s | {len(snapshots)} snapshots | measured V: '
              f'min={voltage.min():.2f} max={voltage.max():.2f} mV | samples={voltage.size}')
    else:
        print(f'Done in {elapsed:.2f}s | {len(snapshots)} snapshots | no samples recorded')

    if not args.no_gif:
        write_gif(snapshots, args.gif, fps=args.fps)


if __name__ == '__main__':
    main()
