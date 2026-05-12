import math
import pickle

import numpy as np
from scipy.integrate import solve_ivp


CURRENT_NAMES = [
    'INa', 'Ito', 'ICaL', 'IKs', 'IpK', 'INaK',
    'IKr', 'INaCa', 'IK1', 'IbCa', 'IpCa', 'IbNa',
]

MATLAB_INITIAL_STATE = np.array([
    2.06466599e-04, 4.73142295e-01, 3.22170878e-03,
    1.64959539e-03, 7.50058790e-01, 7.49677891e-01,
    3.28817457e-05, 9.77249617e-01, 9.99410381e-01,
    9.99972734e-01, 9.99997921e-01, 2.34167244e-08,
    1.04622621e-04, 9.89176725e-01, 3.50685456e+00,
    2.12791420e-04, 9.79776168e+00, -8.54001370e+01,
    1.35733296e+02,
], dtype=float)


def get_drug_data(drug_name):
    with open('drug_dict.pkl', 'rb') as f:
        drug_dict = pickle.load(f)
    if drug_name not in drug_dict:
        print(f"Warning: Drug '{drug_name}' not found in drug_dict. Returning empty parameters.")
        return {}
    return drug_dict.get(drug_name, {})


def update_conductivity(C, EFTPC, EFTPC_multiplier, IC50, h):
    if IC50 == 0 or IC50 is None:
        return C
    return C * (1 / (1 + (EFTPC * EFTPC_multiplier / IC50) ** h))


def _safe_exp(x):
    return math.exp(max(-700.0, min(700.0, x)))


def _base_parameters():
    return {
        'P_kna': 0.03,
        'g_K1': 5.405,
        'g_Kr': 0.153,
        'g_Ks': 0.392,
        'g_Na': 14.838,
        'g_bna': 0.00029,
        'g_CaL': 3.98e-05,
        'g_bca': 0.000592,
        'g_to': 0.294,
        'K_mNa': 40.0,
        'K_mk': 1.0,
        'P_NaK': 2.724,
        'K_NaCa': 1000.0,
        'K_sat': 0.1,
        'Km_Ca': 1.38,
        'Km_Nai': 87.5,
        'alpha': 2.5,
        'gamma': 0.35,
        'K_pCa': 0.0005,
        'g_pCa': 0.1238,
        'g_pK': 0.0146,
        'Buf_c': 0.2,
        'Buf_sr': 10.0,
        'Buf_ss': 0.4,
        'Ca_o': 2.0,
        'EC': 1.5,
        'K_buf_c': 0.001,
        'K_buf_sr': 0.3,
        'K_buf_ss': 0.00025,
        'K_up': 0.00025,
        'V_leak': 0.00036,
        'V_rel': 0.102,
        'V_sr': 0.001094,
        'V_ss': 5.468e-05,
        'V_xfer': 0.0038,
        'Vmax_up': 0.006375,
        'k1_prime': 0.15,
        'k2_prime': 0.045,
        'k3': 0.06,
        'k4': 0.005,
        'max_sr': 2.5,
        'min_sr': 1.0,
        'Na_o': 140.0,
        'Cm': 0.185,
        'F': 96485.3415,
        'R': 8314.472,
        'T': 310.0,
        'V_c': 0.016404,
        'stim_amplitude': 40.0,
        'stim_duration': 1.0,
        'stim_start': 0.0,
        'K_o': 5.4,
    }


def _conductance_multipliers(drug_name, drug_concentration_multiplier, perturb_multipliers):
    multipliers = {
        'INa': 1.0, 'IKr': 1.0, 'ICaL': 1.0, 'IKs': 1.0,
        'Ito': 1.0, 'IK1': 1.0, 'IpK': 1.0, 'INaK': 1.0,
        'INaCa': 1.0, 'IbCa': 1.0, 'IpCa': 1.0, 'IbNa': 1.0,
        'Ileak': 1.0, 'Iup': 1.0, 'Ixfer': 1.0, 'Irel': 1.0,
    }

    drug_data = get_drug_data(drug_name)
    EFTPC = drug_data.get('EFTPCmax', 0.0)
    for channel in ['INa', 'IKr', 'ICaL', 'IKs', 'Ito', 'IK1']:
        if channel in drug_data:
            ic50 = drug_data[channel].get('IC50', 0)
            h = drug_data[channel].get('h', 1.0)
            if h is None:
                h = 1.0
            if ic50 > 0:
                multipliers[channel] = update_conductivity(
                    multipliers[channel], EFTPC, drug_concentration_multiplier, ic50, h
                )

    if perturb_multipliers is not None:
        for channel, value in perturb_multipliers.items():
            if channel in multipliers:
                multipliers[channel] *= value

    return multipliers


def _stimulus(t, pacing_period, p):
    phase = t - pacing_period * math.floor(t / pacing_period)
    if p['stim_start'] <= phase <= p['stim_start'] + p['stim_duration']:
        return -p['stim_amplitude']
    return 0.0


def _currents_from_state(state, p, c):
    Xr1, Xr2, Xs, m, h, j, d, f, f2, fCass, s, r, Ca_i, R_prime, Ca_SR, Ca_ss, Na_i, V, K_i = state

    E_Na = p['R'] * p['T'] * math.log(p['Na_o'] / Na_i) / p['F']
    E_K = p['R'] * p['T'] * math.log(p['K_o'] / K_i) / p['F']
    E_Ks = p['R'] * p['T'] * math.log((p['K_o'] + p['Na_o'] * p['P_kna']) / (p['P_kna'] * Na_i + K_i)) / p['F']
    E_Ca = 0.5 * p['R'] * p['T'] * math.log(p['Ca_o'] / Ca_i) / p['F']

    alpha_K1 = 0.1 / (1 + 6.14421235333e-06 * _safe_exp(0.06 * V - 0.06 * E_K))
    beta_K1 = (
        0.367879441171 * _safe_exp(0.1 * V - 0.1 * E_K)
        + 3.06060402008 * _safe_exp(0.0002 * V - 0.0002 * E_K)
    ) / (1 + _safe_exp(0.5 * E_K - 0.5 * V))
    i_K1 = 0.430331482912 * p['g_K1'] * c['IK1'] * math.sqrt(p['K_o']) * (V - E_K) * alpha_K1 / (alpha_K1 + beta_K1)
    i_Kr = 0.430331482912 * p['g_Kr'] * c['IKr'] * math.sqrt(p['K_o']) * (V - E_K) * Xr1 * Xr2
    i_Ks = p['g_Ks'] * c['IKs'] * Xs ** 2 * (V - E_Ks)
    i_Na = p['g_Na'] * c['INa'] * m ** 3 * (V - E_Na) * h * j
    i_b_Na = p['g_bna'] * c['IbNa'] * (V - E_Na)

    exponent = p['F'] * (-30 + 2 * V) / (p['R'] * p['T'])
    if abs(V - 15.0) < 1e-7:
        diff = 1e-7
        exp_term = _safe_exp(2 * p['F'] * diff / (p['R'] * p['T']))
        i_CaL = (
            p['g_CaL'] * c['ICaL'] * 4 * diff * p['F'] ** 2
            * (-p['Ca_o'] + 0.25 * Ca_ss * exp_term)
            * d * f * f2 * fCass / (p['R'] * p['T'] * (-1 + exp_term))
        )
    else:
        exp_term = _safe_exp(exponent)
        i_CaL = (
            4 * p['g_CaL'] * c['ICaL'] * p['F'] ** 2 * (V - 15)
            * (-p['Ca_o'] + 0.25 * Ca_ss * exp_term)
            * d * f * f2 * fCass / (p['R'] * p['T'] * (-1 + exp_term))
        )

    i_b_Ca = p['g_bca'] * c['IbCa'] * (V - E_Ca)
    i_to = p['g_to'] * c['Ito'] * (V - E_K) * r * s
    i_NaK = (
        p['K_o'] * p['P_NaK'] * c['INaK'] * Na_i
        / ((p['K_mNa'] + Na_i) * (p['K_mk'] + p['K_o'])
           * (1 + 0.0353 * _safe_exp(-p['F'] * V / (p['R'] * p['T']))
              + 0.1245 * _safe_exp(-0.1 * p['F'] * V / (p['R'] * p['T']))))
    )
    i_NaCa = (
        p['K_NaCa'] * c['INaCa']
        * (p['Ca_o'] * Na_i ** 3 * _safe_exp(p['F'] * p['gamma'] * V / (p['R'] * p['T']))
           - p['alpha'] * p['Na_o'] ** 3 * Ca_i * _safe_exp(p['F'] * (-1 + p['gamma']) * V / (p['R'] * p['T'])))
        / ((1 + p['K_sat'] * _safe_exp(p['F'] * (-1 + p['gamma']) * V / (p['R'] * p['T'])))
           * (p['Ca_o'] + p['Km_Ca']) * (p['Km_Nai'] ** 3 + p['Na_o'] ** 3))
    )
    i_p_Ca = p['g_pCa'] * c['IpCa'] * Ca_i / (p['K_pCa'] + Ca_i)
    i_p_K = p['g_pK'] * c['IpK'] * (V - E_K) / (1 + 65.4052157419 * _safe_exp(-0.167224080268 * V))

    i_up = p['Vmax_up'] * c['Iup'] / (1 + p['K_up'] ** 2 / Ca_i ** 2)
    i_leak = p['V_leak'] * c['Ileak'] * (Ca_SR - Ca_i)
    i_xfer = p['V_xfer'] * c['Ixfer'] * (Ca_ss - Ca_i)
    kcasr = p['max_sr'] - (p['max_sr'] - p['min_sr']) / (1 + p['EC'] ** 2 / Ca_SR ** 2)
    k1 = p['k1_prime'] / kcasr
    O = Ca_ss ** 2 * R_prime * k1 / (p['k3'] + Ca_ss ** 2 * k1)
    i_rel = p['V_rel'] * c['Irel'] * (Ca_SR - Ca_ss) * O

    return {
        'INa': i_Na, 'Ito': i_to, 'ICaL': i_CaL, 'IKs': i_Ks,
        'IpK': i_p_K, 'INaK': i_NaK, 'IKr': i_Kr, 'INaCa': i_NaCa,
        'IK1': i_K1, 'IbCa': i_b_Ca, 'IpCa': i_p_Ca, 'IbNa': i_b_Na,
        'Iup': i_up, 'Ileak': i_leak, 'Ixfer': i_xfer, 'Irel': i_rel,
    }


def _rhs(t, state, p, c, pacing_period):
    Xr1, Xr2, Xs, m, h, j, d, f, f2, fCass, s, r, Ca_i, R_prime, Ca_SR, Ca_ss, Na_i, V, K_i = state
    currents = _currents_from_state(state, p, c)
    values = np.zeros(19, dtype=float)

    xr1_inf = 1.0 / (1 + _safe_exp(-26 / 7 - V / 7))
    alpha_xr1 = 450 / (1 + _safe_exp(-9 / 2 - V / 10))
    beta_xr1 = 6 / (1 + 13.5813245226 * _safe_exp(0.0869565217391 * V))
    values[0] = (-Xr1 + xr1_inf) / (alpha_xr1 * beta_xr1)

    xr2_inf = 1.0 / (1 + _safe_exp(11 / 3 + V / 24))
    alpha_xr2 = 3 / (1 + _safe_exp(-3 - V / 20))
    beta_xr2 = 1.12 / (1 + _safe_exp(-3 + V / 20))
    values[1] = (-Xr2 + xr2_inf) / (alpha_xr2 * beta_xr2)

    xs_inf = 1.0 / (1 + _safe_exp(-5 / 14 - V / 14))
    alpha_xs = 1400 / math.sqrt(1 + _safe_exp(5 / 6 - V / 6))
    beta_xs = 1.0 / (1 + _safe_exp(-7 / 3 + V / 15))
    values[2] = (-Xs + xs_inf) / (80 + alpha_xs * beta_xs)

    m_inf = (1 + 0.00184221158117 * _safe_exp(-0.110741971207 * V)) ** -2
    alpha_m = 1.0 / (1 + _safe_exp(-12 - V / 5))
    beta_m = 0.1 / (1 + _safe_exp(7 + V / 5)) + 0.1 / (1 + _safe_exp(-1 / 4 + V / 200))
    values[3] = (-m + m_inf) / (alpha_m * beta_m)

    h_inf = (1 + 15212.5932857 * _safe_exp(0.134589502019 * V)) ** -2
    if V < -40:
        alpha_h = 4.43126792958e-07 * _safe_exp(-0.147058823529 * V)
        beta_h = 310000 * _safe_exp(0.3485 * V) + 2.7 * _safe_exp(0.079 * V)
        alpha_j = (
            (37.78 + V)
            * (-25428 * _safe_exp(0.2444 * V) - 6.948e-06 * _safe_exp(-0.04391 * V))
            / (1 + 50262745826.0 * _safe_exp(0.311 * V))
        )
        beta_j = 0.02424 * _safe_exp(-0.01052 * V) / (1 + 0.0039608683399 * _safe_exp(-0.1378 * V))
    else:
        alpha_h = 0.0
        beta_h = 0.77 / (0.13 + 0.0497581410839 * _safe_exp(-0.0900900900901 * V))
        alpha_j = 0.0
        beta_j = 0.6 * _safe_exp(0.057 * V) / (1 + 0.0407622039784 * _safe_exp(-0.1 * V))
    values[4] = (-h + h_inf) / (1.0 / (alpha_h + beta_h))
    values[5] = (-j + h_inf) / (1.0 / (alpha_j + beta_j))

    d_inf = 1.0 / (1 + 0.344153786865 * _safe_exp(-0.133333333333 * V))
    alpha_d = 0.25 + 1.4 / (1 + _safe_exp(-35 / 13 - V / 13))
    beta_d = 1.4 / (1 + _safe_exp(1 + V / 5))
    gamma_d = 1.0 / (1 + _safe_exp(5 / 2 - V / 20))
    values[6] = (-d + d_inf) / (alpha_d * beta_d + gamma_d)

    f_inf = 1.0 / (1 + _safe_exp(20 / 7 + V / 7))
    tau_f = 20 + 180 / (1 + _safe_exp(3 + V / 10)) + 200 / (1 + _safe_exp(13 / 10 - V / 10)) + 1102.5 * _safe_exp(-(27 + V) ** 2 / 225)
    values[7] = (-f + f_inf) / tau_f

    f2_inf = 0.33 + 0.67 / (1 + _safe_exp(5 + V / 7))
    tau_f2 = 31 / (1 + _safe_exp(5 / 2 - V / 10)) + 80 / (1 + _safe_exp(3 + V / 10)) + 562 * _safe_exp(-(27 + V) ** 2 / 240)
    values[8] = (-f2 + f2_inf) / tau_f2

    fCass_inf = 0.4 + 0.6 / (1 + 400.0 * Ca_ss ** 2)
    tau_fCass = 2 + 80 / (1 + 400.0 * Ca_ss ** 2)
    values[9] = (-fCass + fCass_inf) / tau_fCass

    values[10] = (-s + 1.0 / (1 + _safe_exp(4 + V / 5))) / (
        3 + 5 / (1 + _safe_exp(-4 + V / 5)) + 85 * _safe_exp(-(45 + V) ** 2 / 320)
    )
    values[11] = (-r + 1.0 / (1 + _safe_exp(10 / 3 - V / 6))) / (
        0.8 + 9.5 * _safe_exp(-(40 + V) ** 2 / 1800)
    )

    Ca_i_bufc = 1.0 / (1 + p['Buf_c'] * p['K_buf_c'] / (p['K_buf_c'] + Ca_i) ** 2)
    Ca_sr_bufsr = 1.0 / (1 + p['Buf_sr'] * p['K_buf_sr'] / (p['K_buf_sr'] + Ca_SR) ** 2)
    Ca_ss_bufss = 1.0 / (1 + p['Buf_ss'] * p['K_buf_ss'] / (p['K_buf_ss'] + Ca_ss) ** 2)
    kcasr = p['max_sr'] - (p['max_sr'] - p['min_sr']) / (1 + p['EC'] ** 2 / Ca_SR ** 2)
    k1 = p['k1_prime'] / kcasr
    k2 = p['k2_prime'] * kcasr

    values[12] = (
        p['V_sr'] * (-currents['Iup'] + currents['Ileak']) / p['V_c']
        - p['Cm'] * (-2 * currents['INaCa'] + currents['IbCa'] + currents['IpCa']) / (2 * p['F'] * p['V_c'])
        + currents['Ixfer']
    ) * Ca_i_bufc
    values[13] = p['k4'] * (1 - R_prime) - Ca_ss * R_prime * k2
    values[14] = (-currents['Ileak'] - currents['Irel'] + currents['Iup']) * Ca_sr_bufsr
    values[15] = (
        p['V_sr'] * currents['Irel'] / p['V_ss']
        - p['V_c'] * currents['Ixfer'] / p['V_ss']
        - p['Cm'] * currents['ICaL'] / (2 * p['F'] * p['V_ss'])
    ) * Ca_ss_bufss
    values[16] = p['Cm'] * (-currents['INa'] - currents['IbNa'] - 3 * currents['INaCa'] - 3 * currents['INaK']) / (p['F'] * p['V_c'])

    i_stim = _stimulus(t, pacing_period, p)
    values[17] = -(
        currents['ICaL'] + currents['IK1'] + currents['IKr'] + currents['IKs']
        + currents['INa'] + currents['INaCa'] + currents['INaK'] + i_stim
        + currents['IbCa'] + currents['IbNa'] + currents['IpCa'] + currents['IpK']
        + currents['Ito']
    )
    values[18] = p['Cm'] * (
        -currents['IK1'] - currents['IKr'] - currents['IKs'] - i_stim
        - currents['IpK'] - currents['Ito'] + 2 * currents['INaK']
    ) / (p['F'] * p['V_c'])

    return values


def run_tnnp_simulation(
    drug_name,
    pacing_period,
    drug_concentration_multiplier,
    total_time=None,
    perturb_multipliers=None,
    initial_wait=20000,
    record_pre=100,
    record_post=500,
    record_interval=0.1,
    integration_method='BDF',
):
    if total_time is None:
        total_time = initial_wait + record_post

    p = _base_parameters()
    c = _conductance_multipliers(drug_name, drug_concentration_multiplier, perturb_multipliers)

    record_start = initial_wait - record_pre
    record_stop = initial_wait + record_post
    sample_count = int(round((record_stop - record_start) / record_interval)) + 1
    t_eval = record_start + np.arange(sample_count) * record_interval
    t_eval = t_eval[(t_eval >= 0) & (t_eval <= total_time + 1e-9)]

    sol = solve_ivp(
        lambda t, y: _rhs(t, y, p, c, pacing_period),
        (0.0, float(total_time)),
        MATLAB_INITIAL_STATE.copy(),
        method=integration_method,
        t_eval=t_eval,
        rtol=1e-3,
        atol=1e-6,
        max_step=1.0,
    )
    if not sol.success:
        raise RuntimeError(f"TNNP integration failed: {sol.message}")

    recorded_time = (sol.t - record_start).tolist()
    states = sol.y.T
    recorded_voltage = states[:, 17].tolist()
    recorded_currents = {name: [] for name in CURRENT_NAMES}
    for state in states:
        currents = _currents_from_state(state, p, c)
        for name in CURRENT_NAMES:
            recorded_currents[name].append(currents[name])

    return recorded_time, recorded_voltage, recorded_currents


if __name__ == '__main__':
    t, v, c = run_tnnp_simulation('Amiodarone I', 1000, 5)
    print(f"Voltage at end: {v[-1]}")
