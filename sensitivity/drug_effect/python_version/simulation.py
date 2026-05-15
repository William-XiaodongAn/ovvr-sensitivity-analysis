import math
import pickle

import numpy as np
from scipy.integrate import solve_ivp


CURRENT_NAMES = [
    'INa', 'Ito', 'ICaL', 'IKs', 'IpK', 'INaK',
    'IKr', 'INaCa', 'IK1', 'IbCa', 'IpCa', 'IbNa',
]

MATLAB_INITIAL_STATE = np.array([
    0.000206466599002, 0.473142295376, 0.00322170878365,
    0.00164959539136, 0.750058790356, 0.749677890643,
    3.28817457136e-05, 0.977249616825, 0.999410380975,
    0.999972733836, 0.999997921133, 2.34167244267e-08,
    0.000104622621021, 0.989176724937, 3.50685455674,
    0.000212791419736, 9.79776167522, -85.4001369649,
    135.733295852,
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


def _rhs_with_stimulus_current(state, p, c, i_stim):
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


def _rhs(t, state, p, c, pacing_period):
    return _rhs_with_stimulus_current(state, p, c, _stimulus(t, pacing_period, p))


def _integrate_forward_euler(rhs, initial_state, total_time, t_eval, dt):
    if dt <= 0:
        raise ValueError('Forward Euler integration step must be positive')

    state = np.array(initial_state, dtype=float).copy()
    states = []
    times = []
    next_eval = 0
    num_steps = int(math.ceil(total_time / dt))

    for step in range(num_steps + 1):
        t = min(step * dt, total_time)
        while next_eval < len(t_eval) and t + 1e-9 >= t_eval[next_eval]:
            times.append(float(t_eval[next_eval]))
            states.append(state.copy())
            next_eval += 1

        if t >= total_time:
            break

        step_dt = min(dt, total_time - t)
        state = state + step_dt * rhs(t, state)
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f'Forward Euler integration produced a non-finite state at t={t + step_dt:g} ms')

    return np.array(times), np.array(states)


def _rush_larsen_gate_step(state, dt):
    state = state.copy()
    V = state[17]
    Ca_ss = state[15]

    gate_specs = []
    gate_specs.append((0, 1.0 / (1 + _safe_exp(-26 / 7 - V / 7)),
                       450 / (1 + _safe_exp(-9 / 2 - V / 10)) * 6 / (1 + 13.5813245226 * _safe_exp(0.0869565217391 * V))))
    gate_specs.append((1, 1.0 / (1 + _safe_exp(11 / 3 + V / 24)),
                       3 / (1 + _safe_exp(-3 - V / 20)) * 1.12 / (1 + _safe_exp(-3 + V / 20))))
    alpha_xs = 1400 / math.sqrt(1 + _safe_exp(5 / 6 - V / 6))
    beta_xs = 1.0 / (1 + _safe_exp(-7 / 3 + V / 15))
    gate_specs.append((2, 1.0 / (1 + _safe_exp(-5 / 14 - V / 14)), 80 + alpha_xs * beta_xs))
    alpha_m = 1.0 / (1 + _safe_exp(-12 - V / 5))
    beta_m = 0.1 / (1 + _safe_exp(7 + V / 5)) + 0.1 / (1 + _safe_exp(-1 / 4 + V / 200))
    gate_specs.append((3, (1 + 0.00184221158117 * _safe_exp(-0.110741971207 * V)) ** -2, alpha_m * beta_m))

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
    gate_specs.append((4, h_inf, 1.0 / (alpha_h + beta_h)))
    gate_specs.append((5, h_inf, 1.0 / (alpha_j + beta_j)))

    alpha_d = 0.25 + 1.4 / (1 + _safe_exp(-35 / 13 - V / 13))
    beta_d = 1.4 / (1 + _safe_exp(1 + V / 5))
    gamma_d = 1.0 / (1 + _safe_exp(5 / 2 - V / 20))
    gate_specs.append((6, 1.0 / (1 + 0.344153786865 * _safe_exp(-0.133333333333 * V)), alpha_d * beta_d + gamma_d))
    gate_specs.append((7, 1.0 / (1 + _safe_exp(20 / 7 + V / 7)),
                       20 + 180 / (1 + _safe_exp(3 + V / 10)) + 200 / (1 + _safe_exp(13 / 10 - V / 10)) + 1102.5 * _safe_exp(-(27 + V) ** 2 / 225)))
    gate_specs.append((8, 0.33 + 0.67 / (1 + _safe_exp(5 + V / 7)),
                       31 / (1 + _safe_exp(5 / 2 - V / 10)) + 80 / (1 + _safe_exp(3 + V / 10)) + 562 * _safe_exp(-(27 + V) ** 2 / 240)))
    gate_specs.append((9, 0.4 + 0.6 / (1 + 400.0 * Ca_ss ** 2),
                       2 + 80 / (1 + 400.0 * Ca_ss ** 2)))
    gate_specs.append((10, 1.0 / (1 + _safe_exp(4 + V / 5)),
                       3 + 5 / (1 + _safe_exp(-4 + V / 5)) + 85 * _safe_exp(-(45 + V) ** 2 / 320)))
    gate_specs.append((11, 1.0 / (1 + _safe_exp(10 / 3 - V / 6)),
                       0.8 + 9.5 * _safe_exp(-(40 + V) ** 2 / 1800)))

    for idx, steady, tau in gate_specs:
        state[idx] = steady - (steady - state[idx]) * math.exp(-dt / tau)
    return state


def _integrate_rush_larsen_forward(rhs, initial_state, total_time, t_eval, dt):
    if dt <= 0:
        raise ValueError('Fixed-step integration step must be positive')

    state = np.array(initial_state, dtype=float).copy()
    states = []
    times = []
    next_eval = 0
    num_steps = int(math.ceil(total_time / dt))

    for step in range(num_steps + 1):
        t = min(step * dt, total_time)
        while next_eval < len(t_eval) and t + 1e-9 >= t_eval[next_eval]:
            times.append(float(t_eval[next_eval]))
            states.append(state.copy())
            next_eval += 1

        if t >= total_time:
            break

        step_dt = min(dt, total_time - t)
        state = _rush_larsen_gate_step(state, step_dt)
        rates = rhs(t, state)
        state[12:19] = state[12:19] + step_dt * rates[12:19]
        if not np.all(np.isfinite(state)):
            raise RuntimeError(f'Rush-Larsen/forward integration produced a non-finite state at t={t + step_dt:g} ms')

    return np.array(times), np.array(states)


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
    integration_step=None,
    initial_state=None,
):
    if total_time is None:
        total_time = initial_wait + record_post

    p = _base_parameters()
    c = _conductance_multipliers(drug_name, drug_concentration_multiplier, perturb_multipliers)
    if initial_state is None:
        initial_state = MATLAB_INITIAL_STATE

    record_start = initial_wait - record_pre
    record_stop = initial_wait + record_post
    sample_count = int(round((record_stop - record_start) / record_interval)) + 1
    t_eval = record_start + np.arange(sample_count) * record_interval
    t_eval = t_eval[(t_eval >= 0) & (t_eval <= total_time + 1e-9)]

    method = integration_method.lower()
    if method in {'forward_euler', 'euler'}:
        step_dt = record_interval if integration_step is None else integration_step
        sol_t, states = _integrate_forward_euler(
            lambda t, y: _rhs(t, y, p, c, pacing_period),
            initial_state,
            float(total_time),
            t_eval,
            step_dt,
        )
    elif method in {'rush_larsen_forward', 'rush-larsen-forward', 'forward-newton', 'forward_newton', 'simple'}:
        step_dt = record_interval if integration_step is None else integration_step
        sol_t, states = _integrate_rush_larsen_forward(
            lambda t, y: _rhs(t, y, p, c, pacing_period),
            initial_state,
            float(total_time),
            t_eval,
            step_dt,
        )
    else:
        sol = solve_ivp(
            lambda t, y: _rhs(t, y, p, c, pacing_period),
            (0.0, float(total_time)),
            np.array(initial_state, dtype=float).copy(),
            method=integration_method,
            t_eval=t_eval,
            rtol=1e-3,
            atol=1e-6,
            max_step=1.0,
        )
        if not sol.success:
            raise RuntimeError(f"TNNP integration failed: {sol.message}")
        sol_t = sol.t
        states = sol.y.T

    recorded_time = (sol_t - record_start).tolist()
    recorded_voltage = states[:, 17].tolist()
    recorded_currents = {name: [] for name in CURRENT_NAMES}
    for state in states:
        currents = _currents_from_state(state, p, c)
        for name in CURRENT_NAMES:
            recorded_currents[name].append(currents[name])

    return recorded_time, recorded_voltage, recorded_currents


def _point_to_index(point, nx, ny, name):
    if len(point) != 2:
        raise ValueError(f'{name} must contain exactly two values')
    x, y = float(point[0]), float(point[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        return (
            int(round(y * (ny - 1))),
            int(round(x * (nx - 1))),
        )
    ix = int(round(x))
    iy = int(round(y))
    if 0 <= ix < nx and 0 <= iy < ny:
        return iy, ix
    raise ValueError(f'{name} must be normalized to [0, 1]^2 or valid grid indices')


def _point_mask(point, radius, nx, ny, name):
    center_y, center_x = _point_to_index(point, nx, ny, name)
    if radius <= 0:
        mask = np.zeros((ny, nx), dtype=bool)
        mask[center_y, center_x] = True
        return mask

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y)
    px = center_x / (nx - 1) if nx > 1 else 0.0
    py = center_y / (ny - 1) if ny > 1 else 0.0
    mask = (xx - px) ** 2 + (yy - py) ** 2 <= radius ** 2
    if not np.any(mask):
        mask[center_y, center_x] = True
    return mask


def _html_2d_laplacian(voltage, dx, dy):
    padded = np.pad(voltage, 1, mode='edge')
    center = padded[1:-1, 1:-1]
    east = padded[1:-1, 2:]
    west = padded[1:-1, :-2]
    north = padded[:-2, 1:-1]
    south = padded[2:, 1:-1]
    northeast = padded[:-2, 2:]
    northwest = padded[:-2, :-2]
    southwest = padded[2:, :-2]
    southeast = padded[2:, 2:]

    cddx = 1.0 / dx ** 2
    cddy = 1.0 / dy ** 2
    gamma = 1.0 / 3.0
    return (
        (1.0 - gamma) * (
            (east - 2.0 * center + west) * cddx
            + (north - 2.0 * center + south) * cddy
        )
        + gamma * 0.5 * (
            northeast + northwest + southwest + southeast - 4.0 * center
        ) * (cddx + cddy)
    )


def _torch_module():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('PyTorch backend requested, but torch is not installed') from exc
    return torch


def _torch_auto_device(torch):
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _torch_exp(torch, x):
    max_value = 80.0 if x.dtype in (torch.float16, torch.float32) else 700.0
    return torch.exp(torch.clamp(x, -max_value, max_value))


def _torch_currents_from_state(torch, state, p, c):
    Xr1 = state[..., 0]
    Xr2 = state[..., 1]
    Xs = state[..., 2]
    m = state[..., 3]
    h = state[..., 4]
    j = state[..., 5]
    d = state[..., 6]
    f = state[..., 7]
    f2 = state[..., 8]
    fCass = state[..., 9]
    s = state[..., 10]
    r = state[..., 11]
    Ca_i = state[..., 12]
    R_prime = state[..., 13]
    Ca_SR = state[..., 14]
    Ca_ss = state[..., 15]
    Na_i = state[..., 16]
    V = state[..., 17]
    K_i = state[..., 18]

    E_Na = p['R'] * p['T'] * torch.log(p['Na_o'] / Na_i) / p['F']
    E_K = p['R'] * p['T'] * torch.log(p['K_o'] / K_i) / p['F']
    E_Ks = p['R'] * p['T'] * torch.log((p['K_o'] + p['Na_o'] * p['P_kna']) / (p['P_kna'] * Na_i + K_i)) / p['F']
    E_Ca = 0.5 * p['R'] * p['T'] * torch.log(p['Ca_o'] / Ca_i) / p['F']

    alpha_K1 = 0.1 / (1 + 6.14421235333e-06 * _torch_exp(torch, 0.06 * V - 0.06 * E_K))
    beta_K1 = (
        0.367879441171 * _torch_exp(torch, 0.1 * V - 0.1 * E_K)
        + 3.06060402008 * _torch_exp(torch, 0.0002 * V - 0.0002 * E_K)
    ) / (1 + _torch_exp(torch, 0.5 * E_K - 0.5 * V))
    i_K1 = 0.430331482912 * p['g_K1'] * c['IK1'] * math.sqrt(p['K_o']) * (V - E_K) * alpha_K1 / (alpha_K1 + beta_K1)
    i_Kr = 0.430331482912 * p['g_Kr'] * c['IKr'] * math.sqrt(p['K_o']) * (V - E_K) * Xr1 * Xr2
    i_Ks = p['g_Ks'] * c['IKs'] * Xs ** 2 * (V - E_Ks)
    i_Na = p['g_Na'] * c['INa'] * m ** 3 * (V - E_Na) * h * j
    i_b_Na = p['g_bna'] * c['IbNa'] * (V - E_Na)

    exponent = p['F'] * (-30 + 2 * V) / (p['R'] * p['T'])
    exp_term = _torch_exp(torch, exponent)
    i_CaL_regular = (
        4 * p['g_CaL'] * c['ICaL'] * p['F'] ** 2 * (V - 15)
        * (-p['Ca_o'] + 0.25 * Ca_ss * exp_term)
        * d * f * f2 * fCass / (p['R'] * p['T'] * (-1 + exp_term))
    )
    diff = 1e-7
    exp_term_near = _torch_exp(torch, torch.as_tensor(2 * p['F'] * diff / (p['R'] * p['T']), dtype=V.dtype, device=V.device))
    i_CaL_near = (
        p['g_CaL'] * c['ICaL'] * 4 * diff * p['F'] ** 2
        * (-p['Ca_o'] + 0.25 * Ca_ss * exp_term_near)
        * d * f * f2 * fCass / (p['R'] * p['T'] * (-1 + exp_term_near))
    )
    i_CaL = torch.where(torch.abs(V - 15.0) < 1e-7, i_CaL_near, i_CaL_regular)

    i_b_Ca = p['g_bca'] * c['IbCa'] * (V - E_Ca)
    i_to = p['g_to'] * c['Ito'] * (V - E_K) * r * s
    i_NaK = (
        p['K_o'] * p['P_NaK'] * c['INaK'] * Na_i
        / ((p['K_mNa'] + Na_i) * (p['K_mk'] + p['K_o'])
           * (1 + 0.0353 * _torch_exp(torch, -p['F'] * V / (p['R'] * p['T']))
              + 0.1245 * _torch_exp(torch, -0.1 * p['F'] * V / (p['R'] * p['T']))))
    )
    i_NaCa = (
        p['K_NaCa'] * c['INaCa']
        * (p['Ca_o'] * Na_i ** 3 * _torch_exp(torch, p['F'] * p['gamma'] * V / (p['R'] * p['T']))
           - p['alpha'] * p['Na_o'] ** 3 * Ca_i * _torch_exp(torch, p['F'] * (-1 + p['gamma']) * V / (p['R'] * p['T'])))
        / ((1 + p['K_sat'] * _torch_exp(torch, p['F'] * (-1 + p['gamma']) * V / (p['R'] * p['T'])))
           * (p['Ca_o'] + p['Km_Ca']) * (p['Km_Nai'] ** 3 + p['Na_o'] ** 3))
    )
    i_p_Ca = p['g_pCa'] * c['IpCa'] * Ca_i / (p['K_pCa'] + Ca_i)
    i_p_K = p['g_pK'] * c['IpK'] * (V - E_K) / (1 + 65.4052157419 * _torch_exp(torch, -0.167224080268 * V))

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


def _torch_rhs_with_stimulus_current(torch, state, p, c, i_stim):
    Xr1 = state[..., 0]
    Xr2 = state[..., 1]
    Xs = state[..., 2]
    m = state[..., 3]
    h = state[..., 4]
    j = state[..., 5]
    d = state[..., 6]
    f = state[..., 7]
    f2 = state[..., 8]
    fCass = state[..., 9]
    s = state[..., 10]
    r = state[..., 11]
    Ca_i = state[..., 12]
    R_prime = state[..., 13]
    Ca_SR = state[..., 14]
    Ca_ss = state[..., 15]
    V = state[..., 17]
    values = torch.zeros_like(state)
    currents = _torch_currents_from_state(torch, state, p, c)

    xr1_inf = 1.0 / (1 + _torch_exp(torch, -26 / 7 - V / 7))
    alpha_xr1 = 450 / (1 + _torch_exp(torch, -9 / 2 - V / 10))
    beta_xr1 = 6 / (1 + 13.5813245226 * _torch_exp(torch, 0.0869565217391 * V))
    values[..., 0] = (-Xr1 + xr1_inf) / (alpha_xr1 * beta_xr1)

    xr2_inf = 1.0 / (1 + _torch_exp(torch, 11 / 3 + V / 24))
    alpha_xr2 = 3 / (1 + _torch_exp(torch, -3 - V / 20))
    beta_xr2 = 1.12 / (1 + _torch_exp(torch, -3 + V / 20))
    values[..., 1] = (-Xr2 + xr2_inf) / (alpha_xr2 * beta_xr2)

    xs_inf = 1.0 / (1 + _torch_exp(torch, -5 / 14 - V / 14))
    alpha_xs = 1400 / torch.sqrt(1 + _torch_exp(torch, 5 / 6 - V / 6))
    beta_xs = 1.0 / (1 + _torch_exp(torch, -7 / 3 + V / 15))
    values[..., 2] = (-Xs + xs_inf) / (80 + alpha_xs * beta_xs)

    m_inf = (1 + 0.00184221158117 * _torch_exp(torch, -0.110741971207 * V)) ** -2
    alpha_m = 1.0 / (1 + _torch_exp(torch, -12 - V / 5))
    beta_m = 0.1 / (1 + _torch_exp(torch, 7 + V / 5)) + 0.1 / (1 + _torch_exp(torch, -1 / 4 + V / 200))
    values[..., 3] = (-m + m_inf) / (alpha_m * beta_m)

    h_inf = (1 + 15212.5932857 * _torch_exp(torch, 0.134589502019 * V)) ** -2
    alpha_h_low = 4.43126792958e-07 * _torch_exp(torch, -0.147058823529 * V)
    beta_h_low = 310000 * _torch_exp(torch, 0.3485 * V) + 2.7 * _torch_exp(torch, 0.079 * V)
    alpha_j_low = (
        (37.78 + V)
        * (-25428 * _torch_exp(torch, 0.2444 * V) - 6.948e-06 * _torch_exp(torch, -0.04391 * V))
        / (1 + 50262745826.0 * _torch_exp(torch, 0.311 * V))
    )
    beta_j_low = 0.02424 * _torch_exp(torch, -0.01052 * V) / (1 + 0.0039608683399 * _torch_exp(torch, -0.1378 * V))
    alpha_h_high = torch.zeros_like(V)
    beta_h_high = 0.77 / (0.13 + 0.0497581410839 * _torch_exp(torch, -0.0900900900901 * V))
    alpha_j_high = torch.zeros_like(V)
    beta_j_high = 0.6 * _torch_exp(torch, 0.057 * V) / (1 + 0.0407622039784 * _torch_exp(torch, -0.1 * V))
    low_v = V < -40
    alpha_h = torch.where(low_v, alpha_h_low, alpha_h_high)
    beta_h = torch.where(low_v, beta_h_low, beta_h_high)
    alpha_j = torch.where(low_v, alpha_j_low, alpha_j_high)
    beta_j = torch.where(low_v, beta_j_low, beta_j_high)
    values[..., 4] = (-h + h_inf) / (1.0 / (alpha_h + beta_h))
    values[..., 5] = (-j + h_inf) / (1.0 / (alpha_j + beta_j))

    d_inf = 1.0 / (1 + 0.344153786865 * _torch_exp(torch, -0.133333333333 * V))
    alpha_d = 0.25 + 1.4 / (1 + _torch_exp(torch, -35 / 13 - V / 13))
    beta_d = 1.4 / (1 + _torch_exp(torch, 1 + V / 5))
    gamma_d = 1.0 / (1 + _torch_exp(torch, 5 / 2 - V / 20))
    values[..., 6] = (-d + d_inf) / (alpha_d * beta_d + gamma_d)

    f_inf = 1.0 / (1 + _torch_exp(torch, 20 / 7 + V / 7))
    tau_f = 20 + 180 / (1 + _torch_exp(torch, 3 + V / 10)) + 200 / (1 + _torch_exp(torch, 13 / 10 - V / 10)) + 1102.5 * _torch_exp(torch, -(27 + V) ** 2 / 225)
    values[..., 7] = (-f + f_inf) / tau_f

    f2_inf = 0.33 + 0.67 / (1 + _torch_exp(torch, 5 + V / 7))
    tau_f2 = 31 / (1 + _torch_exp(torch, 5 / 2 - V / 10)) + 80 / (1 + _torch_exp(torch, 3 + V / 10)) + 562 * _torch_exp(torch, -(27 + V) ** 2 / 240)
    values[..., 8] = (-f2 + f2_inf) / tau_f2

    fCass_inf = 0.4 + 0.6 / (1 + 400.0 * Ca_ss ** 2)
    tau_fCass = 2 + 80 / (1 + 400.0 * Ca_ss ** 2)
    values[..., 9] = (-fCass + fCass_inf) / tau_fCass
    values[..., 10] = (-s + 1.0 / (1 + _torch_exp(torch, 4 + V / 5))) / (
        3 + 5 / (1 + _torch_exp(torch, -4 + V / 5)) + 85 * _torch_exp(torch, -(45 + V) ** 2 / 320)
    )
    values[..., 11] = (-r + 1.0 / (1 + _torch_exp(torch, 10 / 3 - V / 6))) / (
        0.8 + 9.5 * _torch_exp(torch, -(40 + V) ** 2 / 1800)
    )

    Ca_i_bufc = 1.0 / (1 + p['Buf_c'] * p['K_buf_c'] / (p['K_buf_c'] + Ca_i) ** 2)
    Ca_sr_bufsr = 1.0 / (1 + p['Buf_sr'] * p['K_buf_sr'] / (p['K_buf_sr'] + Ca_SR) ** 2)
    Ca_ss_bufss = 1.0 / (1 + p['Buf_ss'] * p['K_buf_ss'] / (p['K_buf_ss'] + Ca_ss) ** 2)
    kcasr = p['max_sr'] - (p['max_sr'] - p['min_sr']) / (1 + p['EC'] ** 2 / Ca_SR ** 2)
    k1 = p['k1_prime'] / kcasr
    k2 = p['k2_prime'] * kcasr

    values[..., 12] = (
        p['V_sr'] * (-currents['Iup'] + currents['Ileak']) / p['V_c']
        - p['Cm'] * (-2 * currents['INaCa'] + currents['IbCa'] + currents['IpCa']) / (2 * p['F'] * p['V_c'])
        + currents['Ixfer']
    ) * Ca_i_bufc
    values[..., 13] = p['k4'] * (1 - R_prime) - Ca_ss * R_prime * k2
    values[..., 14] = (-currents['Ileak'] - currents['Irel'] + currents['Iup']) * Ca_sr_bufsr
    values[..., 15] = (
        p['V_sr'] * currents['Irel'] / p['V_ss']
        - p['V_c'] * currents['Ixfer'] / p['V_ss']
        - p['Cm'] * currents['ICaL'] / (2 * p['F'] * p['V_ss'])
    ) * Ca_ss_bufss
    values[..., 16] = p['Cm'] * (-currents['INa'] - currents['IbNa'] - 3 * currents['INaCa'] - 3 * currents['INaK']) / (p['F'] * p['V_c'])
    values[..., 17] = -(
        currents['ICaL'] + currents['IK1'] + currents['IKr'] + currents['IKs']
        + currents['INa'] + currents['INaCa'] + currents['INaK'] + i_stim
        + currents['IbCa'] + currents['IbNa'] + currents['IpCa'] + currents['IpK']
        + currents['Ito']
    )
    values[..., 18] = p['Cm'] * (
        -currents['IK1'] - currents['IKr'] - currents['IKs'] - i_stim
        - currents['IpK'] - currents['Ito'] + 2 * currents['INaK']
    ) / (p['F'] * p['V_c'])
    return values


def _torch_rush_larsen_gate_step(torch, state, dt):
    next_state = state.clone()
    V = state[..., 17]
    Ca_ss = state[..., 15]

    gate_specs = []
    gate_specs.append((0, 1.0 / (1 + _torch_exp(torch, -26 / 7 - V / 7)),
                       450 / (1 + _torch_exp(torch, -9 / 2 - V / 10)) * 6 / (1 + 13.5813245226 * _torch_exp(torch, 0.0869565217391 * V))))
    gate_specs.append((1, 1.0 / (1 + _torch_exp(torch, 11 / 3 + V / 24)),
                       3 / (1 + _torch_exp(torch, -3 - V / 20)) * 1.12 / (1 + _torch_exp(torch, -3 + V / 20))))
    alpha_xs = 1400 / torch.sqrt(1 + _torch_exp(torch, 5 / 6 - V / 6))
    beta_xs = 1.0 / (1 + _torch_exp(torch, -7 / 3 + V / 15))
    gate_specs.append((2, 1.0 / (1 + _torch_exp(torch, -5 / 14 - V / 14)), 80 + alpha_xs * beta_xs))
    alpha_m = 1.0 / (1 + _torch_exp(torch, -12 - V / 5))
    beta_m = 0.1 / (1 + _torch_exp(torch, 7 + V / 5)) + 0.1 / (1 + _torch_exp(torch, -1 / 4 + V / 200))
    gate_specs.append((3, (1 + 0.00184221158117 * _torch_exp(torch, -0.110741971207 * V)) ** -2, alpha_m * beta_m))

    h_inf = (1 + 15212.5932857 * _torch_exp(torch, 0.134589502019 * V)) ** -2
    alpha_h_low = 4.43126792958e-07 * _torch_exp(torch, -0.147058823529 * V)
    beta_h_low = 310000 * _torch_exp(torch, 0.3485 * V) + 2.7 * _torch_exp(torch, 0.079 * V)
    alpha_j_low = (
        (37.78 + V)
        * (-25428 * _torch_exp(torch, 0.2444 * V) - 6.948e-06 * _torch_exp(torch, -0.04391 * V))
        / (1 + 50262745826.0 * _torch_exp(torch, 0.311 * V))
    )
    beta_j_low = 0.02424 * _torch_exp(torch, -0.01052 * V) / (1 + 0.0039608683399 * _torch_exp(torch, -0.1378 * V))
    beta_h_high = 0.77 / (0.13 + 0.0497581410839 * _torch_exp(torch, -0.0900900900901 * V))
    beta_j_high = 0.6 * _torch_exp(torch, 0.057 * V) / (1 + 0.0407622039784 * _torch_exp(torch, -0.1 * V))
    low_v = V < -40
    alpha_h = torch.where(low_v, alpha_h_low, torch.zeros_like(V))
    beta_h = torch.where(low_v, beta_h_low, beta_h_high)
    alpha_j = torch.where(low_v, alpha_j_low, torch.zeros_like(V))
    beta_j = torch.where(low_v, beta_j_low, beta_j_high)
    gate_specs.append((4, h_inf, 1.0 / (alpha_h + beta_h)))
    gate_specs.append((5, h_inf, 1.0 / (alpha_j + beta_j)))

    alpha_d = 0.25 + 1.4 / (1 + _torch_exp(torch, -35 / 13 - V / 13))
    beta_d = 1.4 / (1 + _torch_exp(torch, 1 + V / 5))
    gamma_d = 1.0 / (1 + _torch_exp(torch, 5 / 2 - V / 20))
    gate_specs.append((6, 1.0 / (1 + 0.344153786865 * _torch_exp(torch, -0.133333333333 * V)), alpha_d * beta_d + gamma_d))
    gate_specs.append((7, 1.0 / (1 + _torch_exp(torch, 20 / 7 + V / 7)),
                       20 + 180 / (1 + _torch_exp(torch, 3 + V / 10)) + 200 / (1 + _torch_exp(torch, 13 / 10 - V / 10)) + 1102.5 * _torch_exp(torch, -(27 + V) ** 2 / 225)))
    gate_specs.append((8, 0.33 + 0.67 / (1 + _torch_exp(torch, 5 + V / 7)),
                       31 / (1 + _torch_exp(torch, 5 / 2 - V / 10)) + 80 / (1 + _torch_exp(torch, 3 + V / 10)) + 562 * _torch_exp(torch, -(27 + V) ** 2 / 240)))
    gate_specs.append((9, 0.4 + 0.6 / (1 + 400.0 * Ca_ss ** 2),
                       2 + 80 / (1 + 400.0 * Ca_ss ** 2)))
    gate_specs.append((10, 1.0 / (1 + _torch_exp(torch, 4 + V / 5)),
                       3 + 5 / (1 + _torch_exp(torch, -4 + V / 5)) + 85 * _torch_exp(torch, -(45 + V) ** 2 / 320)))
    gate_specs.append((11, 1.0 / (1 + _torch_exp(torch, 10 / 3 - V / 6)),
                       0.8 + 9.5 * _torch_exp(torch, -(40 + V) ** 2 / 1800)))

    for idx, steady, tau in gate_specs:
        next_state[..., idx] = steady - (steady - state[..., idx]) * torch.exp(-dt / tau)
    return next_state


def _torch_html_2d_laplacian(torch, voltage, dx, dy):
    center = voltage
    east = torch.empty_like(voltage)
    west = torch.empty_like(voltage)
    north = torch.empty_like(voltage)
    south = torch.empty_like(voltage)
    northeast = torch.empty_like(voltage)
    northwest = torch.empty_like(voltage)
    southwest = torch.empty_like(voltage)
    southeast = torch.empty_like(voltage)

    east[:, :-1] = voltage[:, 1:]
    east[:, -1] = voltage[:, -1]
    west[:, 1:] = voltage[:, :-1]
    west[:, 0] = voltage[:, 0]
    north[1:, :] = voltage[:-1, :]
    north[0, :] = voltage[0, :]
    south[:-1, :] = voltage[1:, :]
    south[-1, :] = voltage[-1, :]

    northeast[1:, :-1] = voltage[:-1, 1:]
    northeast[0, :-1] = voltage[0, 1:]
    northeast[:, -1] = north[:, -1]
    northwest[1:, 1:] = voltage[:-1, :-1]
    northwest[0, 1:] = voltage[0, :-1]
    northwest[:, 0] = north[:, 0]
    southwest[:-1, 1:] = voltage[1:, :-1]
    southwest[-1, 1:] = voltage[-1, :-1]
    southwest[:, 0] = south[:, 0]
    southeast[:-1, :-1] = voltage[1:, 1:]
    southeast[-1, :-1] = voltage[-1, 1:]
    southeast[:, -1] = south[:, -1]

    cddx = 1.0 / dx ** 2
    cddy = 1.0 / dy ** 2
    gamma = 1.0 / 3.0
    return (
        (1.0 - gamma) * (
            (east - 2.0 * center + west) * cddx
            + (north - 2.0 * center + south) * cddy
        )
        + gamma * 0.5 * (
            northeast + northwest + southwest + southeast - 4.0 * center
        ) * (cddx + cddy)
    )


def _default_2d_step(record_interval, nx, ny, diff_coef):
    if diff_coef <= 0:
        return record_interval
    dx = 1.0 / nx
    dy = 1.0 / ny
    stable_dt = 0.2 * min(dx, dy) ** 2 / diff_coef
    return min(record_interval, stable_dt)


def run_tnnp_simulation_2d(
    drug_name,
    pacing_period,
    drug_concentration_multiplier,
    total_time=None,
    perturb_multipliers=None,
    initial_wait=20000,
    record_pre=100,
    record_post=500,
    record_interval=0.1,
    integration_step=None,
    initial_state=None,
    nx=32,
    ny=32,
    pacemaker_point=(0.05, 0.5),
    pacemaker_radius=0.03,
    measurement_point=(0.5, 0.5),
    diff_coef=0.001,
    c_m=1.0,
    activation_threshold=-40.0,
    recovery_tolerance=1.0,
    max_activation_wait=1000.0,
    return_voltage_maps=False,
    backend='cpu',
    torch_device=None,
):
    """Run a fixed-step 2D TNNP simulation on [0, 1]^2.

    The voltage diffusion term follows the 9-point stencil used by the local
    2D-TNNP HTML shader. Ionic states are shared with the 1D implementation.
    """
    if backend not in {'cpu', 'torch', 'auto'}:
        raise ValueError("backend must be one of 'cpu', 'torch', or 'auto'")
    if backend in {'torch', 'auto'}:
        try:
            torch = _torch_module()
            selected_device = torch.device(torch_device) if torch_device else _torch_auto_device(torch)
            if backend == 'torch' or selected_device.type in {'cuda', 'mps'}:
                return _run_tnnp_simulation_2d_torch(
                    drug_name,
                    pacing_period,
                    drug_concentration_multiplier,
                    total_time=total_time,
                    perturb_multipliers=perturb_multipliers,
                    initial_wait=initial_wait,
                    record_pre=record_pre,
                    record_post=record_post,
                    record_interval=record_interval,
                    integration_step=integration_step,
                    initial_state=initial_state,
                    nx=nx,
                    ny=ny,
                    pacemaker_point=pacemaker_point,
                    pacemaker_radius=pacemaker_radius,
                    measurement_point=measurement_point,
                    diff_coef=diff_coef,
                    c_m=c_m,
                    activation_threshold=activation_threshold,
                    recovery_tolerance=recovery_tolerance,
                    max_activation_wait=max_activation_wait,
                    return_voltage_maps=return_voltage_maps,
                    device=selected_device,
                )
        except RuntimeError:
            if backend == 'torch':
                raise

    if nx < 2 or ny < 2:
        raise ValueError('nx and ny must both be at least 2 for 2D simulation')
    if pacemaker_radius < 0:
        raise ValueError('pacemaker_radius must be non-negative')
    if diff_coef < 0:
        raise ValueError('diff_coef must be non-negative')
    if c_m <= 0:
        raise ValueError('c_m must be positive')
    if recovery_tolerance < 0:
        raise ValueError('recovery_tolerance must be non-negative')

    if total_time is None:
        total_time = initial_wait + max_activation_wait

    p = _base_parameters()
    c = _conductance_multipliers(drug_name, drug_concentration_multiplier, perturb_multipliers)
    if initial_state is None:
        initial_state = MATLAB_INITIAL_STATE

    step_dt = _default_2d_step(record_interval, nx, ny, diff_coef) if integration_step is None else integration_step
    if step_dt <= 0:
        raise ValueError('2D integration step must be positive')

    state = np.broadcast_to(np.array(initial_state, dtype=float), (ny, nx, 19)).copy()
    pace_mask = _point_mask(pacemaker_point, pacemaker_radius, nx, ny, 'pacemaker_point')
    measure_y, measure_x = _point_to_index(measurement_point, nx, ny, 'measurement_point')
    dx = 1.0 / nx
    dy = 1.0 / ny

    recorded_time = []
    recorded_voltage = []
    recorded_currents = {name: [] for name in CURRENT_NAMES}
    recorded_voltage_maps = [] if return_voltage_maps else None
    next_record_time = float(initial_wait)
    measuring = False
    activated = False
    recovered = False
    resting_voltage = None
    peak_voltage = float(state[measure_y, measure_x, 17])
    previous_measured_voltage = float(state[measure_y, measure_x, 17])
    num_steps = int(math.ceil(total_time / step_dt))

    for step in range(num_steps + 1):
        t = min(step * step_dt, total_time)
        measured_voltage = float(state[measure_y, measure_x, 17])

        if not measuring and t + 1e-9 >= initial_wait:
            measuring = True
            resting_voltage = measured_voltage
            peak_voltage = measured_voltage

        if measuring and not activated and previous_measured_voltage < activation_threshold <= measured_voltage:
            activated = True

        if activated:
            peak_voltage = max(peak_voltage, measured_voltage)
            if (
                peak_voltage > activation_threshold
                and previous_measured_voltage > resting_voltage + recovery_tolerance
                and measured_voltage <= resting_voltage + recovery_tolerance
            ):
                recovered = True

        while measuring and t + 1e-9 >= next_record_time:
            measured_state = state[measure_y, measure_x]
            measured_currents = _currents_from_state(measured_state, p, c)
            recorded_time.append(float(next_record_time - initial_wait))
            recorded_voltage.append(float(measured_state[17]))
            for name in CURRENT_NAMES:
                recorded_currents[name].append(measured_currents[name])
            if return_voltage_maps:
                recorded_voltage_maps.append(state[:, :, 17].copy())
            next_record_time += record_interval

        if recovered:
            if not recorded_time or recorded_time[-1] < t - initial_wait:
                measured_state = state[measure_y, measure_x]
                measured_currents = _currents_from_state(measured_state, p, c)
                recorded_time.append(float(t - initial_wait))
                recorded_voltage.append(float(measured_state[17]))
                for name in CURRENT_NAMES:
                    recorded_currents[name].append(measured_currents[name])
                if return_voltage_maps:
                    recorded_voltage_maps.append(state[:, :, 17].copy())
            break

        if t >= total_time:
            break

        dt = min(step_dt, total_time - t)
        old_voltage = state[:, :, 17].copy()
        laplacian = _html_2d_laplacian(old_voltage, dx, dy)
        stimulus = _stimulus(t, pacing_period, p)

        next_state = state.copy()
        for y in range(ny):
            for x in range(nx):
                cell_state = _rush_larsen_gate_step(state[y, x], dt)
                i_stim = stimulus if pace_mask[y, x] else 0.0
                rates = _rhs_with_stimulus_current(cell_state, p, c, i_stim)
                cell_state[12:19] = cell_state[12:19] + dt * rates[12:19]
                cell_state[17] += dt * ((1.0 / c_m - 1.0) * rates[17] + diff_coef * laplacian[y, x])
                next_state[y, x] = cell_state

        if not np.all(np.isfinite(next_state)):
            raise RuntimeError(f'2D integration produced a non-finite state at t={t + dt:g} ms')
        state = next_state
        previous_measured_voltage = measured_voltage

    if return_voltage_maps:
        return recorded_time, recorded_voltage, recorded_currents, np.array(recorded_voltage_maps), state.copy()
    return recorded_time, recorded_voltage, recorded_currents


def _run_tnnp_simulation_2d_torch(
    drug_name,
    pacing_period,
    drug_concentration_multiplier,
    total_time=None,
    perturb_multipliers=None,
    initial_wait=20000,
    record_pre=100,
    record_post=500,
    record_interval=0.1,
    integration_step=None,
    initial_state=None,
    nx=32,
    ny=32,
    pacemaker_point=(0.05, 0.5),
    pacemaker_radius=0.03,
    measurement_point=(0.5, 0.5),
    diff_coef=0.001,
    c_m=1.0,
    activation_threshold=-40.0,
    recovery_tolerance=1.0,
    max_activation_wait=1000.0,
    return_voltage_maps=False,
    device=None,
):
    torch = _torch_module()
    if device is None:
        device = _torch_auto_device(torch)
    dtype = torch.float32 if device.type in {'cuda', 'mps'} else torch.float64

    if nx < 2 or ny < 2:
        raise ValueError('nx and ny must both be at least 2 for 2D simulation')
    if pacemaker_radius < 0:
        raise ValueError('pacemaker_radius must be non-negative')
    if diff_coef < 0:
        raise ValueError('diff_coef must be non-negative')
    if c_m <= 0:
        raise ValueError('c_m must be positive')
    if recovery_tolerance < 0:
        raise ValueError('recovery_tolerance must be non-negative')

    if total_time is None:
        total_time = initial_wait + max_activation_wait

    p = _base_parameters()
    c = _conductance_multipliers(drug_name, drug_concentration_multiplier, perturb_multipliers)
    if initial_state is None:
        initial_state = MATLAB_INITIAL_STATE

    step_dt = _default_2d_step(record_interval, nx, ny, diff_coef) if integration_step is None else integration_step
    if step_dt <= 0:
        raise ValueError('2D integration step must be positive')

    initial_state_tensor = torch.as_tensor(initial_state, dtype=dtype, device=device)
    state = initial_state_tensor.view(1, 1, 19).expand(ny, nx, 19).clone()
    pace_mask_np = _point_mask(pacemaker_point, pacemaker_radius, nx, ny, 'pacemaker_point')
    pace_mask = torch.as_tensor(pace_mask_np, dtype=torch.bool, device=device)
    measure_y, measure_x = _point_to_index(measurement_point, nx, ny, 'measurement_point')
    dx = 1.0 / nx
    dy = 1.0 / ny

    recorded_time = []
    recorded_voltage = []
    recorded_currents = {name: [] for name in CURRENT_NAMES}
    recorded_voltage_maps = [] if return_voltage_maps else None
    next_record_time = float(initial_wait)
    measuring = False
    activated = False
    recovered = False
    resting_voltage = None
    peak_voltage = float(state[measure_y, measure_x, 17].detach().cpu())
    previous_measured_voltage = peak_voltage
    num_steps = int(math.ceil(total_time / step_dt))

    for step in range(num_steps + 1):
        t = min(step * step_dt, total_time)
        measured_voltage = float(state[measure_y, measure_x, 17].detach().cpu())

        if not measuring and t + 1e-9 >= initial_wait:
            measuring = True
            resting_voltage = measured_voltage
            peak_voltage = measured_voltage

        if measuring and not activated and previous_measured_voltage < activation_threshold <= measured_voltage:
            activated = True

        if activated:
            peak_voltage = max(peak_voltage, measured_voltage)
            if (
                peak_voltage > activation_threshold
                and previous_measured_voltage > resting_voltage + recovery_tolerance
                and measured_voltage <= resting_voltage + recovery_tolerance
            ):
                recovered = True

        while measuring and t + 1e-9 >= next_record_time:
            measured_state = state[measure_y, measure_x].detach().cpu().numpy().astype(float)
            measured_currents = _currents_from_state(measured_state, p, c)
            recorded_time.append(float(next_record_time - initial_wait))
            recorded_voltage.append(float(measured_state[17]))
            for name in CURRENT_NAMES:
                recorded_currents[name].append(measured_currents[name])
            if return_voltage_maps:
                recorded_voltage_maps.append(state[:, :, 17].detach().cpu().numpy().copy())
            next_record_time += record_interval

        if recovered:
            if not recorded_time or recorded_time[-1] < t - initial_wait:
                measured_state = state[measure_y, measure_x].detach().cpu().numpy().astype(float)
                measured_currents = _currents_from_state(measured_state, p, c)
                recorded_time.append(float(t - initial_wait))
                recorded_voltage.append(float(measured_state[17]))
                for name in CURRENT_NAMES:
                    recorded_currents[name].append(measured_currents[name])
                if return_voltage_maps:
                    recorded_voltage_maps.append(state[:, :, 17].detach().cpu().numpy().copy())
            break

        if t >= total_time:
            break

        dt = min(step_dt, total_time - t)
        old_voltage = state[:, :, 17]
        laplacian = _torch_html_2d_laplacian(torch, old_voltage, dx, dy)
        stimulus = _stimulus(t, pacing_period, p)
        i_stim = torch.where(
            pace_mask,
            torch.as_tensor(stimulus, dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        )

        next_state = _torch_rush_larsen_gate_step(torch, state, dt)
        rates = _torch_rhs_with_stimulus_current(torch, next_state, p, c, i_stim)
        next_state[..., 12:19] = next_state[..., 12:19] + dt * rates[..., 12:19]
        next_state[..., 17] = next_state[..., 17] + dt * ((1.0 / c_m - 1.0) * rates[..., 17] + diff_coef * laplacian)

        if not bool(torch.isfinite(next_state).all().detach().cpu()):
            raise RuntimeError(f'PyTorch 2D integration produced a non-finite state at t={t + dt:g} ms')
        state = next_state
        previous_measured_voltage = measured_voltage

    if return_voltage_maps:
        return recorded_time, recorded_voltage, recorded_currents, np.array(recorded_voltage_maps), state.detach().cpu().numpy().copy()
    return recorded_time, recorded_voltage, recorded_currents


def generate_tnnp_2d_measurement_gif(
    output_path,
    drug_name,
    pacing_period,
    drug_concentration_multiplier,
    total_time=None,
    perturb_multipliers=None,
    initial_wait=20000,
    integration_step=None,
    initial_state=None,
    nx=32,
    ny=32,
    pacemaker_point=(0.05, 0.5),
    pacemaker_radius=0.03,
    measurement_point=(0.5, 0.5),
    diff_coef=0.001,
    c_m=1.0,
    activation_threshold=-40.0,
    recovery_tolerance=1.0,
    max_activation_wait=1000.0,
    frame_interval=1.0,
    fps=20,
    vmin=-90.0,
    vmax=40.0,
    colormap='turbo',
    pixel_scale=8,
    mark_points=True,
    backend='auto',
    torch_device=None,
):
    """Generate a GIF of the 2D voltage maps during the measured APD window."""
    if frame_interval <= 0:
        raise ValueError('frame_interval must be positive')
    if fps <= 0:
        raise ValueError('fps must be positive')
    if pixel_scale < 1:
        raise ValueError('pixel_scale must be at least 1')

    import imageio.v2 as imageio
    from matplotlib import colormaps

    time, voltage, currents, voltage_maps, _ = run_tnnp_simulation_2d(
        drug_name,
        pacing_period,
        drug_concentration_multiplier,
        total_time=total_time,
        perturb_multipliers=perturb_multipliers,
        initial_wait=initial_wait,
        record_pre=0,
        record_post=max_activation_wait,
        record_interval=frame_interval,
        integration_step=integration_step,
        initial_state=initial_state,
        nx=nx,
        ny=ny,
        pacemaker_point=pacemaker_point,
        pacemaker_radius=pacemaker_radius,
        measurement_point=measurement_point,
        diff_coef=diff_coef,
        c_m=c_m,
        activation_threshold=activation_threshold,
        recovery_tolerance=recovery_tolerance,
        max_activation_wait=max_activation_wait,
        return_voltage_maps=True,
        backend=backend,
        torch_device=torch_device,
    )
    if len(voltage_maps) == 0:
        raise RuntimeError('No 2D frames were recorded for the measurement window')

    cmap = colormaps[colormap]
    frames = []
    pace_y, pace_x = _point_to_index(pacemaker_point, nx, ny, 'pacemaker_point')
    measure_y, measure_x = _point_to_index(measurement_point, nx, ny, 'measurement_point')
    marker_radius = max(1, int(round(pixel_scale * 0.5)))

    def draw_marker(frame, y, x, color):
        cy = y * pixel_scale + pixel_scale // 2
        cx = x * pixel_scale + pixel_scale // 2
        y0 = max(0, cy - marker_radius)
        y1 = min(frame.shape[0], cy + marker_radius + 1)
        x0 = max(0, cx - marker_radius)
        x1 = min(frame.shape[1], cx + marker_radius + 1)
        frame[y0:y1, x0:x1] = color

    for voltage_map in voltage_maps:
        normalized = np.clip((voltage_map - vmin) / (vmax - vmin), 0.0, 1.0)
        frame = (cmap(normalized)[:, :, :3] * 255).astype(np.uint8)
        frame = np.repeat(np.repeat(frame, pixel_scale, axis=0), pixel_scale, axis=1)
        if mark_points:
            draw_marker(frame, pace_y, pace_x, np.array([255, 64, 64], dtype=np.uint8))
            draw_marker(frame, measure_y, measure_x, np.array([255, 255, 255], dtype=np.uint8))
        frames.append(frame)

    imageio.mimsave(output_path, frames, duration=1.0 / fps)
    return {
        'output_path': output_path,
        'frame_count': len(frames),
        'time': time,
        'voltage': voltage,
        'currents': currents,
        'duration_ms': time[-1] if time else 0.0,
        'voltage_min': min(voltage) if voltage else None,
        'voltage_max': max(voltage) if voltage else None,
        'activated': bool(voltage and max(voltage) >= activation_threshold),
        'recovered': bool(voltage and voltage[-1] <= voltage[0] + recovery_tolerance),
    }


def run_tnnp_final_state(
    drug_name,
    pacing_period,
    drug_concentration_multiplier,
    total_time,
    perturb_multipliers=None,
    integration_method='BDF',
    integration_step=None,
    initial_state=None,
):
    p = _base_parameters()
    c = _conductance_multipliers(drug_name, drug_concentration_multiplier, perturb_multipliers)
    if initial_state is None:
        initial_state = MATLAB_INITIAL_STATE

    t_eval = np.array([float(total_time)])
    method = integration_method.lower()
    if method in {'forward_euler', 'euler'}:
        step_dt = 0.1 if integration_step is None else integration_step
        _, states = _integrate_forward_euler(
            lambda t, y: _rhs(t, y, p, c, pacing_period),
            initial_state,
            float(total_time),
            t_eval,
            step_dt,
        )
    elif method in {'rush_larsen_forward', 'rush-larsen-forward', 'forward-newton', 'forward_newton', 'simple'}:
        step_dt = 0.1 if integration_step is None else integration_step
        _, states = _integrate_rush_larsen_forward(
            lambda t, y: _rhs(t, y, p, c, pacing_period),
            initial_state,
            float(total_time),
            t_eval,
            step_dt,
        )
    else:
        sol = solve_ivp(
            lambda t, y: _rhs(t, y, p, c, pacing_period),
            (0.0, float(total_time)),
            np.array(initial_state, dtype=float).copy(),
            method=integration_method,
            t_eval=t_eval,
            rtol=1e-3,
            atol=1e-6,
            max_step=1.0,
        )
        if not sol.success:
            raise RuntimeError(f"TNNP integration failed: {sol.message}")
        states = sol.y.T

    return states[-1].copy()


if __name__ == '__main__':
    t, v, c = run_tnnp_simulation('Amiodarone I', 1000, 5)
    print(f"Voltage at end: {v[-1]}")
