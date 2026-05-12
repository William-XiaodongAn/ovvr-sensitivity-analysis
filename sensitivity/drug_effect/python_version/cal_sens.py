import argparse
import numpy as np
import sys
from simulation import run_tnnp_simulation

CURRENT_NAMES = ['INa', 'IbNa', 'INaK', 'IKr', 'IpK', 'IKs', 'Ito', 'IK1', 'ICaL', 'INaCa', 'IpCa', 'IbCa']
RECORDED_DT = 0.1
MATLAB_RECORD_PRE = 100
MATLAB_RECORD_POST = 400

class ProgressBar:
    def __init__(self, total, enabled=True, width=32):
        self.total = total
        self.enabled = enabled
        self.width = width
        self.current = 0
        if self.enabled:
            self.render('starting')

    def update(self, label=''):
        self.current += 1
        if self.enabled:
            self.render(label)

    def skip(self, count, label=''):
        self.current += count
        if self.enabled:
            self.render(label)

    def render(self, label=''):
        done = min(self.current, self.total)
        filled = int(self.width * done / self.total)
        bar = '#' * filled + '-' * (self.width - filled)
        percent = 100 * done / self.total
        sys.stderr.write(f'\r[{bar}] {done}/{self.total} {percent:5.1f}% {label}')
        sys.stderr.flush()

    def close(self):
        if self.enabled:
            self.current = self.total
            self.render('done')
            sys.stderr.write('\n')
            sys.stderr.flush()

def get_currents(drug_name, pacing_period, drug_concentration, epsilon_current=None, epsilon_value=0.0):
    perturb_multipliers = None
    if epsilon_current is not None:
        perturb_multipliers = {epsilon_current: 1.0 + epsilon_value}

    _, voltage, currents = run_tnnp_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=perturb_multipliers,
    )
    return voltage, currents

def run_baseline_pacing_simulation(drug_name, pacing_period, drug_concentration):
    """MATLAB run_* stage: baseline run that records voltage and currents."""
    time, voltage, currents = run_tnnp_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=None,
        record_pre=MATLAB_RECORD_PRE,
        record_post=MATLAB_RECORD_POST,
    )
    return np.array(time), np.array(voltage), currents

def run_perturbation_voltage_simulation(drug_name, pacing_period, drug_concentration, perturb_multipliers):
    """MATLAB full_cost stage: separate perturbation run that records voltage only."""
    time, voltage, _ = run_tnnp_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=perturb_multipliers,
        record_pre=MATLAB_RECORD_PRE,
        record_post=MATLAB_RECORD_POST,
    )
    return np.array(time), np.array(voltage)

def cal_identifiability(drug_name, pacing_period, drug_concentration, show_progress=False):
    # To calculate relative identifiability index for all currents
    current_names = CURRENT_NAMES

    progress = ProgressBar(total=1 + 12 * 2, enabled=show_progress)
    t, v, base_c = run_baseline_pacing_simulation(drug_name, pacing_period, drug_concentration)
    progress.update('baseline simulation')
    
    matrix = np.zeros((len(v), len(current_names)))
    for i, current_name in enumerate(current_names):
        matrix[:, i] = np.array(base_c[current_name])
        
    _, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    V = Vt
    
    def compute_apd(voltage, time, factor):
        voltage = np.array(voltage)
        time = np.array(time)
        t_half = np.max(time) / 2
        idx_t_half = np.argmin(np.abs(time - t_half))
        v_max_idx = np.argmax(voltage[:idx_t_half + 1])
        v_max = voltage[v_max_idx]
        v_min = np.min(voltage)
        threshold = v_min + (1 - factor / 100) * (v_max - v_min)

        t_start = 0.0
        for n in range(min(v_max_idx, len(time) - 1)):
            if voltage[n + 1] > threshold and voltage[n] < threshold:
                t_start = time[n] + (time[n + 1] - time[n]) * (threshold - voltage[n]) / (voltage[n + 1] - voltage[n])
                break

        t_end = np.inf
        for n in range(max(1, v_max_idx), len(time)):
            if voltage[n - 1] > threshold and voltage[n] < threshold:
                t_end = time[n] + (time[n - 1] - time[n]) * (threshold - voltage[n]) / (voltage[n - 1] - voltage[n])
                break

        return t_end - t_start

    def dvdt_max(voltages, time):
        return np.max((voltages[1:] - voltages[:-1]) / (time[1:] - time[:-1]))
        
    def safe_relative_difference(value, reference):
        if not np.isfinite(value) or not np.isfinite(reference):
            return np.inf
        return abs(value - reference) / (abs(reference) + 1e-9)

    def H_test(t_star, v_star, t_bar, v_bar):
        min_len = min(len(v_star), len(v_bar), len(t_star), len(t_bar))
        t_star = t_star[:min_len]
        v_star = v_star[:min_len]
        t_bar = t_bar[:min_len]
        v_bar = v_bar[:min_len]

        H = np.linalg.norm(v_bar - v_star) / (np.linalg.norm(v_star) + 1e-9)
        H += safe_relative_difference(dvdt_max(v_bar, t_bar), dvdt_max(v_star, t_star))
        for factor in [30, 50, 80]:
            H += safe_relative_difference(
                compute_apd(v_bar, t_bar, factor),
                compute_apd(v_star, t_star, factor),
            )
        return H

    def get_voltage(vec, epsilon):
        perturb_multipliers = {
            current_name: 1.0 + epsilon * V[vec][i]
            for i, current_name in enumerate(current_names)
        }
        return run_perturbation_voltage_simulation(
            drug_name, pacing_period, drug_concentration, perturb_multipliers
        )

    def calculate_unidentifiable_space():
        vec_lst = list(range(12))
        epsilon_list = [-0.5, 0.5]
        Delta = 0.25
        output = list(range(12))
        v_star = v
        for vec in vec_lst:
            for epsilon_index, epsilon in enumerate(epsilon_list):
                t_bar, v_bar = get_voltage(vec, epsilon)
                progress.update(f'vector {vec}, epsilon {epsilon}')
                if len(v_bar) == 0:
                    continue
                H_value = H_test(t, v_star, t_bar, v_bar)
                if H_value > Delta:
                    output.remove(vec)
                    progress.skip(len(epsilon_list) - epsilon_index - 1, f'vector {vec} resolved')
                    break
        return output

    try:
        unidentifiable_space = calculate_unidentifiable_space()

        def projection_S(v_I):
            ps = np.zeros_like(v_I)
            for i in unidentifiable_space:
                ps += np.inner(v_I, V[i]) / np.inner(V[i], V[i]) * V[i]
            return ps

        identifiability = {}
        for i, current_name in enumerate(current_names):
            v_I = np.zeros(len(S))
            v_I[i] = 1.0
            k = np.linalg.norm(v_I - projection_S(v_I))
            identifiability[current_name] = k

        return dict(sorted(identifiability.items(), key=lambda item: item[1], reverse=True))
    finally:
        progress.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate identifiability index for TNNP model currents.')
    parser.add_argument('--Drug', type=str, required=True, help='Name of the drug')
    parser.add_argument('--pacing_period', type=float, required=True, help='Pacing period in ms')
    parser.add_argument('--drug_concentration', type=float, required=True, help='Drug concentration multiplier (EFTPC_multiplier)')
    parser.add_argument('--no-progress', action='store_true', help='Disable the progress bar')
    
    args = parser.parse_args()
    
    identifiabilities = cal_identifiability(
        args.Drug,
        args.pacing_period,
        args.drug_concentration,
        show_progress=not args.no_progress,
    )
    print("Relative Identifiability Index:")
    for current, val in identifiabilities.items():
        print(f"{current}: {val:.4f}")
