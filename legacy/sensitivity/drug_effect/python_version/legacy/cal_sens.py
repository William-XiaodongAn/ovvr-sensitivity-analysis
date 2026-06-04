import argparse
import numpy as np
import sys
from simulation import run_tnnp_simulation

CURRENT_NAMES = ['INa', 'Ito', 'ICaL', 'IKs', 'IpK', 'INaK', 'IKr', 'INaCa', 'IK1', 'IbCa', 'IpCa', 'IbNa']
RECORDED_DT = 0.1

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
    """First HTML stage: baseline pacing run that records voltage and currents."""
    _, voltage, currents = run_tnnp_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=None,
    )
    return np.array(voltage), currents

def run_perturbation_voltage_simulation(drug_name, pacing_period, drug_concentration, perturb_multipliers):
    """Second HTML stage: separate perturbation run that records voltage only."""
    _, voltage, _ = run_tnnp_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=perturb_multipliers,
    )
    return np.array(voltage)

def cal_identifiability(drug_name, pacing_period, drug_concentration, show_progress=False):
    # To calculate relative identifiability index for all currents
    current_names = CURRENT_NAMES

    progress = ProgressBar(total=1 + 12 * 2, enabled=show_progress)
    v, base_c = run_baseline_pacing_simulation(drug_name, pacing_period, drug_concentration)
    progress.update('baseline simulation')
    
    mask = np.array(v) > -100
    
    matrix = np.zeros((np.sum(mask), len(current_names)))
    for i, current_name in enumerate(current_names):
        matrix[:, i] = np.array(base_c[current_name])[mask]
        
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    V = Vt
    
    def get_APD_from_voltage(voltage,threshhold = 0.4,dt = RECORDED_DT):
        voltage = np.array(voltage)
        if np.all(voltage < threshhold):
            return [0.0]
        if np.all(voltage > threshhold):
            return [0.0]
        APD = []
        def linear_interpolation(y0,y1,x0,x1,x):
            if x1 == x0: return y0
            return (y0 * (x1-x) + y1 * (x-x0)) / (x1-x0)
        APD_value = 0
        start_apd = False
        for j in range(len(voltage)):
            if j != 0 and start_apd == False and voltage[j] > threshhold and j+1 < len(voltage) and voltage[j+1] > threshhold:
                start_apd = True
                APD_start_error = linear_interpolation(1,0,voltage[j],voltage[j-1],threshhold)
                APD_start_error = 1 - APD_start_error
            if start_apd == True and voltage[j] < threshhold and j+1 < len(voltage) and voltage[j+1] < threshhold: 
                start_apd = False
                APD_end_error = linear_interpolation(0,-1,voltage[j],voltage[j-1],threshhold)
                APD_value += (APD_start_error + APD_end_error) * dt
                APD.append(APD_value)
                APD_value = 0
                APD_start_error = -100
                APD_end_error = -100 
            if start_apd == True:
                APD_value += dt
        if len(voltage) > 2 and voltage[1] > threshhold and voltage[2] > threshhold and len(APD) > 0:
            APD.pop(0)
        if len(APD) == 0:
            return [0.0]
        return APD

    def dvdt_max(voltages):
        return np.max(np.gradient(voltages)) / RECORDED_DT
        
    def H_test(v_star, v_bar):
        dt = RECORDED_DT
        apd_star_07 = get_APD_from_voltage(v_star, 0.7, dt)[0]
        apd_bar_07 = get_APD_from_voltage(v_bar, 0.7, dt)[0]
        H1 = abs(apd_star_07 - apd_bar_07) / (abs(apd_star_07) + 1e-9)
        
        apd_star_05 = get_APD_from_voltage(v_star, 0.5, dt)[0]
        apd_bar_05 = get_APD_from_voltage(v_bar, 0.5, dt)[0]
        H2 = abs(apd_star_05 - apd_bar_05) / (abs(apd_star_05) + 1e-9)
        
        apd_star_02 = get_APD_from_voltage(v_star, 0.2, dt)[0]
        apd_bar_02 = get_APD_from_voltage(v_bar, 0.2, dt)[0]
        H3 = abs(apd_star_02 - apd_bar_02) / (abs(apd_star_02) + 1e-9)
        
        H4 = abs(dvdt_max(v_star) - dvdt_max(v_bar)) / (abs(dvdt_max(v_star)) + 1e-9)
        H5 = np.linalg.norm(np.array(v_star) - np.array(v_bar)) / (np.linalg.norm(v_star) + 1e-9)
        
        return H1 + H2 + H3 + H4 + H5

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
                v_bar = get_voltage(vec, epsilon)
                progress.update(f'vector {vec}, epsilon {epsilon}')
                if len(v_bar) == 0:
                    continue
                # The original step-3 notebook compares perturbed traces as
                # v_bar[1:], because those files come from the separate
                # sensitivity HTML simulation.
                v_bar = v_bar[1:]
                # Align lengths
                min_len = min(len(v_star), len(v_bar))
                if min_len == 0:
                    continue
                H_value = H_test(v_star[:min_len], v_bar[:min_len])
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
