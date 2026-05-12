import argparse
import numpy as np
import os
import csv
from simulation import run_tnnp_simulation

def get_currents(drug_name, pacing_period, drug_concentration, epsilon_current=None, epsilon_value=0.0):
    pass

def cal_identifiability(drug_name, pacing_period, drug_concentration):
    # To calculate relative identifiability index for all currents
    # Instead of simulating, we read the exact CSVs used by JS if possible, to match exactly.
    # The reference is generated from JS traces.
    
    current_names = ['INa', 'Ito', 'ICaL', 'IKs', 'IpK', 'INaK', 'IKr', 'INaCa', 'IK1', 'IbCa', 'IpCa', 'IbNa']
    
    # Check if we can read the baseline simulation from the JS output
    base_file = f"2D-TNNP-pacing-period-{int(pacing_period)}-5xdrug/voltage_TNNP_pacingPeriod_{int(pacing_period)}_{drug_name}.csv"
    
    if os.path.exists(base_file):
        # Read the exact matrix from JS output
        data_dict = {name: [] for name in ['voltage'] + current_names}
        with open(base_file, 'r') as file:
            header_line = file.readline().strip()
            # Read rest
            for line in file:
                if line.startswith("undefined") or not line.strip():
                    continue
                groups = line.strip().split(';')
                all_vals = []
                for g in groups:
                    for v in g.split(','):
                        if v.strip():
                            all_vals.append(float(v.strip()))
                if len(all_vals) >= len(data_dict):
                    data_dict['voltage'].append(all_vals[0])
                    # The mapping in JS output:
                    # 'voltage', 'INa,Ito,ICaL,IKs', 'IpK, INaK, IKr, INaCa', 'IK1, IbCa, IpCa, IbNa'
                    # The order matches current_names exactly.
                    for i, name in enumerate(current_names):
                        data_dict[name].append(all_vals[i+1])
        
        for k in data_dict:
            data_dict[k] = np.array(data_dict[k])
            
        v = data_dict['voltage']
        base_c = data_dict
    else:
        # Fallback to python simulation
        t, v, base_c = run_tnnp_simulation(drug_name, pacing_period, drug_concentration)
    
    mask = np.array(v) > -100
    
    matrix = np.zeros((np.sum(mask), len(current_names)))
    for i, current_name in enumerate(current_names):
        matrix[:, i] = np.array(base_c[current_name])[mask]
        
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    V = Vt
    
    def get_APD_from_voltage(voltage,threshhold = 0.4,dt = 0.1):
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
        dt = 0.1
        return np.max(np.gradient(voltages)) / dt
        
    def H_test(v_star, v_bar):
        dt = 0.1
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

    def get_voltage(vec, epsilon, pacing_period, drug_name):
        path = f"./data-5xdrug/voltage_vector{vec}_epsilon_{epsilon}_pacingPeriod_{int(pacing_period)}_drug_name_{drug_name}.csv"
        voltage = []
        if os.path.exists(path):
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=';')
                for row in csv_reader:
                    if row and len(row) > 1 and row[1].strip() != 'NaN':
                        try:
                            voltage.append(float(row[1]))
                        except ValueError:
                            pass
        return np.array(voltage)

    def calculate_unidentifiable_space(pacing_period, drug_name):
        vec_lst = list(range(12))
        epsilon_list = [-0.5, 0.5]
        Delta = 0.25
        output = list(range(12))
        v_star = v
        for vec in vec_lst:
            for epsilon in epsilon_list:
                v_bar = get_voltage(vec, epsilon, pacing_period, drug_name)
                if len(v_bar) == 0:
                    continue
                # Align lengths
                min_len = min(len(v_star), len(v_bar))
                if min_len == 0:
                    continue
                H_value = H_test(v_star[:min_len], v_bar[:min_len])
                if H_value > Delta:
                    output.remove(vec)
                    break
        return output

    unidentifiable_space = calculate_unidentifiable_space(pacing_period, drug_name)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate identifiability index for TNNP model currents.')
    parser.add_argument('--Drug', type=str, required=True, help='Name of the drug')
    parser.add_argument('--pacing_period', type=float, required=True, help='Pacing period in ms')
    parser.add_argument('--drug_concentration', type=float, required=True, help='Drug concentration multiplier (EFTPC_multiplier)')
    
    args = parser.parse_args()
    
    identifiabilities = cal_identifiability(args.Drug, args.pacing_period, args.drug_concentration)
    print("Relative Identifiability Index:")
    for current, val in identifiabilities.items():
        print(f"{current}: {val:.4f}")
