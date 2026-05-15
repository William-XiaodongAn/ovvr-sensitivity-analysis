import argparse
import multiprocessing
import os
import numpy as np
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.io import loadmat
from simulation import (
    generate_tnnp_2d_measurement_gif,
    run_tnnp_final_state,
    run_tnnp_simulation,
    run_tnnp_simulation_2d,
)

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
        sys.stderr.write(f'\r[{bar}] {done}/{self.total} {percent:5.1f}% {label}\033[K')
        sys.stderr.flush()

    def close(self):
        if self.enabled:
            self.current = self.total
            self.render('done')
            sys.stderr.write('\n')
            sys.stderr.flush()

def describe_run(dimension, integration_method, integration_step, initial_wait, simulation_kwargs):
    if dimension != '2d':
        return f'1d method={integration_method} dt={integration_step} initial_wait={initial_wait:g}ms'
    simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs
    nx = simulation_kwargs.get('nx', '?')
    ny = simulation_kwargs.get('ny', '?')
    backend = simulation_kwargs.get('backend', 'cpu')
    device = simulation_kwargs.get('torch_device') or 'auto'
    pacer = simulation_kwargs.get('pacemaker_point', '?')
    measure = simulation_kwargs.get('measurement_point', '?')
    radius = simulation_kwargs.get('pacemaker_radius', '?')
    return (
        f'2d {nx}x{ny} backend={backend} device={device} dt={integration_step} '
        f'initial_wait={initial_wait:g}ms pacer={pacer} radius={radius} measure={measure}'
    )

def load_initial_state_mat(path):
    state = np.asarray(loadmat(path)['states']).ravel()
    if state.size != 19:
        raise ValueError(f'Expected 19 state values in {path}, got {state.size}')
    return state

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

def projection_identifiability(current_names, singular_values, right_singular_vectors, unidentifiable_space):
    def projection_S(v_I):
        ps = np.zeros_like(v_I)
        for i in unidentifiable_space:
            ps += (
                np.inner(v_I, right_singular_vectors[i])
                / np.inner(right_singular_vectors[i], right_singular_vectors[i])
                * right_singular_vectors[i]
            )
        return ps

    identifiability = {}
    for i, current_name in enumerate(current_names):
        v_I = np.zeros(len(singular_values))
        v_I[i] = 1.0
        k = np.linalg.norm(v_I - projection_S(v_I))
        identifiability[current_name] = k

    return dict(sorted(identifiability.items(), key=lambda item: item[1], reverse=True))

def parse_point(value):
    try:
        x_text, y_text = value.split(',', 1)
        return float(x_text), float(y_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('point must be formatted as x,y') from exc

def auto_worker_count(task_count=24):
    for name in ['SLURM_CPUS_PER_TASK', 'NSLOTS', 'PBS_NP', 'OMP_NUM_THREADS']:
        value = os.environ.get(name)
        if value:
            try:
                return max(1, min(int(value), task_count))
            except ValueError:
                pass

    slurm_node_cpus = os.environ.get('SLURM_JOB_CPUS_PER_NODE')
    if slurm_node_cpus:
        match = re.search(r'\d+', slurm_node_cpus)
        if match:
            return max(1, min(int(match.group()), task_count))

    return max(1, min(os.cpu_count() or 1, task_count))

def run_simulation(
    dimension,
    drug_name,
    pacing_period,
    drug_concentration,
    perturb_multipliers=None,
    initial_state=None,
    integration_method='BDF',
    integration_step=None,
    initial_wait=20000,
    record_pre=MATLAB_RECORD_PRE,
    record_post=MATLAB_RECORD_POST,
    simulation_kwargs=None,
):
    simulation_kwargs = {} if simulation_kwargs is None else simulation_kwargs
    if dimension == '1d':
        return run_tnnp_simulation(
            drug_name,
            pacing_period,
            drug_concentration,
            perturb_multipliers=perturb_multipliers,
            record_pre=record_pre,
            record_post=record_post,
            initial_state=initial_state,
            integration_method=integration_method,
            integration_step=integration_step,
            initial_wait=initial_wait,
        )
    if dimension == '2d':
        return run_tnnp_simulation_2d(
            drug_name,
            pacing_period,
            drug_concentration,
            perturb_multipliers=perturb_multipliers,
            record_pre=record_pre,
            record_post=record_post,
            initial_state=initial_state,
            integration_step=integration_step,
            initial_wait=initial_wait,
            **simulation_kwargs,
        )
    raise ValueError(f'Unknown simulation dimension: {dimension}')

def get_currents(
    drug_name,
    pacing_period,
    drug_concentration,
    epsilon_current=None,
    epsilon_value=0.0,
    initial_state=None,
    integration_method='BDF',
    integration_step=None,
    initial_wait=20000,
    dimension='1d',
    simulation_kwargs=None,
    record_pre=MATLAB_RECORD_PRE,
    record_post=MATLAB_RECORD_POST,
):
    perturb_multipliers = None
    if epsilon_current is not None:
        perturb_multipliers = {epsilon_current: 1.0 + epsilon_value}

    _, voltage, currents = run_simulation(
        dimension,
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=perturb_multipliers,
        initial_state=initial_state,
        integration_method=integration_method,
        integration_step=integration_step,
        initial_wait=initial_wait,
        simulation_kwargs=simulation_kwargs,
        record_pre=record_pre,
        record_post=record_post,
    )
    return voltage, currents

def run_baseline_pacing_simulation(
    drug_name,
    pacing_period,
    drug_concentration,
    initial_state=None,
    integration_method='BDF',
    integration_step=None,
    initial_wait=20000,
    dimension='1d',
    simulation_kwargs=None,
    record_pre=MATLAB_RECORD_PRE,
    record_post=MATLAB_RECORD_POST,
):
    """MATLAB run_* stage: baseline run that records voltage and currents."""
    time, voltage, currents = run_simulation(
        dimension,
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=None,
        record_pre=record_pre,
        record_post=record_post,
        initial_state=initial_state,
        integration_method=integration_method,
        integration_step=integration_step,
        initial_wait=initial_wait,
        simulation_kwargs=simulation_kwargs,
    )
    return np.array(time), np.array(voltage), currents

def run_perturbation_voltage_simulation(
    drug_name,
    pacing_period,
    drug_concentration,
    perturb_multipliers,
    initial_state=None,
    integration_method='BDF',
    integration_step=None,
    initial_wait=20000,
    dimension='1d',
    simulation_kwargs=None,
    record_pre=MATLAB_RECORD_PRE,
    record_post=MATLAB_RECORD_POST,
):
    """MATLAB full_cost stage: separate perturbation run that records voltage only."""
    time, voltage, _ = run_simulation(
        dimension,
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers=perturb_multipliers,
        record_pre=record_pre,
        record_post=record_post,
        initial_state=initial_state,
        integration_method=integration_method,
        integration_step=integration_step,
        initial_wait=initial_wait,
        simulation_kwargs=simulation_kwargs,
    )
    return np.array(time), np.array(voltage)

def _run_perturbation_voltage_task(task):
    (
        vec,
        epsilon,
        singular_vector,
        current_names,
        drug_name,
        pacing_period,
        drug_concentration,
        initial_state,
        integration_method,
        integration_step,
        initial_wait,
        dimension,
        simulation_kwargs,
        record_pre,
        record_post,
    ) = task
    perturb_multipliers = {
        current_name: 1.0 + epsilon * singular_vector[i]
        for i, current_name in enumerate(current_names)
    }
    t_bar, v_bar = run_perturbation_voltage_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        perturb_multipliers,
        initial_state=initial_state,
        integration_method=integration_method,
        integration_step=integration_step,
        initial_wait=initial_wait,
        dimension=dimension,
        simulation_kwargs=simulation_kwargs,
        record_pre=record_pre,
        record_post=record_post,
    )
    return vec, epsilon, t_bar, v_bar

def cal_identifiability(
    drug_name,
    pacing_period,
    drug_concentration,
    show_progress=False,
    initial_state=None,
    integration_method='BDF',
    integration_step=None,
    initial_wait=20000,
    dimension='1d',
    simulation_kwargs=None,
    record_pre=MATLAB_RECORD_PRE,
    record_post=MATLAB_RECORD_POST,
):
    # To calculate relative identifiability index for all currents
    current_names = CURRENT_NAMES

    progress = ProgressBar(total=1 + 12 * 2, enabled=show_progress)
    if show_progress:
        sys.stderr.write(f'\nRun: {describe_run(dimension, integration_method, integration_step, initial_wait, simulation_kwargs)}\n')
        sys.stderr.write('Baseline simulation started...\n')
        sys.stderr.flush()
    baseline_start = time.perf_counter()
    t, v, base_c = run_baseline_pacing_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        initial_state=initial_state,
        integration_method=integration_method,
        integration_step=integration_step,
        initial_wait=initial_wait,
        dimension=dimension,
        simulation_kwargs=simulation_kwargs,
        record_pre=record_pre,
        record_post=record_post,
    )
    baseline_elapsed = time.perf_counter() - baseline_start
    progress.update(f'baseline simulation ({baseline_elapsed:.1f}s)')
    
    matrix = np.zeros((len(v), len(current_names)))
    for i, current_name in enumerate(current_names):
        matrix[:, i] = np.array(base_c[current_name])
        
    _, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    V = Vt

    def get_voltage(vec, epsilon):
        perturb_multipliers = {
            current_name: 1.0 + epsilon * V[vec][i]
            for i, current_name in enumerate(current_names)
        }
        return run_perturbation_voltage_simulation(
            drug_name,
            pacing_period,
            drug_concentration,
            perturb_multipliers,
            initial_state=initial_state,
            integration_method=integration_method,
            integration_step=integration_step,
            initial_wait=initial_wait,
            dimension=dimension,
            simulation_kwargs=simulation_kwargs,
            record_pre=record_pre,
            record_post=record_post,
        )

    def calculate_unidentifiable_space():
        vec_lst = list(range(12))
        epsilon_list = [-0.5, 0.5]
        Delta = 0.25
        output = list(range(12))
        v_star = v
        for vec in vec_lst:
            for epsilon_index, epsilon in enumerate(epsilon_list):
                if show_progress:
                    progress.render(f'vector {vec}, epsilon {epsilon} running')
                perturb_start = time.perf_counter()
                t_bar, v_bar = get_voltage(vec, epsilon)
                perturb_elapsed = time.perf_counter() - perturb_start
                progress.update(f'vector {vec}, epsilon {epsilon} ({perturb_elapsed:.1f}s)')
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
        return projection_identifiability(current_names, S, V, unidentifiable_space)
    finally:
        progress.close()

def cal_identifiability_parallel(
    drug_name,
    pacing_period,
    drug_concentration,
    show_progress=False,
    initial_state=None,
    integration_method='BDF',
    integration_step=None,
    initial_wait=20000,
    max_workers=None,
    dimension='1d',
    simulation_kwargs=None,
    record_pre=MATLAB_RECORD_PRE,
    record_post=MATLAB_RECORD_POST,
):
    """Parallel sensitivity calculation using one process per perturbation run."""
    current_names = CURRENT_NAMES
    if max_workers is None:
        max_workers = auto_worker_count(12 * 2)
    if max_workers < 1:
        raise ValueError('max_workers must be at least 1')

    progress = ProgressBar(total=1 + 12 * 2, enabled=show_progress)
    if show_progress:
        sys.stderr.write(f'\nRun: {describe_run(dimension, integration_method, integration_step, initial_wait, simulation_kwargs)}\n')
        sys.stderr.write(f'Baseline simulation started; perturbation workers={max_workers}...\n')
        sys.stderr.flush()
    baseline_start = time.perf_counter()
    t, v, base_c = run_baseline_pacing_simulation(
        drug_name,
        pacing_period,
        drug_concentration,
        initial_state=initial_state,
        integration_method=integration_method,
        integration_step=integration_step,
        initial_wait=initial_wait,
        dimension=dimension,
        simulation_kwargs=simulation_kwargs,
        record_pre=record_pre,
        record_post=record_post,
    )
    baseline_elapsed = time.perf_counter() - baseline_start
    progress.update(f'baseline simulation ({baseline_elapsed:.1f}s)')

    matrix = np.zeros((len(v), len(current_names)))
    for i, current_name in enumerate(current_names):
        matrix[:, i] = np.array(base_c[current_name])

    _, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    V = Vt

    tasks = []
    for vec in range(12):
        for epsilon in [-0.5, 0.5]:
            tasks.append((
                vec,
                epsilon,
                V[vec].copy(),
                tuple(current_names),
                drug_name,
                pacing_period,
                drug_concentration,
                None if initial_state is None else np.array(initial_state, dtype=float).copy(),
                integration_method,
                integration_step,
                initial_wait,
                dimension,
                simulation_kwargs,
                record_pre,
                record_post,
            ))

    try:
        resolved_vectors = set()
        if max_workers == 1:
            for task in tasks:
                vec_label, epsilon_label = task[0], task[1]
                if show_progress:
                    progress.render(f'vector {vec_label}, epsilon {epsilon_label} running')
                perturb_start = time.perf_counter()
                vec, epsilon, t_bar, v_bar = _run_perturbation_voltage_task(task)
                perturb_elapsed = time.perf_counter() - perturb_start
                progress.update(f'vector {vec}, epsilon {epsilon} ({perturb_elapsed:.1f}s)')
                if len(v_bar) == 0:
                    continue
                H_value = H_test(t, v, t_bar, v_bar)
                if H_value > 0.25:
                    resolved_vectors.add(vec)
        else:
            mp_context = multiprocessing.get_context('fork') if hasattr(os, 'fork') else None
            with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
                if show_progress:
                    sys.stderr.write(f'\nSubmitted {len(tasks)} perturbation simulations...\n')
                    sys.stderr.flush()
                futures = [executor.submit(_run_perturbation_voltage_task, task) for task in tasks]
                for future in as_completed(futures):
                    perturb_finish = time.perf_counter()
                    vec, epsilon, t_bar, v_bar = future.result()
                    progress.update(f'vector {vec}, epsilon {epsilon} finished at {perturb_finish - baseline_start:.1f}s')
                    if len(v_bar) == 0:
                        continue
                    H_value = H_test(t, v, t_bar, v_bar)
                    if H_value > 0.25:
                        resolved_vectors.add(vec)

        unidentifiable_space = [vec for vec in range(12) if vec not in resolved_vectors]
        return projection_identifiability(current_names, S, V, unidentifiable_space)
    finally:
        progress.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate identifiability index for TNNP model currents.')
    parser.add_argument('--Drug', type=str, required=True, help='Name of the drug')
    parser.add_argument('--pacing_period', type=float, required=True, help='Pacing period in ms')
    parser.add_argument('--drug_concentration', type=float, required=True, help='Drug concentration multiplier (EFTPC_multiplier)')
    parser.add_argument('--no-progress', action='store_true', help='Disable the progress bar')
    parser.add_argument('--initial-state-mat', type=str, default=None, help='Optional MATLAB .mat file containing a 19-value states vector')
    parser.add_argument('--integration-method', type=str, default='BDF', help='Integration method: BDF, LSODA, or forward_euler')
    parser.add_argument('--integration-step', type=float, default=None, help='Fixed integration step in ms for forward_euler; defaults to record dt')
    parser.add_argument('--initial-wait', type=float, default=20000, help='Short pre-pacing time in ms used for baseline and perturbation runs')
    parser.add_argument('--equilibrate-wait', type=float, default=None, help='Optional one-time unperturbed pre-pacing time in ms used to create an initial state before sensitivity runs')
    parser.add_argument('--serial', action='store_true', help='Run perturbation simulations in one process instead of the default parallel mode')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel worker processes; defaults to scheduler CPU allocation or min(CPU count, 24)')
    parser.add_argument('--dimension', choices=['1d', '2d'], default='1d', help='Run the original single-cell model or the [0,1]^2 spatial model')
    parser.add_argument('--pacemaker-point', type=parse_point, default=(0.05, 0.5), help='2D pacing point as x,y in [0,1]^2 or grid indices')
    parser.add_argument('--pacemaker-radius', type=float, default=0.03, help='2D pacing radius in normalized map units; 0 means nearest pixel only')
    parser.add_argument('--measurement-point', type=parse_point, default=(0.5, 0.5), help='2D measurement point as x,y in [0,1]^2 or grid indices')
    parser.add_argument('--map-size', type=int, default=32, help='2D square grid size; uses nx=ny=map_size')
    parser.add_argument('--backend', choices=['auto', 'cpu', 'torch'], default='auto', help='2D compute backend: auto uses PyTorch on CUDA/MPS when available')
    parser.add_argument('--torch-device', type=str, default=None, help='Optional PyTorch device override, e.g. cuda, cuda:0, mps, or cpu')
    parser.add_argument('--torch-compile', action='store_true', help='Use torch.compile for the 2D PyTorch stepper on CUDA')
    parser.add_argument('--measurement-gif', nargs='?', const='tnnp_2d_measurement.gif', default=None, help='Generate a 2D APD measurement GIF at the optional output path and exit')
    
    args = parser.parse_args()
    initial_state = load_initial_state_mat(args.initial_state_mat) if args.initial_state_mat else None
    if args.equilibrate_wait is not None:
        sys.stderr.write(f'Equilibrating initial state for {args.equilibrate_wait:g} ms...\n')
        sys.stderr.flush()
        initial_state = run_tnnp_final_state(
            args.Drug,
            args.pacing_period,
            args.drug_concentration,
            total_time=args.equilibrate_wait,
            initial_state=initial_state,
            integration_method=args.integration_method,
            integration_step=args.integration_step,
        )
    
    identifiability_fn = cal_identifiability if args.serial else cal_identifiability_parallel
    identifiability_kwargs = {}
    if not args.serial:
        identifiability_kwargs['max_workers'] = args.workers

    simulation_kwargs = None
    if args.dimension == '2d':
        simulation_kwargs = {
            'nx': args.map_size,
            'ny': args.map_size,
            'pacemaker_point': args.pacemaker_point,
            'pacemaker_radius': args.pacemaker_radius,
            'measurement_point': args.measurement_point,
            'backend': args.backend,
            'torch_device': args.torch_device,
            'torch_compile': args.torch_compile,
        }
        if args.backend in {'auto', 'torch'} and not args.serial and args.workers is None:
            identifiability_kwargs['max_workers'] = 1

    if args.measurement_gif is not None:
        meta = generate_tnnp_2d_measurement_gif(
            args.measurement_gif,
            args.Drug,
            args.pacing_period,
            args.drug_concentration,
            initial_wait=args.initial_wait,
            integration_step=args.integration_step,
            initial_state=initial_state,
            nx=args.map_size,
            ny=args.map_size,
            pacemaker_point=args.pacemaker_point,
            pacemaker_radius=args.pacemaker_radius,
            measurement_point=args.measurement_point,
            backend=args.backend,
            torch_device=args.torch_device,
            torch_compile=args.torch_compile,
        )
        print(f"Saved 2D measurement GIF: {meta['output_path']}")
        print(f"Frames: {meta['frame_count']}")
        print(f"Measured duration [ms]: {meta['duration_ms']:.4f}")
        print(f"Voltage min/max [mV]: {meta['voltage_min']:.4f}, {meta['voltage_max']:.4f}")
        print(f"Activated: {meta['activated']}")
        print(f"Recovered: {meta['recovered']}")
        sys.exit(0)

    identifiabilities = identifiability_fn(
        args.Drug,
        args.pacing_period,
        args.drug_concentration,
        show_progress=not args.no_progress,
        initial_state=initial_state,
        integration_method=args.integration_method,
        integration_step=args.integration_step,
        initial_wait=args.initial_wait,
        dimension=args.dimension,
        simulation_kwargs=simulation_kwargs,
        **identifiability_kwargs,
    )
    print("Relative Identifiability Index:")
    for current, val in identifiabilities.items():
        print(f"{current}: {val:.4f}")
