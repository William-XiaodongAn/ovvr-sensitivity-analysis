"""Run a single TNNP simulation (1D single cell or 2D tissue) and print timing.

Examples
--------
1D single cell (adaptive BDF):
    python run_tnnp.py --mode 1d --drug "Amiodarone I" --pacing-period 1000 \
        --concentration 5

2D tissue on a dx-defined grid, GPU if available, pacing at the left edge:
    python run_tnnp.py --mode 2d --drug "Amiodarone I" --pacing-period 1000 \
        --concentration 5 --dx 0.03125 --dt 0.05 --pace-point 0.05,0.5 \
        --backend auto

The heavy lifting (TNNP kinetics, integrators, torch GPU 2D stepper) lives in
simulation.py; this file is only a thin, well-timed CLI in front of it.
"""

import argparse
import os
import sys
import time

import numpy as np

from simulation import run_tnnp_simulation, run_tnnp_simulation_2d


def parse_point(value):
    """Parse an 'x,y' pair (normalized [0,1] coordinates or grid indices)."""
    try:
        x_text, y_text = value.split(',', 1)
        return float(x_text), float(y_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('point must be formatted as x,y') from exc


def grid_size_from_dx(dx):
    """Map a spatial step dx on the unit square to a grid size n (nx = ny = n)."""
    if dx <= 0:
        raise argparse.ArgumentTypeError('--dx must be positive')
    return max(2, int(round(1.0 / dx)))


def build_parser():
    parser = argparse.ArgumentParser(
        description='Run a single 1D or 2D TNNP simulation and report wall-clock time.',
    )
    # --- Core model inputs (required by the TNNP engine) ---
    parser.add_argument('--mode', choices=['1d', '2d'], default='1d',
                        help='1d single cell or 2d tissue sheet on [0,1]^2')
    parser.add_argument('--drug', type=str, default='Amiodarone I',
                        help='Drug name as keyed in drug_dict.pkl')
    parser.add_argument('--pacing-period', type=float, default=1000.0,
                        help='Pacing cycle length in ms')
    parser.add_argument('--concentration', type=float, default=1.0,
                        help='Drug concentration multiplier (EFTPC multiplier); 0 = drug-free')

    # --- Discretization: the dt / dx the user asked for ---
    parser.add_argument('--dt', type=float, default=None,
                        help='Fixed integration step in ms. 2d: time step; '
                             '1d: only used by fixed-step methods (euler/rush-larsen). '
                             'Defaults to a stable/record step when omitted.')
    parser.add_argument('--dx', type=float, default=None,
                        help='2d spatial step on the unit square; sets nx=ny=round(1/dx). '
                             'Ignored in 1d. Overrides --grid-size.')
    parser.add_argument('--grid-size', type=int, default=32,
                        help='2d grid size n (nx=ny=n) when --dx is not given')

    # --- Spatial layout (2d only) ---
    parser.add_argument('--pace-point', type=parse_point, default=(0.05, 0.5),
                        help='2d stimulus location as x,y in [0,1]^2 (or grid indices)')
    parser.add_argument('--pace-radius', type=float, default=0.03,
                        help='2d stimulus radius in normalized units; 0 = single pixel')
    parser.add_argument('--measure-point', type=parse_point, default=(0.5, 0.5),
                        help='2d location whose action potential is recorded')
    parser.add_argument('--diff-coef', type=float, default=0.001,
                        help='2d voltage diffusion coefficient')
    parser.add_argument('--wait-mode', choices=['faithful', 'fast'], default='faithful',
                        help='2d initial-wait handling: faithful steps the full paced 2d sheet '
                             'through the whole wait; fast equilibrates one cell and broadcasts')

    # --- Time window ---
    parser.add_argument('--initial-wait', type=float, default=20000.0,
                        help='Pre-pacing transient time in ms (must pass to reach the limit cycle)')
    parser.add_argument('--record-pre', type=float, default=100.0,
                        help='ms recorded before the measured beat (1d)')
    parser.add_argument('--record-post', type=float, default=500.0,
                        help='ms recorded after initial-wait (1d) / measurement budget (2d)')
    parser.add_argument('--record-interval', type=float, default=0.1,
                        help='Sampling interval in ms for recorded traces')

    # --- Solver / hardware ---
    parser.add_argument('--method', type=str, default='BDF',
                        help='1d integration method: BDF, LSODA, forward_euler, rush_larsen_forward')
    parser.add_argument('--backend', choices=['auto', 'cpu', 'torch'], default='auto',
                        help='2d backend: auto uses torch on CUDA/MPS when available')
    parser.add_argument('--device', type=str, default=None,
                        help='torch device override, e.g. cuda, cuda:0, mps, cpu')
    parser.add_argument('--torch-compile', action='store_true',
                        help='torch.compile the 2d CUDA stepper')

    # --- Output ---
    parser.add_argument('--no-progress', action='store_true',
                        help='Disable live simulation-time progress output')
    parser.add_argument('--progress-interval', type=float, default=5.0,
                        help='Wall-clock seconds between 2d time-progress updates')
    parser.add_argument('--snapshot-interval', type=float, default=None,
                        help='Emit a simulation snapshot every N ms of SIM time '
                             '(2d: voltage grid; 1d: cell voltage). Off when omitted.')
    parser.add_argument('--snapshot-dir', type=str, default=None,
                        help='If set, save each 2d snapshot grid as a .npy file in this directory')
    return parser


def make_2d_snapshot_callback(initial_wait, snapshot_dir):
    """Return a callback(t_abs_ms, voltage_grid) that logs and optionally saves snapshots."""
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)
    counter = {'n': 0}

    def callback(t_abs, grid):
        rel = t_abs - initial_wait
        sys.stderr.write(
            f'\n[snapshot #{counter["n"]:04d}] sim t={t_abs:.1f} ms (window +{rel:.1f} ms) '
            f'V min/mean/max = {grid.min():.2f} / {grid.mean():.2f} / {grid.max():.2f} mV\n'
        )
        sys.stderr.flush()
        if snapshot_dir:
            path = os.path.join(snapshot_dir, f'snapshot_{counter["n"]:04d}_t{t_abs:.1f}.npy')
            np.save(path, grid)
        counter['n'] += 1

    return callback


def run_1d(args):
    return run_tnnp_simulation(
        args.drug,
        args.pacing_period,
        args.concentration,
        initial_wait=args.initial_wait,
        record_pre=args.record_pre,
        record_post=args.record_post,
        record_interval=args.record_interval,
        integration_method=args.method,
        integration_step=args.dt,
    )


def run_2d(args, nx, snapshot_callback=None):
    return run_tnnp_simulation_2d(
        args.drug,
        args.pacing_period,
        args.concentration,
        initial_wait=args.initial_wait,
        record_pre=args.record_pre,
        record_post=args.record_post,
        record_interval=args.record_interval,
        integration_step=args.dt,
        nx=nx,
        ny=nx,
        pacemaker_point=args.pace_point,
        pacemaker_radius=args.pace_radius,
        measurement_point=args.measure_point,
        diff_coef=args.diff_coef,
        backend=args.backend,
        torch_device=args.device,
        torch_compile=args.torch_compile,
        show_time_progress=not args.no_progress,
        time_progress_interval=args.progress_interval,
        snapshot_interval=args.snapshot_interval,
        snapshot_callback=snapshot_callback,
        spatial_prewait=(args.wait_mode == 'faithful'),
    )


def print_1d_snapshots(recorded_time, voltage, snapshot_interval):
    """1d 'snapshots' are just the single-cell voltage sampled every snapshot_interval ms."""
    if not recorded_time or snapshot_interval is None or snapshot_interval <= 0:
        return
    next_t = recorded_time[0]
    n = 0
    for t, v in zip(recorded_time, voltage):
        if t + 1e-9 >= next_t:
            print(f'[snapshot #{n:04d}] window t={t:.1f} ms  V={v:.2f} mV')
            n += 1
            while next_t <= t + 1e-9:
                next_t += snapshot_interval


def main():
    args = build_parser().parse_args()

    nx = grid_size_from_dx(args.dx) if args.dx is not None else args.grid_size
    header = [
        f'mode={args.mode}',
        f'drug={args.drug!r}',
        f'pacing_period={args.pacing_period:g}ms',
        f'concentration={args.concentration:g}',
        f'dt={"auto" if args.dt is None else f"{args.dt:g}ms"}',
        f'initial_wait={args.initial_wait:g}ms',
    ]
    if args.mode == '2d':
        header += [
            f'grid={nx}x{nx}',
            f'dx={1.0 / nx:.5g}',
            f'pace_point={args.pace_point}',
            f'diff_coef={args.diff_coef:g}',
            f'wait_mode={args.wait_mode}',
            f'backend={args.backend}',
            f'device={args.device or "auto"}',
        ]
    else:
        header.append(f'method={args.method}')
        if args.dx is not None:
            sys.stderr.write('Note: --dx is ignored in 1d mode.\n')
    if args.snapshot_interval is not None:
        header.append(f'snapshot_every={args.snapshot_interval:g}ms(sim)')
    sys.stderr.write('Running TNNP: ' + ' '.join(header) + '\n')
    sys.stderr.flush()

    start = time.perf_counter()
    if args.mode == '1d':
        recorded_time, voltage, _ = run_1d(args)
    else:
        snapshot_callback = None
        if args.snapshot_interval is not None:
            snapshot_callback = make_2d_snapshot_callback(args.initial_wait, args.snapshot_dir)
        recorded_time, voltage, _ = run_2d(args, nx, snapshot_callback=snapshot_callback)
    elapsed = time.perf_counter() - start

    samples = len(voltage)
    print(f'Done in {elapsed:.3f} s')
    print(f'Recorded samples: {samples}')
    if samples:
        print(f'Recorded window: {recorded_time[0]:.2f} -> {recorded_time[-1]:.2f} ms')
        print(f'Voltage min/max: {min(voltage):.3f} / {max(voltage):.3f} mV')
        print(f'Final voltage:   {voltage[-1]:.3f} mV')
    if args.mode == '1d':
        print_1d_snapshots(recorded_time, voltage, args.snapshot_interval)


if __name__ == '__main__':
    main()
