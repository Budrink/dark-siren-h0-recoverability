"""Window sensitivity analysis: robustness of H0 probability mass metrics to window choice."""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import metrics functions
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from epjc_h0.results_metrics import posterior_mass_in_window_centered


def load_posteriors(npz_path: Path):
    """Load posteriors from NPZ file."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Posteriors NPZ not found: {npz_path}")
    
    data = np.load(npz_path)
    zmax_list = data['zmax_list']
    h0_grid = data['h0_grid']
    post_std_3d = data['post_std_3d']
    post_ell_3d = data['post_ell_3d']
    
    return zmax_list, h0_grid, post_std_3d, post_ell_3d


def compute_window_sensitivity(zmax_list: np.ndarray, h0_grid: np.ndarray,
                               post_std_3d: np.ndarray, post_ell_3d: np.ndarray,
                               cmb_centers: list, local_centers: list,
                               halfwidths: list) -> pd.DataFrame:
    """
    Compute probability masses and logR for all window combinations.
    
    Parameters
    ----------
    zmax_list : np.ndarray
        Array of zmax values
    h0_grid : np.ndarray
        H0 grid values
    post_std_3d : np.ndarray
        STD 3D posteriors, shape (n_zmax, n_h0)
    post_ell_3d : np.ndarray
        ELL 3D posteriors, shape (n_zmax, n_h0)
    cmb_centers : list
        List of CMB window centers
    local_centers : list
        List of Local window centers
    halfwidths : list
        List of window half-widths
    
    Returns
    -------
    pd.DataFrame
        Results with columns: zmax, model, cmb_center, cmb_width, local_center, 
        local_width, P_CMB, P_Local, logR
    """
    results = []
    
    n_zmax = len(zmax_list)
    
    print(f"\nComputing window sensitivity for {n_zmax} zmax values", flush=True)
    print(f"CMB centers: {cmb_centers}", flush=True)
    print(f"Local centers: {local_centers}", flush=True)
    print(f"Half-widths: {halfwidths}", flush=True)
    print(f"Total combinations: {n_zmax} zmax × 2 models × {len(cmb_centers)} CMB × {len(local_centers)} Local × {len(halfwidths)} widths = {n_zmax * 2 * len(cmb_centers) * len(local_centers) * len(halfwidths)}", flush=True)
    
    total_combinations = n_zmax * 2 * len(cmb_centers) * len(local_centers) * len(halfwidths)
    count = 0
    
    for zmax_idx, zmax in enumerate(zmax_list):
        print(f"\nProcessing zmax = {zmax:.3f} ({zmax_idx+1}/{n_zmax})", flush=True)
        
        # Get posteriors for this zmax
        post_std = post_std_3d[zmax_idx, :]
        post_ell = post_ell_3d[zmax_idx, :]
        
        # Process STD model
        for cmb_center in cmb_centers:
            for local_center in local_centers:
                for halfwidth in halfwidths:
                    # Compute P_CMB and P_Local for STD
                    P_CMB_std = posterior_mass_in_window_centered(
                        h0_grid, post_std, cmb_center, halfwidth
                    )
                    P_Local_std = posterior_mass_in_window_centered(
                        h0_grid, post_std, local_center, halfwidth
                    )
                    
                    # Compute logR with regularization
                    R_std = P_CMB_std / max(P_Local_std, 1e-12)
                    logR_std = np.log10(R_std)
                    
                    results.append({
                        'zmax': zmax,
                        'model': 'STD_3D',
                        'cmb_center': cmb_center,
                        'cmb_width': 2.0 * halfwidth,  # Full width
                        'local_center': local_center,
                        'local_width': 2.0 * halfwidth,  # Full width
                        'P_CMB': P_CMB_std,
                        'P_Local': P_Local_std,
                        'logR': logR_std
                    })
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"  Progress: {count}/{total_combinations} ({100*count/total_combinations:.1f}%)", flush=True)
        
        # Process ELL model
        for cmb_center in cmb_centers:
            for local_center in local_centers:
                for halfwidth in halfwidths:
                    # Compute P_CMB and P_Local for ELL
                    P_CMB_ell = posterior_mass_in_window_centered(
                        h0_grid, post_ell, cmb_center, halfwidth
                    )
                    P_Local_ell = posterior_mass_in_window_centered(
                        h0_grid, post_ell, local_center, halfwidth
                    )
                    
                    # Compute logR with regularization
                    R_ell = P_CMB_ell / max(P_Local_ell, 1e-12)
                    logR_ell = np.log10(R_ell)
                    
                    results.append({
                        'zmax': zmax,
                        'model': 'ELL_3D',
                        'cmb_center': cmb_center,
                        'cmb_width': 2.0 * halfwidth,  # Full width
                        'local_center': local_center,
                        'local_width': 2.0 * halfwidth,  # Full width
                        'P_CMB': P_CMB_ell,
                        'P_Local': P_Local_ell,
                        'logR': logR_ell
                    })
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"  Progress: {count}/{total_combinations} ({100*count/total_combinations:.1f}%)", flush=True)
    
    print(f"\nCompleted: {count} combinations", flush=True)
    
    return pd.DataFrame(results)


def plot_window_sensitivity(df_results: pd.DataFrame, event_id: str, out_path: Path):
    """
    Plot median(logR) with 16-84 percentile bands vs zmax for STD and ELL.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        Results from compute_window_sensitivity
    event_id : str
        Event identifier
    out_path : Path
        Output figure path
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by zmax and model, compute statistics
    df_grouped = df_results.groupby(['zmax', 'model'])['logR'].agg([
        ('median', 'median'),
        ('p16', lambda x: np.percentile(x, 16)),
        ('p84', lambda x: np.percentile(x, 84))
    ]).reset_index()
    
    # Separate STD and ELL
    df_std = df_grouped[df_grouped['model'] == 'STD_3D'].sort_values('zmax')
    df_ell = df_grouped[df_grouped['model'] == 'ELL_3D'].sort_values('zmax')
    
    # Plot STD
    if len(df_std) > 0:
        ax.plot(df_std['zmax'], df_std['median'], 'b-o', 
               linewidth=2, markersize=6, label='STD 3D (median)', alpha=0.8)
        ax.fill_between(df_std['zmax'], df_std['p16'], df_std['p84'],
                       alpha=0.2, color='blue', label='STD 3D (16-84%)')
    
    # Plot ELL
    if len(df_ell) > 0:
        ax.plot(df_ell['zmax'], df_ell['median'], 'r--s',
               linewidth=2, markersize=6, label='ELL 3D (median)', alpha=0.8)
        ax.fill_between(df_ell['zmax'], df_ell['p16'], df_ell['p84'],
                       alpha=0.2, color='red', label='ELL 3D (16-84%)')
    
    # Add horizontal line at zero (R=1)
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5, label='R=1 (equal masses)')
    
    ax.set_xlabel('Catalog depth zmax', fontsize=12)
    ax.set_ylabel('log₁₀(R) = log₁₀(P[CMB]/P[Local])', fontsize=12)
    ax.set_title(f'Window Sensitivity: Median logR vs Depth ({event_id})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: {out_path}", flush=True)


def main():
    """Main function for window sensitivity analysis."""
    parser = argparse.ArgumentParser(
        description="Window sensitivity analysis: robustness to H0 window choice"
    )
    parser.add_argument(
        '--event',
        type=str,
        required=True,
        help='Event name (used for output file names)'
    )
    parser.add_argument(
        '--posteriors-npz',
        type=str,
        default=None,
        help='Path to posteriors NPZ file (default: results/h0_depth_scan_<event>_posteriors.npz)'
    )
    parser.add_argument(
        '--cmb-centers',
        type=float,
        nargs='+',
        default=[66.0, 67.0, 68.0, 69.0],
        help='CMB window centers (default: 66 67 68 69)'
    )
    parser.add_argument(
        '--local-centers',
        type=float,
        nargs='+',
        default=[71.0, 72.0, 73.0, 74.0],
        help='Local window centers (default: 71 72 73 74)'
    )
    parser.add_argument(
        '--halfwidths',
        type=float,
        nargs='+',
        default=[1.0, 1.5, 2.0],
        help='Window half-widths (default: 1.0 1.5 2.0)'
    )
    parser.add_argument(
        '--out-csv',
        type=str,
        default=None,
        help='Output CSV path (default: results/h0_window_sensitivity_<event>.csv)'
    )
    parser.add_argument(
        '--out-fig',
        type=str,
        default=None,
        help='Output figure path (default: figures/h0_window_sensitivity_vs_depth_<event>.png)'
    )
    
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Posteriors NPZ path
    if args.posteriors_npz:
        npz_path = Path(args.posteriors_npz)
    else:
        npz_path = results_dir / f"h0_depth_scan_{args.event}_posteriors.npz"
    
    # Output paths
    if args.out_csv:
        out_csv_path = Path(args.out_csv)
    else:
        out_csv_path = results_dir / f"h0_window_sensitivity_{args.event}.csv"
    
    if args.out_fig:
        out_fig_path = Path(args.out_fig)
    else:
        out_fig_path = project_root / "figures" / f"h0_window_sensitivity_vs_depth_{args.event}.png"
    
    print(f"\n{'='*60}", flush=True)
    print(f"Window Sensitivity Analysis: {args.event}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Load posteriors
    print(f"\nLoading posteriors from: {npz_path}", flush=True)
    zmax_list, h0_grid, post_std_3d, post_ell_3d = load_posteriors(npz_path)
    print(f"  Loaded {len(zmax_list)} zmax values", flush=True)
    print(f"  H0 grid: [{h0_grid[0]:.1f}, {h0_grid[-1]:.1f}] with {len(h0_grid)} points", flush=True)
    print(f"  Posteriors shape: STD {post_std_3d.shape}, ELL {post_ell_3d.shape}", flush=True)
    
    # Compute sensitivity
    df_results = compute_window_sensitivity(
        zmax_list, h0_grid, post_std_3d, post_ell_3d,
        args.cmb_centers, args.local_centers, args.halfwidths
    )
    
    # Save CSV
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv_path, index=False)
    print(f"\nResults saved to CSV: {out_csv_path}", flush=True)
    print(f"  Total rows: {len(df_results)}", flush=True)
    
    # Plot results
    plot_window_sensitivity(df_results, args.event, out_fig_path)
    
    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Summary", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total window combinations: {len(df_results)}", flush=True)
    print(f"zmax values: {len(zmax_list)}", flush=True)
    print(f"CMB centers: {len(args.cmb_centers)}", flush=True)
    print(f"Local centers: {len(args.local_centers)}", flush=True)
    print(f"Half-widths: {len(args.halfwidths)}", flush=True)
    
    # Print statistics by zmax
    print(f"\nMedian logR by zmax and model:", flush=True)
    df_summary = df_results.groupby(['zmax', 'model'])['logR'].agg(['median', 'std']).reset_index()
    print(df_summary.to_string(index=False), flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"SUCCESS: Window sensitivity analysis completed", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{'='*60}", flush=True)
        print(f"INTERRUPTED by user", flush=True)
        print(f"{'='*60}", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL ERROR: {e}", flush=True)
        print(f"{'='*60}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
