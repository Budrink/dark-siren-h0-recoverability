"""Generate figure showing predicted H0 drift as a function of redshift."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epjc_h0.cosmology import ratio_lowz_kappa1
from epjc_h0.plots import plot_predicted_drift


def main():
    """Generate predicted H0 drift figure."""
    # Set random seed for reproducibility (not needed here, but good practice)
    np.random.seed(42)
    
    # Redshift range
    z = np.linspace(0.01, 0.8, 1000)
    
    # Compute ratio R(z) = dL_std / dL_ell
    R_z = np.array([ratio_lowz_kappa1(zi) for zi in z])
    
    # H0_app / H0 = 1 / R(z)
    H0_app_over_H0 = 1.0 / R_z
    
    # Output path
    out_path = Path(__file__).parent.parent / "figures" / "predicted_H0_drift.png"
    
    # Generate plot
    plot_predicted_drift(z, H0_app_over_H0, out_path)
    
    print(f"Figure saved to: {out_path}")


if __name__ == "__main__":
    main()
