"""Cosmology calculations for H0 depth-drift analysis."""

import numpy as np
from scipy.integrate import quad, cumulative_trapezoid
from typing import Optional


# Speed of light in km/s
C_LIGHT = 299792.458


def E_z(z: float, Om: float = 0.3, Ol: Optional[float] = None) -> float:
    """
    Calculate E(z) = H(z)/H0 for flat LCDM cosmology.
    
    Parameters
    ----------
    z : float
        Redshift
    Om : float, optional
        Matter density parameter, by default 0.3
    Ol : float, optional
        Dark energy density parameter. If None, computed as Ol = 1 - Om
        for flat universe, by default None
    
    Returns
    -------
    float
        E(z) = sqrt(Om*(1+z)^3 + Ol)
    """
    if Ol is None:
        Ol = 1.0 - Om  # Flat universe
    
    return np.sqrt(Om * (1.0 + z) ** 3 + Ol)


def Dc_std(z: float, H0: float, Om: float = 0.3, ngrid: int = 6000) -> float:
    """
    Calculate comoving line-of-sight distance Dc(z) = (c/H0) * integral_0^z dz'/E(z').
    
    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    ngrid : int, optional
        Number of integration points (not used in quad, kept for API consistency),
        by default 6000
    
    Returns
    -------
    float
        Comoving distance in Mpc
    """
    def integrand(z_prime: float) -> float:
        """Integrand 1/E(z')."""
        return 1.0 / E_z(z_prime, Om=Om)
    
    integral, _ = quad(integrand, 0.0, z, limit=ngrid)
    Dc = (C_LIGHT / H0) * integral
    
    return Dc


def dL_std(z: float, H0: float, Om: float = 0.3, ngrid: int = 6000) -> float:
    """
    Calculate standard luminosity distance using numerical integration.
    
    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    ngrid : int, optional
        Number of integration points (not used in quad, kept for API consistency),
        by default 6000
    
    Returns
    -------
    float
        Luminosity distance in Mpc
    """
    Dc = Dc_std(z, H0, Om=Om, ngrid=ngrid)
    dL = (1.0 + z) * Dc
    
    return dL


def ddL_dz_std(z: float, H0: float, Om: float = 0.3, ngrid: int = 6000) -> float:
    """
    Calculate derivative of luminosity distance with respect to redshift.
    
    ddL/dz = Dc(z) + (1+z) * (c/H0) * 1/E(z)
    
    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    ngrid : int, optional
        Number of integration points (not used in quad, kept for API consistency),
        by default 6000
    
    Returns
    -------
    float
        Derivative ddL/dz in Mpc
    """
    Dc = Dc_std(z, H0, Om=Om, ngrid=ngrid)
    E_z_val = E_z(z, Om=Om)
    ddL_dz = Dc + (1.0 + z) * (C_LIGHT / H0) / E_z_val
    
    return ddL_dz


def dL_ell(z: float, H0: float, Om: float = 0.3, kappa: float = 1.0, 
           ngrid: int = 6000) -> float:
    """
    Calculate modified luminosity distance with ellipsoidal propagation model.
    
    The integrand includes a (1+z)^(-2kappa/3) factor.
    
    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    kappa : float, optional
        Propagation parameter, by default 1.0
    ngrid : int, optional
        Number of integration points (not used in quad, kept for API consistency),
        by default 6000
    
    Returns
    -------
    float
        Modified luminosity distance in Mpc
    """
    def integrand(z_prime: float) -> float:
        """Integrand (1+z')^(-2kappa/3) / E(z')."""
        return (1.0 + z_prime) ** (-2.0 * kappa / 3.0) / E_z(z_prime, Om=Om)
    
    integral, _ = quad(integrand, 0.0, z, limit=ngrid)
    dL = (C_LIGHT / H0) * (1.0 + z) * integral
    
    return dL


def ddL_dz_ell(z: float, H0: float, Om: float = 0.3, kappa: float = 1.0,
               ngrid: int = 6000, dz_step: float = None) -> float:
    """
    Calculate derivative of modified luminosity distance with respect to redshift.
    
    Uses numerical derivative with central difference.
    
    Parameters
    ----------
    z : float
        Redshift
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    kappa : float, optional
        Propagation parameter, by default 1.0
    ngrid : int, optional
        Number of integration points (not used in quad, kept for API consistency),
        by default 6000
    dz_step : float, optional
        Step size for numerical derivative. If None, uses dz = 1e-4 * (1+z)
    
    Returns
    -------
    float
        Derivative ddL/dz in Mpc
    """
    if dz_step is None:
        dz = 1e-4 * (1.0 + z)
    else:
        dz = dz_step
    
    # Ensure z - dz >= 0
    z_minus = max(0.0, z - dz)
    z_plus = z + dz
    
    dL_minus = dL_ell(z_minus, H0, Om=Om, kappa=kappa, ngrid=ngrid)
    dL_plus = dL_ell(z_plus, H0, Om=Om, kappa=kappa, ngrid=ngrid)
    
    # Central difference
    ddL_dz = (dL_plus - dL_minus) / (z_plus - z_minus)
    
    return ddL_dz


def ddL_dz_std_vectorized(z_array: np.ndarray, H0: float, Om: float = 0.3,
                          ngrid: int = 50) -> np.ndarray:
    """
    Vectorized version of ddL_dz_std for arrays of redshifts.
    
    Uses analytical formula: ddL/dz = Dc(z) + (1+z) * (c/H0) / E(z)
    where Dc(z) is computed via cumulative integral + interpolation.
    
    Parameters
    ----------
    z_array : np.ndarray
        Array of redshifts
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    ngrid : int, optional
        Base number of integration points (scaled by z_max), by default 50
    
    Returns
    -------
    np.ndarray
        Array of derivatives ddL/dz in Mpc
    """
    z_array = np.asarray(z_array)
    result = np.zeros_like(z_array, dtype=np.float32)
    
    # Filter valid redshifts
    mask_valid = z_array > 0
    if not np.any(mask_valid):
        return result
    
    z_valid = z_array[mask_valid]
    if len(z_valid) == 0:
        return result
    
    z_max = np.max(z_valid)
    
    # Build single global integration grid
    n_points = min(max(ngrid * 10, int(z_max * 200)), 10000)
    z_grid = np.linspace(0.0, z_max, n_points)
    
    # Compute integrand on grid: 1/E(z)
    E_grid = E_z(z_grid, Om=Om)
    integrand_grid = 1.0 / E_grid
    
    # Compute cumulative integral
    cumint_grid = cumulative_trapezoid(integrand_grid, z_grid, initial=0.0)
    
    # Process unique z values
    z_unique, z_inverse = np.unique(z_valid, return_inverse=True)
    
    # Interpolate cumulative integral to get I(z)
    I_unique = np.interp(z_unique, z_grid, cumint_grid)
    
    # Compute Dc(z) = (c/H0) * I(z)
    Dc_unique = (C_LIGHT / H0) * I_unique
    
    # Compute E(z) for unique z values
    E_z_unique = E_z(z_unique, Om=Om)
    
    # Compute ddL/dz = Dc(z) + (1+z) * (c/H0) / E(z)
    ddL_dz_unique = Dc_unique + (1.0 + z_unique) * (C_LIGHT / H0) / E_z_unique
    
    # Map back to full array
    result[mask_valid] = ddL_dz_unique[z_inverse]
    
    return result


def dL_std_vectorized(z_array: np.ndarray, H0: float, Om: float = 0.3, 
                      ngrid: int = 50) -> np.ndarray:
    """
    Vectorized version of dL_std for arrays of redshifts.
    
    Uses a single global integration grid with cumulative integral and interpolation
    for accuracy and efficiency.
    
    Parameters
    ----------
    z_array : np.ndarray
        Array of redshifts
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    ngrid : int, optional
        Base number of integration points (scaled by z_max), by default 50
    
    Returns
    -------
    np.ndarray
        Array of luminosity distances in Mpc
    """
    z_array = np.asarray(z_array)
    result = np.zeros_like(z_array, dtype=np.float32)
    
    # Filter valid redshifts
    mask_valid = z_array > 0
    if not np.any(mask_valid):
        return result
    
    z_valid = z_array[mask_valid]
    if len(z_valid) == 0:
        return result
    
    z_max = np.max(z_valid)
    
    # Build single global integration grid
    n_points = min(max(ngrid * 10, int(z_max * 200)), 10000)  # Cap at 10000 points
    z_grid = np.linspace(0.0, z_max, n_points)
    
    # Compute integrand on grid: 1/E(z)
    E_grid = E_z(z_grid, Om=Om)
    integrand_grid = 1.0 / E_grid
    
    # Compute cumulative integral using cumulative_trapezoid
    # This gives I(z_grid[i]) = integral from 0 to z_grid[i]
    cumint_grid = cumulative_trapezoid(integrand_grid, z_grid, initial=0.0)
    
    # For each unique z, interpolate cumulative integral
    z_unique, z_inverse = np.unique(z_valid, return_inverse=True)
    
    # Interpolate cumulative integral at unique z values
    I_unique = np.interp(z_unique, z_grid, cumint_grid)
    
    # Compute Dc(z) = (c/H0) * I(z)
    Dc_unique = (C_LIGHT / H0) * I_unique
    
    # Compute dL_std = (1+z) * Dc(z)
    dL_unique = (1.0 + z_unique) * Dc_unique
    
    # Map back to full array
    result[mask_valid] = dL_unique[z_inverse]
    
    return result


def ddL_dz_ell_vectorized(z_array: np.ndarray, H0: float, Om: float = 0.3,
                          kappa: float = 1.0, ngrid: int = 50) -> np.ndarray:
    """
    Vectorized version of ddL_dz_ell for arrays of redshifts.
    
    Uses analytical formula: ddL/dz = (c/H0) * [I(z) + (1+z)^(1-2*kappa/3) / E(z)]
    where I(z) = integral[0 to z] (1+z')^(-2*kappa/3) / E(z') dz'
    is computed via cumulative integral + interpolation.
    
    Parameters
    ----------
    z_array : np.ndarray
        Array of redshifts
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    kappa : float, optional
        Propagation parameter, by default 1.0
    ngrid : int, optional
        Base number of integration points (scaled by z_max), by default 50
    
    Returns
    -------
    np.ndarray
        Array of derivatives ddL/dz in Mpc
    """
    z_array = np.asarray(z_array)
    result = np.zeros_like(z_array, dtype=np.float32)
    
    # Filter valid redshifts
    mask_valid = z_array > 0
    if not np.any(mask_valid):
        return result
    
    z_valid = z_array[mask_valid]
    if len(z_valid) == 0:
        return result
    
    z_max = np.max(z_valid)
    
    # Build single global integration grid
    n_points = min(max(ngrid * 10, int(z_max * 200)), 10000)
    z_grid = np.linspace(0.0, z_max, n_points)
    
    # Compute integrand on grid: (1+z)^(-2*kappa/3) / E(z)
    E_grid = E_z(z_grid, Om=Om)
    integrand_grid = (1.0 + z_grid) ** (-2.0 * kappa / 3.0) / E_grid
    
    # Compute cumulative integral
    cumint_grid = cumulative_trapezoid(integrand_grid, z_grid, initial=0.0)
    
    # Process unique z values
    z_unique, z_inverse = np.unique(z_valid, return_inverse=True)
    
    # Interpolate cumulative integral to get I(z)
    I_unique = np.interp(z_unique, z_grid, cumint_grid)
    
    # Compute E(z) for unique z values
    E_z_unique = E_z(z_unique, Om=Om)
    
    # Compute ddL/dz = (c/H0) * [I(z) + (1+z)^(1-2*kappa/3) / E(z)]
    ddL_dz_unique = (C_LIGHT / H0) * (I_unique + (1.0 + z_unique) ** (1.0 - 2.0 * kappa / 3.0) / E_z_unique)
    
    # Map back to full array
    result[mask_valid] = ddL_dz_unique[z_inverse]
    
    return result


def dL_ell_vectorized(z_array: np.ndarray, H0: float, Om: float = 0.3, 
                      kappa: float = 1.0, ngrid: int = 50) -> np.ndarray:
    """
    Vectorized version of dL_ell for arrays of redshifts.
    
    Uses numerical integration with numpy.trapz for speed.
    Optimized: processes unique z values and reuses results.
    
    Parameters
    ----------
    z_array : np.ndarray
        Array of redshifts
    H0 : float
        Hubble constant in km/s/Mpc
    Om : float, optional
        Matter density parameter, by default 0.3
    kappa : float, optional
        Propagation parameter, by default 1.0
    ngrid : int, optional
        Number of integration points per redshift, by default 50
    
    Returns
    -------
    np.ndarray
        Array of modified luminosity distances in Mpc
    """
    z_array = np.asarray(z_array)
    result = np.zeros_like(z_array)
    
    # Filter valid redshifts
    mask_valid = z_array > 0
    if not np.any(mask_valid):
        return result
    
    z_valid = z_array[mask_valid]
    z_max = np.max(z_valid)
    
    # Build single global integration grid
    n_points = min(max(ngrid * 10, int(z_max * 200)), 10000)
    z_grid = np.linspace(0.0, z_max, n_points)
    
    # Compute integrand on grid: (1+z)^(-2*kappa/3) / E(z)
    E_grid = E_z(z_grid, Om=Om)
    integrand_grid = (1.0 + z_grid) ** (-2.0 * kappa / 3.0) / E_grid
    
    # Compute cumulative integral
    cumint_grid = cumulative_trapezoid(integrand_grid, z_grid, initial=0.0)
    
    # Process unique z values
    z_unique, z_inverse = np.unique(z_valid, return_inverse=True)
    
    # Interpolate cumulative integral to get I(z)
    I_unique = np.interp(z_unique, z_grid, cumint_grid)
    
    # Compute dL_ell = (c/H0) * (1+z) * I(z)
    dL_unique = (C_LIGHT / H0) * (1.0 + z_unique) * I_unique
    
    # Map back to full array
    result[mask_valid] = dL_unique[z_inverse]
    
    return result


def ratio_lowz_kappa1(z: float) -> float:
    """
    Calculate low-z analytic ratio R(z) for kappa=1.
    
    R(z) = 3 * ((1+z)^(1/3) - 1) / z
    
    This is the ratio dL_std / dL_ell in the low-z limit for kappa=1.
    
    Parameters
    ----------
    z : float
        Redshift
    
    Returns
    -------
    float
        Ratio R(z)
    """
    if z == 0.0:
        return 1.0  # Limit as z -> 0
    
    return 3.0 * ((1.0 + z) ** (1.0 / 3.0) - 1.0) / z


def test_vectorized_cosmology():
    """
    Unit test comparing scalar vs vectorized cosmology functions.
    
    Tests dL_std_vectorized against dL_std on random redshifts in [0, 0.3]
    and asserts relative error < 1e-3.
    """
    np.random.seed(42)  # For reproducibility
    n_test = 100
    z_test = np.random.uniform(0.01, 0.3, n_test)  # Avoid z=0
    H0_test = 70.0
    Om_test = 0.3
    
    # Test dL_std_vectorized
    dL_scalar = np.array([dL_std(z, H0_test, Om=Om_test) for z in z_test])
    dL_vectorized = dL_std_vectorized(z_test, H0_test, Om=Om_test)
    
    # Compute relative errors
    rel_errors = np.abs((dL_vectorized - dL_scalar) / dL_scalar)
    max_rel_error = np.max(rel_errors)
    mean_rel_error = np.mean(rel_errors)
    
    print(f"Testing dL_std_vectorized:")
    print(f"  Max relative error: {max_rel_error:.2e}")
    print(f"  Mean relative error: {mean_rel_error:.2e}")
    
    assert max_rel_error < 1e-3, f"Max relative error {max_rel_error:.2e} exceeds 1e-3"
    print(f"  ✓ Test passed: max relative error < 1e-3")
    
    # Test ddL_dz_std_vectorized
    ddL_dz_scalar = np.array([ddL_dz_std(z, H0_test, Om=Om_test) for z in z_test])
    ddL_dz_vectorized = ddL_dz_std_vectorized(z_test, H0_test, Om=Om_test)
    
    rel_errors_dz = np.abs((ddL_dz_vectorized - ddL_dz_scalar) / ddL_dz_scalar)
    max_rel_error_dz = np.max(rel_errors_dz)
    mean_rel_error_dz = np.mean(rel_errors_dz)
    
    print(f"\nTesting ddL_dz_std_vectorized:")
    print(f"  Max relative error: {max_rel_error_dz:.2e}")
    print(f"  Mean relative error: {mean_rel_error_dz:.2e}")
    
    assert max_rel_error_dz < 1e-3, f"Max relative error {max_rel_error_dz:.2e} exceeds 1e-3"
    print(f"  ✓ Test passed: max relative error < 1e-3")
    
    return True


if __name__ == "__main__":
    # Run unit test if script is executed directly
    test_vectorized_cosmology()
    print("\nAll tests passed!")
