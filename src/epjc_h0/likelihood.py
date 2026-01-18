"""Likelihood functions for H0 inference from GW dark sirens."""

from typing import Callable, Tuple
import time
import numpy as np
from epjc_h0.cosmology import dL_std, dL_ell


def sky_prob_at(ra_deg: float, dec_deg: float, prob_map: np.ndarray, 
                metadata: dict, hp_obj=None) -> float:
    """
    Get sky probability at a given (RA, Dec) coordinate.
    
    Parameters
    ----------
    ra_deg : float
        Right ascension in degrees
    dec_deg : float
        Declination in degrees
    prob_map : np.ndarray
        HEALPix probability map (probability density)
    metadata : dict
        Metadata with 'nside' and 'nest' keys
    hp_obj : optional
        Pre-created HEALPix object (for efficiency if calling multiple times)
    
    Returns
    -------
    float
        Probability density at the coordinate
    """
    # Import here to avoid circular dependencies
    from astropy_healpix import HEALPix
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    
    # Create HEALPix object if not provided
    if hp_obj is None:
        nside = metadata.get('nside', 512)
        nested = metadata.get('nest', False)
        hp_obj = HEALPix(nside=nside, order='nested' if nested else 'ring', frame='icrs')
    
    # Convert to SkyCoord
    coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    
    # Find pixel index
    ipix_result = hp_obj.lonlat_to_healpix(coord.ra, coord.dec)
    # Handle both scalar and array returns
    if isinstance(ipix_result, np.ndarray):
        ipix = ipix_result[0] if len(ipix_result) > 0 else 0
    else:
        ipix = int(ipix_result)
    
    # Return probability density at that pixel
    if 0 <= ipix < len(prob_map):
        return prob_map[ipix]
    else:
        return 0.0


def interp_pdf(x_grid: np.ndarray, pdf_grid: np.ndarray, x: float) -> float:
    """
    Linear interpolation of PDF with boundary handling.
    
    Parameters
    ----------
    x_grid : np.ndarray
        Grid of x values (must be sorted)
    pdf_grid : np.ndarray
        PDF values at grid points
    x : float
        Point to interpolate at
    
    Returns
    -------
    float
        Interpolated PDF value (0 if outside bounds)
    """
    # Handle out of bounds
    if x < x_grid[0] or x > x_grid[-1]:
        return 0.0
    
    # Find interpolation indices
    idx = np.searchsorted(x_grid, x)
    
    # Handle edge case
    if idx == 0:
        return pdf_grid[0]
    if idx >= len(x_grid):
        return pdf_grid[-1]
    
    # Linear interpolation
    x0, x1 = x_grid[idx - 1], x_grid[idx]
    p0, p1 = pdf_grid[idx - 1], pdf_grid[idx]
    
    if x1 == x0:
        return p0
    
    t = (x - x0) / (x1 - x0)
    return p0 * (1 - t) + p1 * t


def sky_prob_vectorized(ra_deg: np.ndarray, dec_deg: np.ndarray, 
                        prob_map: np.ndarray, metadata: dict, hp_obj=None) -> np.ndarray:
    """
    Vectorized version: get sky probabilities for arrays of coordinates.
    
    Parameters
    ----------
    ra_deg : np.ndarray
        Array of right ascensions in degrees
    dec_deg : np.ndarray
        Array of declinations in degrees
    prob_map : np.ndarray
        HEALPix probability map
    metadata : dict
        Metadata with 'nside' and 'nest' keys
    hp_obj : optional
        Pre-created HEALPix object
    
    Returns
    -------
    np.ndarray
        Array of probability densities
    """
    from astropy_healpix import HEALPix
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    
    if hp_obj is None:
        nside = metadata.get('nside', 512)
        nested = metadata.get('nest', False)
        hp_obj = HEALPix(nside=nside, order='nested' if nested else 'ring', frame='icrs')
    
    # Convert to SkyCoord (vectorized)
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame='icrs')
    
    # Find pixel indices (vectorized)
    ipix_array = hp_obj.lonlat_to_healpix(coords.ra, coords.dec)
    
    # Handle both scalar and array returns
    if not isinstance(ipix_array, np.ndarray):
        ipix_array = np.array([ipix_array])
    
    # Clip indices to valid range
    ipix_array = np.clip(ipix_array, 0, len(prob_map) - 1)
    
    # Return probability densities
    return prob_map[ipix_array]


def interp_pdf_vectorized(x_grid: np.ndarray, pdf_grid: np.ndarray, 
                          x_array: np.ndarray) -> np.ndarray:
    """
    Vectorized linear interpolation of PDF.
    
    Parameters
    ----------
    x_grid : np.ndarray
        Grid of x values (must be sorted)
    pdf_grid : np.ndarray
        PDF values at grid points
    x_array : np.ndarray
        Array of points to interpolate at
    
    Returns
    -------
    np.ndarray
        Array of interpolated PDF values (0 if outside bounds)
    """
    # Handle out of bounds
    mask_valid = (x_array >= x_grid[0]) & (x_array <= x_grid[-1])
    result = np.zeros_like(x_array)
    
    if not np.any(mask_valid):
        return result
    
    x_valid = x_array[mask_valid]
    
    # Find interpolation indices (vectorized)
    idx = np.searchsorted(x_grid, x_valid)
    
    # Handle edge cases
    idx = np.clip(idx, 1, len(x_grid) - 1)
    
    # Linear interpolation (vectorized)
    x0 = x_grid[idx - 1]
    x1 = x_grid[idx]
    p0 = pdf_grid[idx - 1]
    p1 = pdf_grid[idx]
    
    # Avoid division by zero
    dx = x1 - x0
    mask_dx = dx > 0
    t = np.zeros_like(x_valid)
    t[mask_dx] = (x_valid[mask_dx] - x0[mask_dx]) / dx[mask_dx]
    
    result[mask_valid] = p0 * (1 - t) + p1 * t
    
    return result


def event_likelihood(H0_grid: np.ndarray, galaxies: np.ndarray,
                    sky_prob_func: Callable[[float, float], float],
                    d_grid: np.ndarray, d_pdf: np.ndarray,
                    model: str = "std", kappa: float = 1.0,
                    Om: float = 0.3,
                    distance_pdf_kind: str = "likelihood",
                    d_min: float = 1.0) -> np.ndarray:
    """
    Compute event likelihood L(H0) for a grid of H0 values (OPTIMIZED VERSION).
    
    Assumes separable localization: p(Ω, d) ≈ p(Ω) * p(d)
    
    Parameters
    ----------
    H0_grid : np.ndarray
        Grid of H0 values (km/s/Mpc)
    galaxies : np.ndarray
        Structured array with fields: 'ra_deg', 'dec_deg', 'z', 'weight'
        (weight defaults to 1.0 if not present)
    sky_prob_func : Callable[[float, float], float]
        Function that returns p(Ω) given (ra_deg, dec_deg)
        OR (prob_map, metadata, hp_obj) tuple for vectorized version
    d_grid : np.ndarray
        Distance grid for PDF (Mpc)
    d_pdf : np.ndarray
        Distance PDF values (normalized).
        Interpretation depends on distance_pdf_kind:
        - If "likelihood": GW distance likelihood (prior removed, p(d|GW))
        - If "posterior_d2": GW distance posterior with volumetric prior ~ d^2
    model : str
        Model type: "std" or "ell"
    kappa : float
        kappa parameter for ell model (only used if model="ell")
    Om : float
        Matter density parameter
    distance_pdf_kind : str, optional
        Type of distance PDF: "likelihood" or "posterior_d2", by default "likelihood"
        - "likelihood": d_pdf is already GW likelihood (prior removed)
        - "posterior_d2": d_pdf is GW posterior with d^2 volumetric prior, will be converted
    d_min : float, optional
        Minimum distance for regularization near zero (Mpc), by default 1.0
        Used to avoid division by zero when removing d^2 prior
    
    Returns
    -------
    np.ndarray
        Likelihood values L(H0) for each H0 in grid
    
    Notes
    -----
    This MVP ignores:
    - Selection effects
    - Out-of-catalog term
    - Galaxy completeness corrections
    
    Distance PDF handling:
    - If distance_pdf_kind="posterior_d2", the function removes the volumetric prior
      by dividing by d^2: p_like(d) = p_post(d) / max(d, d_min)^2
    - If distance_pdf_kind="likelihood", d_pdf is used directly as GW likelihood
    """
    # Extract galaxy arrays (vectorized)
    ra_array = galaxies['ra_deg']
    dec_array = galaxies['dec_deg']
    z_array = galaxies['z']
    has_weight = 'weight' in galaxies.dtype.names
    if has_weight:
        weight_array = galaxies['weight']
    else:
        weight_array = np.ones(len(galaxies))
    
    # Filter invalid galaxies
    mask_valid = (
        np.isfinite(ra_array) & np.isfinite(dec_array) & np.isfinite(z_array) &
        (z_array > 0) & (z_array < 10)
    )
    
    ra_array = ra_array[mask_valid]
    dec_array = dec_array[mask_valid]
    z_array = z_array[mask_valid]
    weight_array = weight_array[mask_valid]
    
    print(f"    Processing {len(ra_array):,} valid galaxies...", flush=True)
    
    # Pre-compute sky probabilities for all galaxies (vectorized if possible)
    # Try to get prob_map and metadata from sky_prob_func if it's a tuple
    if isinstance(sky_prob_func, tuple) and len(sky_prob_func) == 3:
        prob_map, metadata, hp_obj = sky_prob_func
        print(f"    Computing sky probabilities (vectorized)...", flush=True)
        p_sky_array = sky_prob_vectorized(ra_array, dec_array, prob_map, metadata, hp_obj)
    else:
        # Fallback: compute one by one (slower)
        print(f"    Computing sky probabilities (scalar, this may take a while)...", flush=True)
        p_sky_array = np.array([sky_prob_func(ra, dec) for ra, dec in zip(ra_array, dec_array)])
    
    # Filter galaxies with zero sky probability (most will be zero!)
    mask_nonzero_sky = p_sky_array > 0
    ra_array = ra_array[mask_nonzero_sky]
    dec_array = dec_array[mask_nonzero_sky]
    z_array = z_array[mask_nonzero_sky]
    weight_array = weight_array[mask_nonzero_sky]
    p_sky_array = p_sky_array[mask_nonzero_sky]
    
    n_valid_galaxies = mask_valid.sum()
    print(f"    {len(ra_array):,} galaxies with non-zero sky probability ({100*len(ra_array)/n_valid_galaxies:.2f}%)", flush=True)
    
    if len(ra_array) == 0:
        print(f"    WARNING: No galaxies with non-zero sky probability!")
        return np.zeros_like(H0_grid)
    
    # Pre-compute cosmology functions (vectorized)
    from epjc_h0.cosmology import (
        dL_std_vectorized, dL_ell_vectorized,
        ddL_dz_std_vectorized, ddL_dz_ell_vectorized
    )
    
    likelihood = np.zeros_like(H0_grid)
    
    # Process H0 values with progress updates
    print(f"    Computing likelihoods for {len(H0_grid)} H0 values...", flush=True)
    print(f"    Using {len(ra_array):,} galaxies with non-zero sky probability", flush=True)
    print(f"    Progress will be reported every 10 H0 values", flush=True)
    
    # time is already imported at module level
    start_time = time.time()
    
    for i, H0 in enumerate(H0_grid):
        if (i + 1) % 10 == 0 or i == 0 or i == len(H0_grid) - 1:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(H0_grid) - i - 1) / rate if rate > 0 else 0
            print(f"    H0[{i+1}/{len(H0_grid)}] = {H0:.1f} km/s/Mpc | "
                  f"Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} H0/s | ETA: {eta:.1f}s", flush=True)
        
        # Compute luminosity distances for all galaxies at this H0 (vectorized)
        if model == "std":
            d_L_array = dL_std_vectorized(z_array, H0, Om=Om)
            ddL_dz_array = ddL_dz_std_vectorized(z_array, H0, Om=Om)
        elif model == "ell":
            d_L_array = dL_ell_vectorized(z_array, H0, Om=Om, kappa=kappa)
            ddL_dz_array = ddL_dz_ell_vectorized(z_array, H0, Om=Om, kappa=kappa)
        else:
            raise ValueError(f"Unknown model: {model}")
        
        # Compute distance probabilities (vectorized)
        p_d_array = interp_pdf_vectorized(d_grid, d_pdf, d_L_array)
        
        # Remove volumetric prior if d_pdf is posterior
        if distance_pdf_kind == "posterior_d2":
            # Compute effective distance with regularization
            d_eff = np.maximum(d_L_array, d_min)
            # Remove d^2 prior: p_like(d) = p_post(d) / d^2
            p_d_array = p_d_array / (d_eff ** 2)
            # Guard against inf/nan
            p_d_array = np.nan_to_num(p_d_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Diagnostic: log statistics for first H0 value
        if i == 0:
            d_L_mode = np.median(d_L_array) if len(d_L_array) > 0 else 0.0
            d_L_min = np.min(d_L_array) if len(d_L_array) > 0 else 0.0
            d_L_max = np.max(d_L_array) if len(d_L_array) > 0 else 0.0
            p_d_min = np.min(p_d_array[p_d_array > 0]) if np.any(p_d_array > 0) else 0.0
            p_d_max = np.max(p_d_array) if len(p_d_array) > 0 else 0.0
            print(f"    Diagnostic (H0={H0:.1f}): d_L range=[{d_L_min:.1f}, {d_L_mode:.1f}, {d_L_max:.1f}] Mpc, "
                  f"p_d range=[{p_d_min:.2e}, {p_d_max:.2e}]", flush=True)
        
        # Guardrail: ensure all values are finite
        assert np.all(np.isfinite(p_d_array)), (
            f"Non-finite values in p_d_array at H0={H0:.1f}. "
            f"distance_pdf_kind={distance_pdf_kind}, "
            f"d_L range=[{np.min(d_L_array):.1f}, {np.max(d_L_array):.1f}] Mpc"
        )
        
        # Filter galaxies with zero distance probability
        mask_nonzero_d = p_d_array > 0
        if not np.any(mask_nonzero_d):
            likelihood[i] = 0.0
            continue
        
        # Compute contributions with Jacobian: p(d) * ddL/dz
        # This accounts for the change of variables from z to d_L
        # Note: p_d_array is now GW likelihood (prior removed if needed)
        
        contributions = (
            weight_array[mask_nonzero_d] * 
            p_sky_array[mask_nonzero_d] * 
            p_d_array[mask_nonzero_d] * 
            ddL_dz_array[mask_nonzero_d]
        )
        
        # Sum all contributions
        likelihood[i] = np.sum(contributions)
    
    total_time = time.time() - start_time
    print(f"    ✓ Likelihood computation complete in {total_time:.1f}s", flush=True)
    return likelihood


def normalize_posterior(likelihood: np.ndarray, H0_grid: np.ndarray,
                       prior: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """
    Normalize likelihood to get posterior P(H0) with flat prior.
    
    Parameters
    ----------
    likelihood : np.ndarray
        Likelihood values L(H0)
    H0_grid : np.ndarray
        Grid of H0 values
    prior : np.ndarray, optional
        Prior values (default: flat prior)
    
    Returns
    -------
    Tuple[np.ndarray, float]
        (posterior, normalization_constant)
    """
    if prior is None:
        prior = np.ones_like(likelihood)
    
    # Unnormalized posterior
    unnorm_posterior = likelihood * prior
    
    # Normalize
    # Use trapezoidal rule for integration
    dx = H0_grid[1] - H0_grid[0] if len(H0_grid) > 1 else 1.0
    norm = np.trapz(unnorm_posterior, H0_grid)
    
    if norm > 0:
        posterior = unnorm_posterior / norm
    else:
        posterior = np.zeros_like(unnorm_posterior)
    
    return posterior, norm
