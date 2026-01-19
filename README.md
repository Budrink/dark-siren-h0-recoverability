# H0 Depth-Drift Analysis

Reproducible analysis project for testing apparent H0 "depth-drift" effects in gravitational-wave cosmology.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

For skymap handling in 3D mode you need:

```bash
pip install astropy astropy-healpix
```

(Optional) For alternative plotting / distance extraction:

```bash
pip install ligo-skymap
```

## Usage

### Quick Start: One Command to First Result

To automatically run the full pipeline and generate the first H0 posterior plot for a top-ranked event:

```bash
python scripts/run_first_result.py --glade-raw <path_to_glade_plus_file>
```

**One-liner example:**

```bash
python scripts/run_first_result.py --glade-raw data/GLADE+.txt
```

This will automatically:
1. Create / update the events catalog from GWTC-3-confident (if missing)
2. Fetch skymaps for the best-localized events (configurable via `--top-n`)
3. Extract **3D distance PDFs** from HEALPix FITS skymaps
4. Choose an event with a valid 3D skymap
5. Ingest the GLADE+ catalog (if `--glade-raw` is provided)
6. Run H0 inference (STD vs ELL) and save a plot to `figures/first_h0_post_<event>.png`

**Notes:**
- If the minimal catalog already exists at `data/glade_plus_min.parquet`, you may omit `--glade-raw`
- `run_first_result.py` supports GLADE+ in FITS/CSV/ASCII, but always converts to a minimal Parquet file for analysis
- If 3D distance information is unavailable for an event, the script will move on to the next candidate

### Manual Workflow (Step-by-Step)

1. **List events from GWTC-3-confident catalog**

```bash
python scripts/list_events.py --max-events 10
```

2. **Fetch and extract skymaps for top-N localized events**

```bash
python scripts/fetch_skymaps.py --top-n 3
```

This downloads Zenodo tarballs and extracts one HEALPix FITS skymap per event into `data/skymaps/<event>/`.

3. **Plot skymaps and extract distance PDFs**

```bash
python scripts/plot_skymaps.py --distance
```

This:
- Visualizes sky localization for each event
- Reads 3D skymap columns (`PROBDENSITY`, `DISTMU`, `DISTSIGMA`, `DISTNORM`)
- Builds a **radial distance posterior** and a prior-removed **distance likelihood**
- Saves diagnostic distance plots in `figures/skymaps/`

4. **Run H0 inference for a single event (STD vs ELL, 3D skymap)**

```bash
python scripts/one_event_h0.py \
  --event GW191204_171526-v1 \
  --model both \
  --h0-min 20 --h0-max 120 --h0-n 201 \
  --compare-3d \
  --scan-kappa
```

This will:
- Use the 3D skymap likelihood \(p(\Omega, r)\) by default
- Compute H0 posteriors for the **STD** model and the **ELL** model (with several `kappa` values when `--scan-kappa` is enabled)
- Optionally compare separable vs full 3D skymap mode on the same figure when `--compare-3d` is passed
- Save the H0 posterior figure in `figures/`

**Requirements:**
- A galaxy catalog at `data/glade_plus_min.parquet` with at least:
  - `ra_deg`, `dec_deg`, `z` and (optionally) `weight`
- A 3D skymap for the event in `data/skymaps/<event>/skymap.fits(.gz)`

### Depth Scan (Catalog Depth vs H0)

To study how the inferred H0 depends on the maximum catalog redshift \(z_{\max}\) for a given event, use:

```bash
python scripts/depth_scan_h0.py \
  --event GW191204_171526-v1 \
  --zmax-list 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5 \
  --h0-min 20 --h0-max 120 --h0-n 201
```

This script:
- Uses the **3D skymap likelihood** for the specified event
- For each `zmax` filters the galaxy catalog to \(0 < z \le z_{\max}\)
- Computes H0 posteriors for:
  - STD 3D model
  - ELL 3D model (with `kappa = 1.0`)
- Extracts MAP and 16/50/84% quantiles
- Saves:
  - Posterior depth-scan results to `results/h0_depth_scan_<event>.csv`
  - Raw posteriors (grid × zmax × model) to `results/h0_depth_scan_<event>_posteriors.npz`
  - A figure `figures/h0_depth_scan_<event>.png` showing:
    - H0\_MAP vs `zmax` for STD/ELL
    - \(\Delta H_0 = H_0^\mathrm{ELL} - H_0^\mathrm{STD}\) vs `zmax`

### H0 Window Sensitivity (CMB vs Local Priors)

To test robustness of the ELL vs STD difference to the choice of H0 windows (CMB-like vs Local-like), use:

```bash
python scripts/window_sensitivity_h0.py \
  --event GW191204_171526-v1
```

By default this:
- Loads `results/h0_depth_scan_<event>_posteriors.npz`
- Scans window grids:
  - CMB centers: [66, 67, 68, 69]
  - Local centers: [71, 72, 73, 74]
  - Half-widths: [1.0, 1.5, 2.0] km/s/Mpc
- For each combination of:
  - catalog depth `zmax`
  - model ∈ {STD 3D, ELL 3D}
  - CMB window (center, width)
  - Local window (center, width)
  computes:
  - \(P_\mathrm{CMB} = \int_{\text{CMB window}} p(H_0)\, dH_0\)
  - \(P_\mathrm{Local} = \int_{\text{Local window}} p(H_0)\, dH_0\)
  - \(\log_{10} R = \log_{10}(P_\mathrm{CMB} / P_\mathrm{Local})\)

Outputs:
- `results/h0_window_sensitivity_<event>.csv` with columns:
  - `zmax, model, cmb_center, cmb_width, local_center, local_width, P_CMB, P_Local, logR`
- `figures/h0_window_sensitivity_vs_depth_<event>.png`:
  - median \(\log_{10} R\) vs `zmax` for STD 3D and ELL 3D
  - shaded 16–84% percentile bands over window choices

You can reproduce the **original fixed windows** CMB=[65,70], Local=[70,75] via:

```bash
python scripts/window_sensitivity_h0.py \
  --event GW191204_171526-v1 \
  --cmb-centers 67.5 \
  --local-centers 72.5 \
  --halfwidths 2.5
```

### Other Scripts

- `scripts/fig_predicted_drift.py`: Generate a toy-model / predicted H0 depth-drift figure from cosmology-only assumptions.
- `scripts/fig_skymap_demo.py`: Generate a demonstration skymap figure for a GW event:

  ```bash
  python scripts/fig_skymap_demo.py GW200311_115853
  ```

## Project Structure

- `src/epjc_h0/`: Main package with cosmology calculations, GWOSC I/O, likelihoods, and utilities
- `scripts/`: Runnable scripts
  - `run_first_result.py`: one-command pipeline to generate the first H0 result
  - `list_events.py`: query GWOSC and create the events CSV
  - `fetch_skymaps.py`: download and extract 2D/3D skymap FITS files
  - `plot_skymaps.py`: visualize skymaps and build distance PDFs from 3D FITS
  - `one_event_h0.py`: single-event H0 inference (STD vs ELL; 3D vs separable comparison)
  - `depth_scan_h0.py`: H0 vs catalog depth \(z_{\max}\) analysis (3D skymap)
  - `window_sensitivity_h0.py`: H0 window sensitivity (CMB vs Local) analysis
  - `ingest_glade_plus_min.py`: create minimal GLADE+ parquet from raw catalog
- `data/`: data files (events CSV, galaxy catalog, skymaps, manifest)
- `figures/`: generated output figures
- `results/`: CSV/NPZ outputs for H0 vs depth and window sensitivity

## Dependencies

- **Required**: `numpy`, `scipy`, `matplotlib`, `pandas`, `requests`, `gwosc`, `astropy`, `astropy-healpix`
- **Optional**: `ligo-skymap` (enhanced plotting / distance tools)

All dependencies are listed in `requirements.txt`.

## GLADE+ Galaxy Catalog

H0 inference requires a galaxy catalog. The GLADE+ catalog is recommended.

### Download GLADE+

**Option 1: Official website (ASCII, \~6 GB)**

1. Go to `https://glade.elte.hu/`
2. Download `GLADE+.txt`
3. Save to `data/glade_plus_raw.txt`

**Option 2: VizieR (FITS/CSV)**

1. Go to `https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=VII/291`
2. Click "Save in CDS portal"
3. Choose FITS (binary) or CSV format
4. Save to `data/glade_plus_raw.fits` or `data/glade_plus_raw.csv`

### Ingest GLADE+

After downloading, create the minimal Parquet catalog:

```bash
python scripts/ingest_glade_plus_min.py data/glade_plus_raw.txt
```

Or let `run_first_result.py` ingest it on the fly:

```bash
python scripts/run_first_result.py --glade-raw data/glade_plus_raw.txt
```

The ingestion script will:
- Map input columns to `ra_deg`, `dec_deg`, `z`, (optional) `weight`
- Filter to valid redshifts (e.g. \(0 < z < 10\))
- Create `data/glade_plus_min.parquet` used by all analysis scripts

**Note:** The full GLADE+ catalog is large (\~6 GB). The minimal Parquet (\~few hundred MB) keeps only the columns required for H0 inference.

## H0 Inference (Current Status)

The `one_event_h0.py` and `depth_scan_h0.py` scripts implement the current H0 inference pipeline:

- Full **3D skymap likelihood** \(p(\Omega, r)\) using `PROBDENSITY`, `DISTMU`, `DISTSIGMA`, `DISTNORM`
- Models:
  - **STD** (standard luminosity distance)
  - **ELL** (modified propagation with parameter `kappa`)
- Proper handling of the GW distance prior (removal of \(d^2\) volumetric prior when needed)
- Correct Jacobian \(d d_L / dz\) for both STD and ELL models

**Current limitations:**
- No selection effects modeling yet
- No explicit out-of-catalog term
- No galaxy completeness corrections
- Single-event analysis (multi-event combination left for future work)

## Reproducibility Checklist (Zenodo)

Before archiving on Zenodo, you can verify:

1. **Environment**
   - `pip install -r requirements.txt`

2. **First result**
   - `python scripts/run_first_result.py --glade-raw data/glade_plus_raw.txt`

3. **Single-event H0 (STD vs ELL, 3D vs separable)**
   - `python scripts/one_event_h0.py --event GW191204_171526-v1 --model both --compare-3d --scan-kappa`

4. **Depth scan**
   - `python scripts/depth_scan_h0.py --event GW191204_171526-v1 --zmax-list 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5`

5. **H0 window sensitivity**
   - `python scripts/window_sensitivity_h0.py --event GW191204_171526-v1`

If all of the above complete and produce figures / CSVs in `figures/` and `results/`, the README is consistent with the current code and the archive is ready for Zenodo.
