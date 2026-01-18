# EPJ C H0 Depth-Drift Analysis

Reproducible analysis project for testing apparent H0 "depth-drift" effects in gravitational wave cosmology.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

(Optional) For better skymap plotting and 3D distance extraction:

```bash
pip install ligo-skymap
```

## Usage

### Quick Start: One Command to First Result

To automatically run the full pipeline and generate the first H0 posterior plot:

```bash
python scripts/run_first_result.py --glade-raw <path_to_glade_plus_file>
```

**One-liner example:**
```bash
python scripts/run_first_result.py --glade-raw /path/to/GLADE_plus.fits
```

This will automatically:
1. Create events catalog (if missing)
2. Fetch skymaps for top-5 events (configurable with `--top-n`)
3. Extract distance PDFs
4. Choose the best localized event (smallest sky_area_90)
5. Ingest GLADE+ catalog (if `--glade-raw` provided)
6. Run H0 inference and save plot to `figures/first_h0_post_<event>.png`

**Note:** 
- If GLADE+ catalog is already at `data/glade_plus_min.parquet`, you can omit `--glade-raw`
- The script supports GLADE+ in FITS, CSV, or Parquet format
- If distance PDFs are not available, try increasing `--top-n` (some skymaps may be 2D-only)

### Manual Workflow

1. List events from GWTC-3-confident catalog:
```bash
python scripts/list_events.py --max-events 10
```

2. Fetch and extract skymaps for top-N localized events:
```bash
python scripts/fetch_skymaps.py --top-n 3
```

3. Plot skymaps:
```bash
python scripts/plot_skymaps.py
```

4. (Optional) Plot with additional features:
```bash
python scripts/plot_skymaps.py --distance --area --cmap viridis
```

5. (Optional) Run H0 inference for a single event:
```bash
# First, ensure distance PDF is available (from step 4)
python scripts/one_event_h0.py --event GW191204_171526-v1
```

Note: H0 inference requires a galaxy catalog (e.g., GLADE+) at `data/glade_plus_min.parquet` with columns: `ra_deg`, `dec_deg`, `z`, `weight` (weight optional).

### Other Scripts

Generate the predicted H0 drift figure:

```bash
python scripts/fig_predicted_drift.py
```

Generate the skymap demonstration figure for a GW event:

```bash
python scripts/fig_skymap_demo.py
```

Or specify a different event:

```bash
python scripts/fig_skymap_demo.py GW200311_115853
```

## Project Structure

- `src/epjc_h0/`: Main package with cosmology calculations, GWOSC I/O, and plotting utilities
- `scripts/`: Runnable scripts
  - `run_first_result.py`: **One-command pipeline** to generate first H0 result
  - `list_events.py`: Query GWOSC and create event catalog
  - `fetch_skymaps.py`: Download and extract skymap FITS files
  - `plot_skymaps.py`: Visualize skymaps (requires astropy or ligo-skymap)
  - `one_event_h0.py`: H0 inference for a single event (MVP)
  - `ingest_glade_plus_min.py`: Create minimal GLADE+ parquet from raw file
- `data/`: Data files (events CSV, skymaps, manifest)
- `figures/`: Generated output figures

## Dependencies

- **Required**: numpy, scipy, matplotlib, pandas, requests, gwosc, astropy, astropy-healpix
- **Optional**: ligo-skymap (for enhanced plotting and distance extraction)

## GLADE+ Galaxy Catalog

H0 inference requires a galaxy catalog. The GLADE+ catalog is recommended.

### Download GLADE+

**Option 1: Official website (ASCII format, ~6 GB)**
1. Go to: https://glade.elte.hu/
2. Download: `GLADE+.txt`
3. Save to: `data/glade_plus_raw.txt`

**Option 2: VizieR (FITS/CSV format)**
1. Go to: https://vizier.cds.unistra.fr/viz-bin/VizieR-2?-source=VII/291
2. Click "Save in CDS portal"
3. Choose format: FITS (binary) or CSV
4. Download and save to: `data/glade_plus_raw.fits` (or `.csv`)

### Ingest GLADE+

After downloading, create the minimal parquet catalog:

```bash
python scripts/ingest_glade_plus_min.py data/glade_plus_raw.txt
```

Or use directly with `run_first_result.py`:

```bash
python scripts/run_first_result.py --glade-raw data/glade_plus_raw.txt
```

The script will automatically:
- Map column names (RA→ra_deg, DEC→dec_deg, z→z)
- Filter valid galaxies (z > 0, z < 10)
- Create minimal parquet at `data/glade_plus_min.parquet`

**Note**: The full GLADE+ catalog is ~6 GB. The minimal parquet (~500 MB) contains only required columns: `ra_deg`, `dec_deg`, `z`, `weight`.

## H0 Inference (MVP)

The `one_event_h0.py` script implements a minimal H0 inference pipeline:

**Assumptions (MVP limitations):**
- Separable localization: p(Ω, d) ≈ p(Ω) * p(d)
- Ignores selection effects
- Ignores out-of-catalog term
- No galaxy completeness corrections

**Requirements:**
- Distance PDF must be extracted first: `python scripts/plot_skymaps.py --distance`
- Galaxy catalog (Parquet/CSV) with columns: `ra_deg`, `dec_deg`, `z`, `weight` (optional)

**Example:**
```bash
python scripts/one_event_h0.py --event GW191204_171526-v1 --model both --h0-min 50 --h0-max 100
```

## Next Steps

Future improvements will include:
- Selection effects modeling
- Out-of-catalog term
- Galaxy completeness corrections
- Multi-event combined analysis
- Statistical analysis of H0 depth-drift effects
