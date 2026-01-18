"""Create minimal GLADE+ parquet from raw GLADE+ file."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def _process_glade_incremental(raw_path: Path, out_path: Path) -> bool:
    """
    Process GLADE+ file incrementally: read chunks, filter, write to parquet.
    This avoids loading entire file into memory.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove output file if exists
    if out_path.exists():
        out_path.unlink()
    
    total_processed = 0
    total_valid = 0
    batch_size = 500000  # Process 500k rows at a time
    write_batch_size = 5  # Write every 5 chunks to reduce I/O and memory
    
    chunk_buffer = []
    first_write = True
    
    try:
        print(f"  Processing in chunks of {batch_size:,} rows...")
        print(f"  Writing batches of {write_batch_size} chunks to reduce I/O...")
        
        for chunk_idx, chunk in enumerate(pd.read_csv(
            raw_path, 
            sep=r'\s+', 
            header=None, 
            engine='python',
            chunksize=batch_size,
            usecols=[8, 9, 27],  # Read only needed columns: RA, Dec, z
            dtype={8: 'float64', 9: 'float64', 27: 'float64'}  # Direct numeric types
        )):
            total_processed += len(chunk)
            
            # Rename columns
            chunk.columns = ['ra_deg', 'dec_deg', 'z']
            
            # Filter invalid values (NaN, inf, out of range)
            mask = (
                chunk['ra_deg'].notna() & 
                chunk['dec_deg'].notna() & 
                chunk['z'].notna() &
                (chunk['z'] > 0) & 
                (chunk['z'] < 10) &
                np.isfinite(chunk['ra_deg']) & 
                np.isfinite(chunk['dec_deg']) & 
                np.isfinite(chunk['z'])
            )
            
            chunk_filtered = chunk[mask].copy()
            chunk_filtered['weight'] = 1.0  # Add weight column
            
            if len(chunk_filtered) > 0:
                chunk_buffer.append(chunk_filtered)
                total_valid += len(chunk_filtered)
            
            # Write in batches to reduce I/O overhead
            if len(chunk_buffer) >= write_batch_size:
                df_batch = pd.concat(chunk_buffer, ignore_index=True)
                
                # Write to parquet (append if not first write)
                if first_write:
                    df_batch.to_parquet(out_path, index=False)
                    first_write = False
                else:
                    # Read existing, append, write back
                    df_existing = pd.read_parquet(out_path)
                    df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
                    df_combined.to_parquet(out_path, index=False)
                    del df_existing, df_combined  # Free memory
                
                del df_batch  # Free memory
                chunk_buffer = []  # Clear buffer
            
            # Progress update
            if (chunk_idx + 1) % 10 == 0:
                pct = 100 * total_valid / total_processed if total_processed > 0 else 0
                print(f"    Processed {total_processed:,} rows, {total_valid:,} valid ({pct:.1f}%)")
        
        # Write remaining chunks
        if len(chunk_buffer) > 0:
            df_batch = pd.concat(chunk_buffer, ignore_index=True)
            if first_write:
                df_batch.to_parquet(out_path, index=False)
            else:
                df_existing = pd.read_parquet(out_path)
                df_combined = pd.concat([df_existing, df_batch], ignore_index=True)
                df_combined.to_parquet(out_path, index=False)
                del df_existing, df_combined
            del df_batch
            chunk_buffer = []
        
        print(f"  ✓ Processed {total_processed:,} rows total")
        print(f"  ✓ Valid galaxies: {total_valid:,}")
        print(f"  ✓ Created minimal catalog: {out_path}")
        
        return total_valid > 0
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


def ingest_glade_plus(raw_path: Path, out_path: Path) -> bool:
    """
    Create minimal GLADE+ parquet from raw file.
    
    Supports common GLADE+ formats (CSV, FITS, etc.)
    """
    if not raw_path.exists():
        print(f"ERROR: GLADE+ file not found: {raw_path}")
        return False
    
    print(f"Reading GLADE+ file: {raw_path}")
    
    try:
        # Try different formats
        if raw_path.suffix in ['.fits', '.fit']:
            from astropy.io import fits
            with fits.open(str(raw_path)) as hdul:
                hdu = hdul[1]  # Usually data is in HDU 1
                df = pd.DataFrame(hdu.data)
        elif raw_path.suffix == '.parquet':
            df = pd.read_parquet(raw_path)
        else:
            # Try space-separated format (GLADE+ ASCII format)
            # GLADE+ format: space-separated, columns: GLADE+, PGC, GWGC, HyperLEDA, 2MASS, WISExSCOS, SDSS-DR16Q, Type,
            # RA (col 9), Dec (col 10), ..., Redshift heliocentric (col 28), ...
            try:
                # Optimized: incremental processing and writing to avoid memory issues
                print(f"  Reading space-separated format with incremental processing...")
                return _process_glade_incremental(raw_path, out_path)
                
            except Exception as e:
                print(f"  Failed to read as space-separated: {e}")
                import traceback
                traceback.print_exc()
                # Fallback to regular CSV (but this will also have memory issues)
                print(f"  WARNING: Falling back to regular CSV (may have memory issues)")
                df = pd.read_csv(raw_path, low_memory=False)
        
        # Fallback path for non-space-separated formats
        print(f"  Loaded {len(df)} rows")
        print(f"  Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        
        # If columns are already named correctly, use them
        if 'ra_deg' in df.columns and 'dec_deg' in df.columns and 'z' in df.columns:
            print(f"  Found required columns: ra_deg, dec_deg, z")
        else:
            # Try to map common column names
            column_mapping = {
                # Common GLADE+ column names
                'RA': 'ra_deg',
                'ra': 'ra_deg',
                'RAJ2000': 'ra_deg',
                'ra_deg': 'ra_deg',
                'DEC': 'dec_deg',
                'dec': 'dec_deg',
                'DEJ2000': 'dec_deg',
                'dec_deg': 'dec_deg',
                'z': 'z',
                'zhelio': 'z',
                'redshift': 'z',
                'z_helio': 'z',
                'z_hel': 'z',
                'weight': 'weight',
                'w': 'weight'
            }
            
            # Rename columns (only if they exist)
            rename_dict = {k: v for k, v in column_mapping.items() if k in df.columns}
            if rename_dict:
                df = df.rename(columns=rename_dict)
                print(f"  Applied column mapping: {rename_dict}")
        
        # Check required columns
        required = ['ra_deg', 'dec_deg', 'z']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"ERROR: Missing required columns after mapping: {missing}")
            print(f"  Available columns: {list(df.columns)}")
            print(f"\nExpected column names (or aliases):")
            print(f"  RA/ra/RAJ2000 -> ra_deg")
            print(f"  DEC/dec/DEJ2000 -> dec_deg")
            print(f"  z/zhelio/redshift -> z")
            return False
        
        # Convert to numeric (handle 'null' strings)
        print(f"  Converting to numeric types...")
        df['ra_deg'] = pd.to_numeric(df['ra_deg'], errors='coerce')
        df['dec_deg'] = pd.to_numeric(df['dec_deg'], errors='coerce')
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        
        # Add weight if missing
        if 'weight' not in df.columns:
            df['weight'] = 1.0
            print(f"  Added default weight=1.0 column")
        
        # Filter to valid data
        initial_count = len(df)
        df = df[
            df['ra_deg'].notna() & df['dec_deg'].notna() & df['z'].notna() &
            (df['z'] > 0) & (df['z'] < 10) &
            np.isfinite(df['ra_deg']) & np.isfinite(df['dec_deg']) & np.isfinite(df['z'])
        ].copy()
        
        filtered_count = len(df)
        print(f"  Filtered: {initial_count} -> {filtered_count} valid galaxies")
        
        if filtered_count == 0:
            print(f"ERROR: No valid galaxies after filtering")
            return False
        
        # Select only needed columns
        df_min = df[['ra_deg', 'dec_deg', 'z', 'weight']].copy()
        
        # Save to parquet
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_min.to_parquet(out_path, index=False)
        
        print(f"✓ Created minimal catalog: {out_path}")
        print(f"  {len(df_min)} galaxies")
        print(f"  Columns: {list(df_min.columns)}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to ingest GLADE+: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Create minimal GLADE+ parquet from raw file"
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to raw GLADE+ file (FITS, CSV, or Parquet)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/glade_plus_min.parquet',
        help='Output parquet path (default: data/glade_plus_min.parquet)'
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    raw_path = Path(args.input)
    out_path = project_root / args.out
    
    if not ingest_glade_plus(raw_path, out_path):
        return 1
    
    print(f"\nDone!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
