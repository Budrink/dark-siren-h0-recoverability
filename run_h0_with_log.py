"""Wrapper script to run H0 inference with explicit logging to file."""
import sys
import subprocess
from pathlib import Path

log_file = Path(__file__).parent / "logs_h0_inference.txt"

# Open log file for writing
with open(log_file, 'w', encoding='utf-8') as f:
    # Run the script and capture output
    process = subprocess.Popen(
        [sys.executable, "scripts/one_event_h0.py"] + sys.argv[1:],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
        cwd=Path(__file__).parent
    )
    
    # Write output to both file and console
    for line in process.stdout:
        print(line, end='', flush=True)
        f.write(line)
        f.flush()
    
    process.wait()
    f.write(f"\n\nExit code: {process.returncode}\n")
    f.flush()

print(f"\n\nLog saved to: {log_file}")
