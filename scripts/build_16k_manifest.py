"""Build the 16kHz nkululeko manifest after running nkululeko's own resampler.

Run this AFTER:
    python -m nkululeko.resample --folder data/persian
(no --replace!). That CLI mode is the safe one -- see the note in
run_wav2vec2_layers.sh: nkululeko's --config-driven RESAMPLE.replace is
mishandled in nkululeko <= main (2026-09-17) and always overwrites files in
place regardless of what the ini says (config_val() returns the string
"False", and a non-empty string is truthy), which is the opposite of the
documented behaviour and would destroy the original 44.1kHz corpus. The
--folder CLI mode is unaffected: argparse gives it a real bool, so leaving
--replace off reliably writes new "<name>_16kHz.wav" files next to the
originals instead of overwriting them.

This script just rewrites scripts/persian_nkululeko_data.csv to point at
those "<name>_16kHz.wav" siblings (nkululeko's own naming convention, see
Resampler.resample() in nkululeko/augmenting/resampler.py), keeping the same
speaker/task columns, and fails loudly if any expected file is missing.
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CSV = os.path.join(REPO_ROOT, "scripts", "persian_nkululeko_data.csv")
DST_CSV = os.path.join(REPO_ROOT, "scripts", "persian_nkululeko_data_16k.csv")


def sixteen_k_path(path):
    return os.path.splitext(path)[0] + "_16kHz.wav"


def main():
    df = pd.read_csv(SRC_CSV)
    new_files = df["file"].apply(sixteen_k_path)
    missing = [f for f in new_files if not os.path.isfile(f)]
    if missing:
        print(f"ERROR: {len(missing)} resampled file(s) missing, e.g.:", file=sys.stderr)
        for f in missing[:5]:
            print(f"  {f}", file=sys.stderr)
        print(
            "Did you run: python -m nkululeko.resample --folder "
            f"{os.path.join(REPO_ROOT, 'data', 'persian')}  (without --replace)?",
            file=sys.stderr,
        )
        sys.exit(1)
    df["file"] = new_files
    df.to_csv(DST_CSV, index=False)
    print(f"wrote {DST_CSV} ({len(df)} rows)")


if __name__ == "__main__":
    main()
