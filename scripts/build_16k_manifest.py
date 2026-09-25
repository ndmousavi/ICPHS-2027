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

This script rewrites scripts/<corpus>_nkululeko_data.csv to point at those
"<name>_16kHz.wav" siblings (nkululeko's own naming convention, see
Resampler.resample() in nkululeko/augmenting/resampler.py), keeping the same
speaker/task columns, and fails loudly if any expected file is missing.

Usage: python scripts/build_16k_manifest.py <corpus>   (e.g. persian, german)
"""

import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sixteen_k_path(path):
    return os.path.splitext(path)[0] + "_16kHz.wav"


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python build_16k_manifest.py <corpus>  (e.g. persian, german)")
    corpus = sys.argv[1]
    src_csv = os.path.join(REPO_ROOT, "scripts", f"{corpus}_nkululeko_data.csv")
    dst_csv = os.path.join(REPO_ROOT, "scripts", f"{corpus}_nkululeko_data_16k.csv")

    df = pd.read_csv(src_csv)
    new_files = df["file"].apply(sixteen_k_path)
    missing = [f for f in new_files if not os.path.isfile(f)]
    if missing:
        print(f"ERROR: {len(missing)} resampled file(s) missing, e.g.:", file=sys.stderr)
        for f in missing[:5]:
            print(f"  {f}", file=sys.stderr)
        print(
            "Did you run: python -m nkululeko.resample --folder "
            f"{os.path.join(REPO_ROOT, 'data', corpus)}  (without --replace)?",
            file=sys.stderr,
        )
        sys.exit(1)
    df["file"] = new_files
    df.to_csv(dst_csv, index=False)
    print(f"wrote {dst_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
