"""Build the nkululeko file,speaker,task manifest for a corpus.

Scans data/<corpus>/<speaker>/*.wav (case-insensitive extension -- corpora
here mix .wav/.WAV across speaker folders). Speaker id is the folder name;
task is the filename suffix after "<speaker>_", mapped from the corpus's own
shortcode convention (as used by data/german, e.g. "sc" -> con) to the
canonical labels already used throughout the Persian dataframes/scripts
(con, m(f), m(i), pic, r(f), r(i), str) when the suffix is a recognized
shortcode; a suffix that's already canonical (as in data/persian) is passed
through unchanged. Keeping task labels canonical across corpora is what lets
results from different languages be merged/compared later.

Usage: python scripts/build_manifest.py <corpus>   (e.g. persian, german)
Writes scripts/<corpus>_nkululeko_data.csv (file,speaker,task).
"""

import glob
import os
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SHORTCODE_MAP = {
    "sc": "con",
    "sp": "pic",
    "st": "str",
    "mf": "m(f)",
    "mi": "m(i)",
    "rf": "r(f)",
    "ri": "r(i)",
}
CANONICAL_TASKS = {"con", "pic", "str", "m(f)", "m(i)", "r(f)", "r(i)"}


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python build_manifest.py <corpus>  (e.g. persian, german)")
    corpus = sys.argv[1]
    corpus_dir = os.path.join(REPO_ROOT, "data", corpus)
    files = sorted(glob.glob(os.path.join(corpus_dir, "*", "*.[wW][aA][vV]")))
    # exclude nkululeko's own "<name>_16kHz.wav" resampled siblings (see
    # run_wav2vec2_layers.sh) if a previous resample run already created them
    files = [f for f in files if not f.endswith("_16kHz.wav")]
    if not files:
        sys.exit(f"no wav files found under {corpus_dir}")

    rows = []
    for f in files:
        speaker = os.path.basename(os.path.dirname(f))
        stem = os.path.splitext(os.path.basename(f))[0]
        prefix = speaker + "_"
        if not stem.startswith(prefix):
            sys.exit(f"filename {f!r} doesn't start with speaker id {speaker!r}")
        suffix = stem[len(prefix):]
        task = SHORTCODE_MAP.get(suffix, suffix)
        if task not in CANONICAL_TASKS:
            sys.exit(f"unrecognized task suffix {suffix!r} in {f}")
        rows.append({"file": os.path.abspath(f), "speaker": speaker, "task": task})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(REPO_ROOT, "scripts", f"{corpus}_nkululeko_data.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} rows, {df['speaker'].nunique()} speakers)")


if __name__ == "__main__":
    main()
