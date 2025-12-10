from pathlib import Path

RAW_DIR = Path("/cbica/projects/executive_function/mebold_trt/ds005250")
PAIRS_TSV = Path(
    "/cbica/projects/executive_function/mebold_trt/code/processing/jobs/tedana_pairs.tsv"
)

pairs = sorted(
    (ses_dir.parent.name, ses_dir.name)
    for ses_dir in RAW_DIR.glob("sub-*/ses-*")
    if ses_dir.is_dir()
)

PAIRS_TSV.parent.mkdir(parents=True, exist_ok=True)
with PAIRS_TSV.open("w", encoding="utf-8") as f:
    f.write("subject\tsession\n")
    for sub, ses in pairs:
        f.write(f"{sub}\t{ses}\n")

print(f"Wrote {len(pairs)} rows to {PAIRS_TSV}")
