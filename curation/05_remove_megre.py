"""Remove MEGRE scans from scans.tsv files."""

import os
from glob import glob

import pandas as pd

if __name__ == "__main__":
    dset_dir = "/cbica/projects/executive_function/mebold_trt/dset/"
    subject_dirs = sorted(glob(os.path.join(dset_dir, "sub-*")))
    for subject_dir in subject_dirs:
        session_dirs = sorted(glob(os.path.join(subject_dir, "ses-*")))
        for session_dir in session_dirs:
            scans_file = sorted(glob(os.path.join(session_dir, "*_scans.tsv")))
            if len(scans_file) != 1:
                raise ValueError(f"scans file not found: {scans_file}")

            scans_file = scans_file[0]
            scans_df = pd.read_table(scans_file)
            scans_df2 = scans_df.loc[~scans_df["filename"].str.contains("MEGRE")]
            os.remove(scans_file)
            scans_df2.to_csv(scans_file, sep="\t", na_rep="n/a", index=False)
            print(f"Processed {os.path.basename(scans_file)}")
