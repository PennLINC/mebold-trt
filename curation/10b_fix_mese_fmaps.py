#!/cbica/home/salot/miniconda3/envs/salot/bin/python
"""Fix BIDS files after heudiconv conversion.

Copy first echo of each multi-echo field map without echo entity.
"""

import os
import shutil
from glob import glob

import pandas as pd


if __name__ == "__main__":
    dset_dir = "/cbica/projects/executive_function/mebold_trt/dset/"
    subject_dirs = sorted(glob(os.path.join(dset_dir, "sub-*")))
    for subject_dir in subject_dirs:
        sub_id = os.path.basename(subject_dir)
        session_dirs = sorted(glob(os.path.join(subject_dir, "ses-*")))
        for session_dir in session_dirs:
            ses_id = os.path.basename(session_dir)
            fmap_dir = os.path.join(session_dir, "fmap")

            # Load scans file
            scans_file = os.path.join(session_dir, f"{sub_id}_{ses_id}_scans.tsv")
            assert os.path.isfile(scans_file), f"Scans file DNE: {scans_file}"
            scans_df = pd.read_table(scans_file)

            # Copy first echo's file of multi-echo spin echo field maps without echo entity.
            me_fmaps = sorted(glob(os.path.join(fmap_dir, "*_acq-MESE*echo-1_epi.*")))
            for me_fmap in me_fmaps:
                out_fmap = me_fmap.replace("_echo-1_", "_")
                if os.path.isfile(out_fmap):
                    print(f"File exists: {os.path.basename(out_fmap)}")
                    continue

                me_fmap_fname = os.path.join("fmap", os.path.basename(me_fmap))
                out_fmap_fname = os.path.join("fmap", os.path.basename(out_fmap))
                shutil.copyfile(me_fmap, out_fmap)
                if me_fmap.endswith(".nii.gz"):
                    i_row = len(scans_df.index)
                    scans_df.loc[i_row] = scans_df.loc[
                        scans_df["filename"] == me_fmap_fname
                    ].iloc[0]
                    scans_df.loc[i_row, "filename"] = out_fmap_fname

            # Save out the modified scans.tsv file.
            scans_df = scans_df.sort_values(by=["acq_time", "filename"])
            os.remove(scans_file)
            scans_df.to_csv(scans_file, sep="\t", na_rep="n/a", index=False)
