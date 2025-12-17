import json
import os
import shutil
import sys
from glob import glob
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.bids import save_glm_to_bids

sys.path.append("..")
from processing.utils import events_to_rtdur


if __name__ == "__main__":
    # ---------- CONFIG ----------
    bids_root = Path("/cbica/projects/executive_function/mebold_trt/ds005250")
    derivatives_dir = Path("/cbica/projects/executive_function/mebold_trt/derivatives")
    fmriprep_dir = derivatives_dir / "nordic_fmriprep_unzipped" / "fmriprep"
    tedana_dir = derivatives_dir / "tedana"
    out_dir = Path("/cbica/projects/executive_function/mebold_trt/derivatives/fracback")
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted(bids_root.glob("sub-*"))
    for subject_dir in subject_dirs:
        sub_id = os.path.basename(subject_dir).split("-")[1]
        session_dirs = sorted(glob(f"{subject_dir}/ses-*"))
        for session_dir in session_dirs:
            ses_id = os.path.basename(session_dir).split("-")[1]
            print(f"Running first-level GLM for subject: {sub_id} and session: {ses_id}")

            bids_func_dir = bids_root / f"sub-{sub_id}" / f"ses-{ses_id}" / "func"
            fmriprep_func_dir = fmriprep_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "func"
            tedana_func_dir = tedana_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "func"
            prefix = f"sub-{sub_id}_ses-{ses_id}_task-fracback_acq-MBME"

            # ---------- Preprocessed data from fMRIPrep ----------
            preproc_file = (
                fmriprep_func_dir / f"{prefix}_part-mag_space-MNI152NLin6Asym_res-2_desc-preproc_bold.nii.gz"
            )
            if not preproc_file.exists():
                print(
                    f"\tPreprocessed file not found for subject: {sub_id} and session: {ses_id}\n"
                    f"\t{preproc_file}"
                )
                continue

            preproc_img = nb.load(preproc_file)

            preproc_json = str(preproc_file).replace(".nii.gz", ".json")
            with open(preproc_json, "r") as f:
                preproc_json_data = json.load(f)
            t_r = preproc_json_data["RepetitionTime"]
            slice_time_ref = preproc_json_data["StartTime"]

            mask_file = fmriprep_func_dir / f"{prefix}_part-mag_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz"
            if not mask_file.exists():
                print(f"\tMask file not found for subject: {sub_id} and session: {ses_id}")
                continue

            mask_img = str(mask_file)

            # ---------- Dummy volumes from fMRIPrep ----------
            confounds_file = fmriprep_func_dir / f"{prefix}_part-mag_desc-confounds_timeseries.tsv"
            if not confounds_file.exists():
                print(f"\tConfounds file not found for subject: {sub_id} and session: {ses_id}")
                continue

            fmriprep_confounds_df = pd.read_table(confounds_file)
            # Infer the number of dummy volumes from the confounds dataframe
            nss_cols = [c for c in fmriprep_confounds_df.columns if c.startswith("non_steady_state_outlier")]

            dummy_scans = 0
            if nss_cols:
                initial_volumes_df = fmriprep_confounds_df[nss_cols]
                dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
                dummy_scans = np.where(dummy_scans)[0]

                # reasonably assumes all NSS volumes are contiguous
                dummy_scans = int(dummy_scans[-1] + 1)

            print(f"\t{dummy_scans} dummy scans")

            # ---------- Events from raw BIDS ----------
            events_file = bids_func_dir / f"{prefix}_events.tsv"
            if not events_file.exists():
                print(f"\tEvents file not found for subject: {sub_id} and session: {ses_id}")
                continue

            events_df = pd.read_table(events_file)
            cons_dur_rt_dur_events_df = events_to_rtdur(events_df)

            # ---------- Confounds from TEDANA ----------
            tedana_confounds = tedana_func_dir / f"{prefix}_desc-rejected_timeseries.tsv"
            if not tedana_confounds.exists():
                print(f"\tTedana classifications file not found for subject: {sub_id} and session: {ses_id}")
                continue
            confounds_df = pd.read_table(tedana_confounds)

            # ---------- Remove dummy volumes if necessary ----------
            if dummy_scans > 0:
                cons_dur_rt_dur_events_df["onset"] = cons_dur_rt_dur_events_df["onset"] - (dummy_scans * t_r)
                cons_dur_rt_dur_events_df = cons_dur_rt_dur_events_df.loc[cons_dur_rt_dur_events_df["onset"] >= 0].reset_index(drop=True)
                preproc_img = preproc_img.slicer[..., dummy_scans:]
                confounds_df = confounds_df.loc[dummy_scans:].reset_index(drop=True)

            # ---------- Fit GLM ----------
            model = FirstLevelModel(
                t_r=t_r,
                slice_time_ref=slice_time_ref,
                hrf_model="glover",
                drift_model="cosine",
                high_pass=None,
                mask_img=mask_img,
                smoothing_fwhm=5,
                noise_model="ar1",
                minimize_memory=False,
            )
            model = model.fit(
                run_imgs=preproc_img,
                events=cons_dur_rt_dur_events_df,
                confounds=confounds_df,
            )

            # Inspect design matrix
            design_matrix = model.design_matrices_[0]
            print("\tDesign matrix columns:")
            print("\t\t", design_matrix.columns)
            print(f"\tTotal # regressors in design matrix: {design_matrix.shape[1]}")

            func_out_dir = out_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / "func"
            func_out_dir.mkdir(parents=True, exist_ok=True)
            save_glm_to_bids(
                model,
                contrasts="two_back - zero_back",
                contrast_types={"two_back - zero_back": "t"},
                out_dir=func_out_dir,
                prefix=prefix,
                bg_img=(
                    "/cbica/projects/executive_function/.cache/templateflow/"
                    "tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"
                ),
            )
            print(f"\tDone fitting GLM for subject: {sub_id} and session: {ses_id}")

            # Post-Nilearn cleanup
            nilearn_func_out_dir = func_out_dir / f"sub-{sub_id}"
            dataset_description_file = out_dir / "dataset_description.json"
            nilearn_dataset_description_file = func_out_dir / "dataset_description.json"
            if not dataset_description_file.exists():
                shutil.copyfile(nilearn_dataset_description_file, dataset_description_file)

            os.remove(nilearn_dataset_description_file)
            # Move contents of nilearn_func_out_dir to func_out_dir
            for item in nilearn_func_out_dir.iterdir():
                shutil.move(item, func_out_dir / item.name)
            nilearn_func_out_dir.rmdir()

    print("\n----\nDONE\n----\n")
