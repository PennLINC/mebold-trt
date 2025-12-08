"""Run tedana using fMRIPrep outputs and task regressors."""

import json
import os
from glob import glob

import nibabel as nb
import numpy as np
import pandas as pd
from tedana.workflows import tedana_workflow


def run_tedana(raw_dir, fmriprep_dir, temp_dir, tedana_out_dir):
    print("TEDANA")

    base_search = os.path.join(
        raw_dir,
        "sub-*",
        "ses-*",
        "func",
        "sub-*_ses-*_echo-1_part-mag_bold.nii.gz",
    )
    base_files = sorted(glob(base_search))
    if not base_files:
        raise FileNotFoundError(base_search)

    for base_file in base_files:
        raw_files = sorted(glob(base_file.replace("echo-1", "echo-*")))

        base_filename = os.path.basename(base_file)
        print(f"\t{base_filename}")
        subject = base_filename.split("_")[0]
        session = base_filename.split("_")[1]
        prefix = base_filename.split("_echo-1")[0]

        # Get the fMRIPrep brain mask
        mask_base = base_filename.split("_echo-1")[0]
        mask = os.path.join(
            fmriprep_dir,
            subject,
            session,
            "func",
            f"{mask_base}_part-mag_desc-brain_mask.nii.gz",
        )
        assert os.path.isfile(mask), mask

        # Get the fMRIPrep confounds file and identify the number of non-steady-state volumes
        confounds_file = os.path.join(
            fmriprep_dir,
            subject,
            session,
            "func",
            f"{mask_base}_part-mag_desc-confounds_timeseries.tsv",
        )
        confounds_df = pd.read_table(confounds_file)
        nss_cols = [c for c in confounds_df.columns if c.startswith("non_steady_state_outlier")]

        dummy_scans = 0
        if nss_cols:
            initial_volumes_df = confounds_df[nss_cols]
            dummy_scans = np.any(initial_volumes_df.to_numpy(), axis=1)
            dummy_scans = np.where(dummy_scans)[0]

            # reasonably assumes all NSS volumes are contiguous
            dummy_scans = int(dummy_scans[-1] + 1)

        print(f"\t\t{dummy_scans} dummy scans")

        echo_times = []
        fmriprep_files = []
        for raw_file in raw_files:
            base_query = os.path.basename(raw_file).split("_bold.nii.gz")[0]

            # Get echo time from json file
            with open(raw_file.replace(".nii.gz", ".json"), "r") as f:
                echo_times.append(json.load(f)["EchoTime"] * 1000)

            # Get the fMRIPrep BOLD files
            fmriprep_file = os.path.join(
                fmriprep_dir,
                subject,
                session,
                "func",
                f"{base_query}_desc-preproc_bold.nii.gz",
            )
            assert os.path.isfile(fmriprep_file), fmriprep_file

            # Remove non-steady-state volumes
            echo_img = nb.load(fmriprep_file)
            echo_img = echo_img.slicer[:, :, :, dummy_scans:]
            temporary_file = os.path.join(
                temp_dir,
                os.path.basename(fmriprep_file),
            )
            echo_img.to_filename(temporary_file)
            fmriprep_files.append(temporary_file)

        tedana_run_out_dir = os.path.join(tedana_out_dir, subject, session, "func")
        os.makedirs(tedana_run_out_dir, exist_ok=True)
        if os.path.isfile(os.path.join(tedana_run_out_dir, f"{prefix}_tedana_report.html")):
            print(f"DONE: {prefix}")
            continue

        tree = "tedana_minimal_rest.json"
        if "task-fracback" in prefix:
            tree = "tedana_minimal_task.json"

        tedana_workflow(
            data=fmriprep_files,
            tes=echo_times,
            mask=mask,
            out_dir=tedana_run_out_dir,
            prefix=prefix,
            fittype="curvefit",
            combmode="t2s",
            tree=tree,
            gscontrol=["mir"],
            tedort=True,
        )


if __name__ == "__main__":
    raw_dir_ = "/cbica/projects/executive_function/mebold-trt/dset"
    fmriprep_dir_ = "/cbica/projects/executive_function/mebold-trt/derivatives/fmriprep"
    temp_dir_ = "/cbica/comp_space/executive_function/tedana_temp"
    tedana_out_dir_ = "/cbica/projects/executive_function/mebold-trt/derivatives/tedana"

    os.makedirs(temp_dir_, exist_ok=True)

    run_tedana(raw_dir_, fmriprep_dir_, temp_dir_, tedana_out_dir_)
