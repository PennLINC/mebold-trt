"""Run tedana using fMRIPrep outputs and task regressors."""

import json
import os
from glob import glob

import nibabel as nb
import numpy as np
import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from tedana.workflows import tedana_workflow


MOTION_COLUMNS = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z"]


def events_to_rtdur(events_df):
    """Implement Jeanette Mumford's ConsDurRTDur model on an events dataframe."""
    # Limit to 0back and 2back trials
    events_df = events_df.loc[
        events_df["trial_type"].str.lower().isin(["0back", "2back"])
    ].copy()
    events_df = events_df[["onset", "duration", "trial_type", "response_time"]]

    # Normalize casing then rename to valid Python identifiers
    events_df["trial_type"] = events_df["trial_type"].str.lower()
    events_df["trial_type"] = events_df["trial_type"].replace(
        {
            "0back": "zero_back",
            "2back": "two_back",
        }
    )

    response_events_df = events_df.loc[~np.isnan(events_df["response_time"])].copy()
    response_events_df.loc[:, "duration"] = response_events_df.loc[:, "response_time"]
    response_events_df.loc[:, "trial_type"] = "RTDur"

    cons_dur_rt_dur_events_df = pd.concat((events_df, response_events_df))
    cons_dur_rt_dur_events_df = cons_dur_rt_dur_events_df.sort_values(by="onset")
    cons_dur_rt_dur_events_df = cons_dur_rt_dur_events_df.reset_index(drop=True)
    return cons_dur_rt_dur_events_df


def build_motion_confounds(confounds_df):
    missing = [c for c in MOTION_COLUMNS if c not in confounds_df.columns]
    if missing:
        raise KeyError(f"Missing motion columns in confounds file: {missing}")

    return confounds_df[MOTION_COLUMNS]


def build_fracback_regressors(events_file, frame_times):
    events_df = pd.read_table(events_file)
    events_df = events_to_rtdur(events_df)

    if events_df.empty:
        zeros = np.zeros((len(frame_times), 3))
        return pd.DataFrame(zeros, columns=["zero_back", "two_back", "RTDur"])

    design_matrix = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events_df,
        hrf_model="glover",
        drift_model=None,
    )

    return design_matrix


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
        tr = None
        n_volumes = None

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
        nss_cols = [
            c for c in confounds_df.columns if c.startswith("non_steady_state_outlier")
        ]

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
            if tr is None:
                tr = echo_img.header.get_zooms()[3]
            if n_volumes is None:
                n_volumes = echo_img.shape[-1]
            temporary_file = os.path.join(
                temp_dir,
                os.path.basename(fmriprep_file),
            )
            echo_img.to_filename(temporary_file)
            fmriprep_files.append(temporary_file)

        if tr is None or n_volumes is None:
            raise RuntimeError(
                f"Unable to determine TR or volume count for {base_filename}"
            )

        motion_confounds = build_motion_confounds(confounds_df)

        if len(motion_confounds) != n_volumes:
            raise ValueError(
                f"Motion confounds ({len(motion_confounds)}) do not match truncated volumes ({n_volumes})"
            )

        confounds = motion_confounds

        if "task-fracback" in prefix:
            events_file = base_file.replace(
                "_echo-1_part-mag_bold.nii.gz", "_events.tsv"
            )
            assert os.path.isfile(events_file), events_file

            frame_times = np.arange(n_volumes) * tr
            fracback_confounds = build_fracback_regressors(events_file, frame_times)

            confounds = pd.concat(
                [motion_confounds.reset_index(drop=True), fracback_confounds], axis=1
            )

        tedana_run_out_dir = os.path.join(tedana_out_dir, subject, session, "func")
        os.makedirs(tedana_run_out_dir, exist_ok=True)
        if os.path.isfile(
            os.path.join(tedana_run_out_dir, f"{prefix}_tedana_report.html")
        ):
            print(f"DONE: {prefix}")
            continue

        tree = "tedana_minimal_rest.json"
        if "task-fracback" in prefix:
            tree = "tedana_minimal_task.json"

        confounds_file = os.path.join(tedana_run_out_dir, f"{prefix}_confounds.tsv")
        confounds.to_csv(confounds_file, sep="\t", index=False)

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
            external_regressors=confounds_file,
        )


if __name__ == "__main__":
    raw_dir_ = "/cbica/projects/executive_function/mebold-trt/dset"
    fmriprep_dir_ = "/cbica/projects/executive_function/mebold-trt/derivatives/fmriprep"
    temp_dir_ = "/cbica/comp_space/executive_function/tedana_temp"
    tedana_out_dir_ = "/cbica/projects/executive_function/mebold-trt/derivatives/tedana"

    os.makedirs(temp_dir_, exist_ok=True)

    run_tedana(raw_dir_, fmriprep_dir_, temp_dir_, tedana_out_dir_)
