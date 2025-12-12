"""Run tedana using fMRIPrep outputs and task regressors."""

import argparse
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

    design_matrix = design_matrix[["zero_back", "two_back", "RTDur"]]

    return design_matrix


def _normalize_session_label(session_label):
    if session_label is None:
        return None
    return session_label if session_label.startswith("ses-") else f"ses-{session_label}"


def _normalize_subject_label(subject_label):
    if subject_label is None:
        return None
    return subject_label if subject_label.startswith("sub-") else f"sub-{subject_label}"


def run_tedana(
    raw_dir,
    fmriprep_dir,
    temp_dir,
    tedana_out_dir,
    session_label=None,
    subject_label=None,
):
    print("TEDANA")

    session_glob = _normalize_session_label(session_label) or "ses-*"
    subject_glob = _normalize_subject_label(subject_label) or "sub-*"

    base_search = os.path.join(
        raw_dir,
        subject_glob,
        session_glob,
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
                "_echo-1_part-mag_bold.nii.gz",
                "_events.tsv",
            )
            assert os.path.isfile(events_file), events_file

            frame_times = np.arange(n_volumes) * tr
            fracback_confounds = build_fracback_regressors(events_file, frame_times)

            confounds = pd.concat(
                [
                    motion_confounds.reset_index(drop=True),
                    fracback_confounds.reset_index(drop=True),
                ],
                axis=1,
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
            tedort=True,
            external_regressors=confounds_file,
            tedpca="mdl",
            dummy_scans=dummy_scans,
        )
        mixing = os.path.join(tedana_run_out_dir, f"{prefix}_desc-ICAOrth_mixing.tsv")
        mixing_df = pd.read_table(mixing)
        metrics = os.path.join(tedana_run_out_dir, f"{prefix}_desc-tedana_metrics.tsv")
        metrics_df = pd.read_table(metrics, index_col="Component")
        comps_rejected = metrics_df[metrics_df["classification"] == "rejected"].index.tolist()
        mixing_df = mixing_df[comps_rejected]
        if dummy_scans > 0:
            # Add dummy volumes to the rejected array
            rejected_arr = mixing_df.to_numpy()
            dummy_arr = np.zeros((dummy_scans, rejected_arr.shape[1]))
            rejected_arr = np.concatenate((dummy_arr, rejected_arr), axis=0)
            mixing_df = pd.DataFrame(rejected_arr, columns=mixing_df.columns)

        out_confounds = os.path.join(tedana_run_out_dir, f"{prefix}_desc-rejected_timeseries.tsv")
        mixing_df.to_csv(out_confounds, sep="\t", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="BIDS raw directory (expects sub-*/ses-*/func).",
    )
    parser.add_argument(
        "--fmriprep-dir",
        required=True,
        help="fMRIPrep derivatives directory.",
    )
    parser.add_argument(
        "--temp-dir",
        required=True,
        help="Directory for temporary files (one per echo).",
    )
    parser.add_argument(
        "--tedana-out-dir",
        required=True,
        help="Destination derivatives directory for tedana outputs.",
    )
    parser.add_argument(
        "--session-label",
        help="Optional session label (with or without 'ses-' prefix) to restrict processing.",
    )
    parser.add_argument(
        "--subject-label",
        help="Optional subject label (with or without 'sub-' prefix) to restrict processing.",
    )
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)

    run_tedana(
        raw_dir=args.raw_dir,
        fmriprep_dir=args.fmriprep_dir,
        temp_dir=args.temp_dir,
        tedana_out_dir=args.tedana_out_dir,
        session_label=args.session_label,
        subject_label=args.subject_label,
    )
