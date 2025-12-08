#!/usr/bin/env python
# coding: utf-8
"""Parse fractal n-back log files and convert them to BIDS format."""

import os
from glob import glob

import numpy as np
import pandas as pd

# NOTE: This ID mapping used to include the actual subject IDs, but they are PII,
# so we cannot share them here.
ID_MAPPER = {
    "01": "sub-01",
    "02": "sub-02",
    "03": "sub-03",
    "04": "sub-04",
    "05": "sub-05",
    "06": "sub-06",
    "07": "sub-07",
    "08": "sub-08",
}


def main(in_file):
    subject_session = os.path.basename(in_file).split("-")[0]
    subject, session = subject_session.split("_")
    subject_id = ID_MAPPER.get(subject, None)
    if not subject_id:
        print(f"Subject {subject}_{session} does not match anything.")
        return

    print(f"Processing {subject_id} ses-{session}...")
    out_file = os.path.join(
        subject_id,
        f"ses-{session}",
        "func",
        f"{subject_id}_ses-{session}_task-fracback_acq-MBME_events.tsv",
    )
    os.makedirs(os.path.dirname(out_file), exist_ok=True)

    with open(in_file, "r") as fo:
        orig_data = fo.read()

    trial_df = pd.read_table("stimuli_and_timing_ungrouped.tsv")

    stimuli_df = trial_df.loc[trial_df["event_type"] == "trial"]
    stimuli_df = stimuli_df.reset_index(drop=True)

    grouped_trial_df = pd.read_table("stimuli_and_timing_grouped.tsv")
    grouped_trial_df["overall"] = grouped_trial_df.index.values + 1

    data = orig_data.split("\n")
    header_idx = [i for i, r in enumerate(data) if r.startswith("Subject")][0]
    end_idx = [i for i, r in enumerate(data) if r == "{"][0]
    data = data[header_idx:end_idx]
    data = [r for r in data if r]
    data = [r.split("\t") for r in data]
    n_cols = len(data[0])
    new_data = []
    for row in data:
        n_missing = n_cols - len(row)
        new_row = row + ([None] * n_missing)
        new_data.append(new_row)

    df = pd.DataFrame(columns=new_data[0], data=new_data[1:])

    df = df.loc[df["Event Type"].isin(["Picture", "Response"])]

    subject_trial_df = df.loc[df["Code"] == "pic1"]
    subject_trial_df["Time"] = subject_trial_df["Time"].astype(float)
    subject_trial_df = subject_trial_df[["Trial", "Time"]]
    subject_trial_df = subject_trial_df.reset_index(drop=True)

    combined_trial_df = pd.concat((stimuli_df, subject_trial_df), axis=1)
    combined_trial_df = combined_trial_df.loc[
        combined_trial_df["stim_file"] != "stimuli/crosshair.jpg"
    ]
    combined_trial_df["Time"] = combined_trial_df["Time"].astype(float)
    combined_trial_df = combined_trial_df.reset_index(drop=True)
    combined_trial_df["trial"] = combined_trial_df.index.values + 1

    response_df = df.loc[df["Event Type"] == "Response"]
    response_df["Time At Response"] = response_df["Time"].astype(float)
    response_df = response_df[["Trial", "Time At Response", "TTime"]]

    combined_trial_df["response_time"] = np.nan
    trial_times = combined_trial_df["Time"].values
    for time_at_response in response_df["Time At Response"].values:
        trial_idx = np.where(time_at_response > trial_times)[0]
        if trial_idx.size:
            trial_idx = trial_idx[-1]
            combined_trial_df.loc[trial_idx, "response_time"] = (
                time_at_response - trial_times[trial_idx]
            )

    # Time seems to be in tenths of milliseconds (1 / 10000 second)
    combined_trial_df["response_time"] = combined_trial_df["response_time"] / 10000
    result = pd.merge(grouped_trial_df, combined_trial_df, on="trial", how="outer")

    result = result.sort_values(by="overall")
    result["trial_type"] = "trial"
    result.loc[result["stim_file_x"] == "stimuli/mask_Fix_xhair.jpg", "trial_type"] = (
        "fixation"
    )
    result.loc[result["stim_file_x"] == "stimuli/crosshair.jpg", "trial_type"] = (
        "fixation"
    )
    result.loc[result["stim_file_x"] == "stimuli/2back_img_0.jpg", "trial_type"] = (
        "instruction"
    )
    result.loc[result["stim_file_x"] == "stimuli/2back_img_2.jpg", "trial_type"] = (
        "instruction"
    )
    result["condition"] = None
    result.loc[result["stim_file_x"] == "stimuli/2back_img_0.jpg", "condition"] = (
        "0back"
    )
    result.loc[result["stim_file_x"] == "stimuli/2back_img_2.jpg", "condition"] = (
        "2back"
    )
    result["condition"] = result["condition"].ffill()
    result.loc[
        (result["trial_type"] == "trial") & (result["condition"] == "0back"),
        "trial_type",
    ] = "0back"
    result.loc[
        (result["trial_type"] == "trial") & (result["condition"] == "2back"),
        "trial_type",
    ] = "2back"
    result = result.reset_index(drop=True)
    onsets = result["duration_x"].cumsum().values
    onsets = np.hstack(([0], onsets))[:-1]
    result["onset"] = onsets

    result[["onset", "duration_x", "trial_type", "response_time", "stim_file_x"]]
    result["stim_file_x_2before"] = result["stim_file_x"].shift(4)

    result["correct"] = None
    result["classification"] = None

    resp_idx = result["response_time"] > 0
    noresp_idx = ~resp_idx

    # 0back
    stim_0back = "stimuli/fnb_formB_19.jpg"
    idx_0back = result["trial_type"] == "0back"
    idx_pos_0back = result["stim_file_x"] == stim_0back
    idx_neg_0back = result["stim_file_x"] != stim_0back

    # 0back true positives
    result.loc[idx_0back & resp_idx & idx_pos_0back, "correct"] = True
    result.loc[idx_0back & resp_idx & idx_pos_0back, "classification"] = "true positive"

    # 0back true negatives
    result.loc[idx_0back & noresp_idx & idx_neg_0back, "correct"] = True
    result.loc[idx_0back & noresp_idx & idx_neg_0back, "classification"] = (
        "true negative"
    )

    # 0back false negatives
    idx_2back = result["trial_type"] == "2back"
    result.loc[idx_0back & noresp_idx & idx_pos_0back, "correct"] = False
    result.loc[idx_0back & noresp_idx & idx_pos_0back, "classification"] = (
        "false negative"
    )

    # 0back false positives
    result.loc[idx_0back & resp_idx & idx_neg_0back, "correct"] = True
    result.loc[idx_0back & resp_idx & idx_neg_0back, "classification"] = (
        "false positive"
    )

    # 2back
    idx_2back = result["trial_type"] == "2back"
    idx_pos_2back = result["stim_file_x"] == result["stim_file_x_2before"]
    idx_neg_2back = result["stim_file_x"] != result["stim_file_x_2before"]

    # 2back true positives
    result.loc[idx_2back & resp_idx & idx_pos_2back, "correct"] = True
    result.loc[idx_2back & resp_idx & idx_pos_2back, "classification"] = "true positive"

    # 2back true negatives
    result.loc[idx_2back & noresp_idx & idx_neg_2back, "correct"] = True
    result.loc[idx_2back & noresp_idx & idx_neg_2back, "classification"] = (
        "true negative"
    )

    # 2back false negatives
    result.loc[idx_2back & noresp_idx & idx_pos_2back, "correct"] = False
    result.loc[idx_2back & noresp_idx & idx_pos_2back, "classification"] = (
        "false negative"
    )

    # 2back false positives
    result.loc[idx_2back & resp_idx & idx_neg_2back, "correct"] = True
    result.loc[idx_2back & resp_idx & idx_neg_2back, "classification"] = (
        "false positive"
    )

    result2 = result[
        [
            "onset",
            "duration_x",
            "trial_type",
            "classification",
            "response_time",
            "stim_file_x",
            "trial",
        ]
    ]
    result2 = result2.rename(
        columns={
            "duration_x": "duration",
            "stim_file_x": "stimulus",
            "trial": "trial_number",
        }
    )
    result2.to_csv(out_file, sep="\t", index=False, na_rep="n/a")


if __name__ == "__main__":
    log_files = sorted(
        glob(
            "/cbica/projects/executive_function/mebold_trt/sourcedata/task_log_files/*.log",
        ),
    )
    for log_file in log_files:
        main(log_file)
