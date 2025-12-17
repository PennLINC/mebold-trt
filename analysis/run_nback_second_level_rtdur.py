#!/usr/bin/env python
from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from scipy.stats import norm
from nilearn.glm.second_level import SecondLevelModel
from nilearn.interfaces.bids import save_glm_to_bids
from nilearn.image import load_img


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
# First-level output root
firstlevel_dir = Path("/cbica/projects/executive_function/mebold_trt/derivatives/fracback")

# Where to write second-level outputs
group_out_dir = firstlevel_dir / "group-all"
group_out_dir.mkdir(exist_ok=True)

# ----------------------------------------------------------
# TEMPLATEFLOW MASK + BACKGROUND (same bg as first level)
# ----------------------------------------------------------
# Brain mask for analysis (binary mask)
group_mask_img = load_img(
    "/cbica/projects/executive_function/.cache/templateflow/tpl-MNI152NLin6Asym/"
    "tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz"
)

# Background image for report visualization (underlay)
bg_img = load_img(
    "/cbica/projects/executive_function/.cache/templateflow/tpl-MNI152NLin6Asym/"
    "tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz"
)

# ----------------------------------------------------------
# COLLECT FIRST-LEVEL EFFECT SIZE MAPS
# ----------------------------------------------------------
prepost_dm = []
effect_maps = []

pattern = (
    "{sub_id}_{ses_id}_task-fracback_acq-MBME_"
    "contrast-twoBackMinusZeroBack_stat-effect_statmap.nii.gz"
)

subject_list = []
subject_dirs = sorted(firstlevel_dir.glob("sub-*"))
for subject_dir in subject_dirs:
    sub_id = subject_dir.name
    ses_ids = ["ses-1", "ses-2"]
    sessions_found = True
    for ses_id in ses_ids:
        effect_map = subject_dir / ses_id / "func" / pattern.format(sub_id=sub_id, ses_id=ses_id)
        if not effect_map.exists():
            print(
                f"No first-level maps found for subject: {sub_id} and session: {ses_id}\n"
                f"\t{effect_map}"
            )
            sessions_found = False
            continue

    if sessions_found:
        subject_list.append(sub_id)

map_labels = []
design_matrix_labels = ["ses1", "ses2"] + [s.replace("-", "") for s in subject_list]
ses_ids = ["ses-1", "ses-2"]
for ses_id in ses_ids:
    subject_effect = np.eye(len(subject_list))
    for i_subject, sub_id in enumerate(subject_list):
        map_labels.append(f"{sub_id}_{ses_id}")
        effect_map = firstlevel_dir / sub_id / ses_id / "func" / pattern.format(sub_id=sub_id, ses_id=ses_id)
        effect_maps.append(effect_map)
        if ses_id == "ses-1":
            prepost_dm.append([1, 0] + list(subject_effect[i_subject, :]))
        else:
            prepost_dm.append([0, 1] + list(subject_effect[i_subject, :]))

print(f"Found {len(effect_maps)} first-level effect-size maps:\n")
for p in effect_maps:
    print("  ", p)

# ----------------------------------------------------------
# ANALYSIS 1: ONE-SAMPLE T-TEST
# ----------------------------------------------------------
design_matrix = pd.DataFrame(
    {"intercept": [1.0] * len(effect_maps)},
    index=map_labels,
)

print("\nSecond-level design matrix:")
print(design_matrix.head())
print(f"\nTotal subjects: {len(design_matrix)}\n")
print("Subjects included:")
print(sorted(map_labels))

# ----------------------------------------------------------
# FIT SECOND-LEVEL MODEL
# ----------------------------------------------------------
model = SecondLevelModel(
    mask_img=group_mask_img,
    minimize_memory=False
)
effect_maps = [nb.load(effect_map) for effect_map in effect_maps]
model = model.fit(
    second_level_input=effect_maps,
    design_matrix=design_matrix,
)

# ----------------------------------------------------------
# SAVE OUTPUTS IN BIDS-LIKE FORMAT
# ----------------------------------------------------------
group_contrast_name = "twoBackMinusZeroBack"
contrasts = {group_contrast_name: "intercept"}

save_glm_to_bids(
    model=model,
    contrasts=contrasts,
    out_dir=group_out_dir,
    threshold=norm.isf(0.001),  # Z ~ 3.09
    height_control=None,        # treat threshold as Z, not p
    cluster_threshold=10,
    bg_img=bg_img,              # <-- same bg as first level
)

print(f"\nSaved second-level BIDS-like outputs to:\n  {group_out_dir}\n")

# ----------------------------------------------------------
# ANALYSIS 2: PAIRED T-TEST
# ----------------------------------------------------------
design_matrix = pd.DataFrame(
    columns=design_matrix_labels,
    data=prepost_dm,
    index=map_labels,
)

print("\nSecond-level design matrix:")
print(design_matrix.head())
print(f"\nTotal subjects: {len(design_matrix)}\n")
print("Subjects included:")
print(sorted(map_labels))

# ----------------------------------------------------------
# FIT SECOND-LEVEL MODEL
# ----------------------------------------------------------
model = SecondLevelModel(
    mask_img=group_mask_img,
    minimize_memory=False
)
model = model.fit(
    second_level_input=effect_maps,
    design_matrix=design_matrix,
)

# ----------------------------------------------------------
# SAVE OUTPUTS IN BIDS-LIKE FORMAT
# ----------------------------------------------------------
group_contrast_name = "ses1MinusSes2"
contrasts = {group_contrast_name: "ses1 - ses2"}

save_glm_to_bids(
    model=model,
    contrasts=contrasts,
    out_dir=group_out_dir,
    threshold=norm.isf(0.001),  # Z ~ 3.09
    height_control=None,        # treat threshold as Z, not p
    cluster_threshold=10,
    bg_img=bg_img,              # <-- same bg as first level
)

print(f"\nSaved second-level BIDS-like outputs to:\n  {group_out_dir}\n")
