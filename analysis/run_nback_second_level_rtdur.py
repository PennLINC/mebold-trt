#!/usr/bin/env python

from pathlib import Path
import pandas as pd
from scipy.stats import norm

from nilearn.glm.second_level import SecondLevelModel
from nilearn.interfaces.bids import save_glm_to_bids
from nilearn.image import load_img


# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
task_label = "fracback"
space_label = "MNI152NLin6Asym"
contrast_label = "twoBackMinusZeroBack"

# First-level output root
firstlevel_dir = Path("/cbica/projects/executive_function/mebold-trt/derivatives/fracback")

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
pattern = (
    f"sub-*/sub-*_task-{task_label}_space-{space_label}_"
    f"contrast-{contrast_label}_stat-effect_statmap.nii.gz"
)

effect_maps = sorted(firstlevel_dir.glob(pattern))

if len(effect_maps) == 0:
    raise RuntimeError(f"No first-level maps found with pattern:\n  {pattern}")

print(f"Found {len(effect_maps)} first-level effect-size maps:\n")
for p in effect_maps:
    print("  ", p)

subject_labels = [p.name.split("_")[0].replace("sub-", "") for p in effect_maps]

# ----------------------------------------------------------
# DESIGN MATRIX: ONE-SAMPLE T-TEST
# ----------------------------------------------------------
design_matrix = pd.DataFrame(
    {"intercept": [1.0] * len(effect_maps)},
    index=subject_labels,
)

print("\nSecond-level design matrix:")
print(design_matrix.head())
print(f"\nTotal subjects: {len(design_matrix)}\n")
print("Subjects included:")
print(sorted(subject_labels))

# ----------------------------------------------------------
# FIT SECOND-LEVEL MODEL
# ----------------------------------------------------------
model = SecondLevelModel(
    mask_img=group_mask_img,   # <-- TemplateFlow mask for analysis
    minimize_memory=False
)
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
    two_sided=True,             # <-- forward to generate_report(two_sided=True)
)

print(f"\nSaved second-level BIDS-like outputs to:\n  {group_out_dir}\n")
