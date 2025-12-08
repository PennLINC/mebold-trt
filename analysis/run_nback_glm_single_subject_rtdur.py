import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from nilearn.glm.first_level import first_level_from_bids
from nilearn.interfaces.bids import save_glm_to_bids


# ============================================================
# GET SUBJECT FROM COMMAND LINE
# ============================================================
sub_id = sys.argv[1]  # subject passed by SLURM array / CLI
sub_labels = [sub_id]  # nilearn expects a list

print(f"Running first-level GLM for subject: {sub_id}")

# ---------- CONFIG ----------
# BIDS root
bids_root = Path("/cbica/projects/executive_function/mebold-trt/dset")

# fMRIPrep derivatives containing preprocessed func + confounds
derivatives_folder = "/cbica/projects/executive_function/mebold-trt/derivatives/fmriprep"
deriv_root = Path(derivatives_folder)

# Task and space
task_label = "fracback"
space_label = "MNI152NLin6Asym"

# Output directory for maps + reports
out_dir = Path(
    "/cbica/projects/executive_function/mebold-trt/derivatives/fracback"
)
out_dir.mkdir(parents=True, exist_ok=True)

# Smoothing to apply in Nilearn
smoothing_fwhm = 5.0

# ============================================================
# FIND FMRIPREP BRAIN MASK FOR THIS SUBJECT
# ============================================================
# Example expected filename:
# sub-01_ses-1_task-fracback_acq-MBME_run-01_space-MNI152NLin6Asym_res-2_desc-brain_mask.nii.gz
mask_pattern = (
    f"sub-{sub_id}/ses-*/func/"
    f"sub-{sub_id}_ses-*_task-{task_label}_acq-MBME"
    f"_space-{space_label}_res-2_desc-brain_mask.nii.gz"
)
mask_candidates = sorted(deriv_root.glob(mask_pattern))

mask_img = None
if len(mask_candidates) == 0:
    print(
        f"[WARNING] No fMRIPrep brain mask found for sub-{sub_id} with pattern:\n"
        f"  {mask_pattern}\n"
        "Falling back to Nilearn's automatic mask.\n"
    )
else:
    # Since only one run per subject, take the first match
    mask_img = str(mask_candidates[0])
    print(f"Using fMRIPrep brain mask for sub-{sub_id}:\n  {mask_img}\n")

# ============================================================
# BUILD FIRST-LEVEL MODEL FROM BIDS
# ============================================================
(
    models,
    models_run_imgs,
    models_events,
    models_confounds,
) = first_level_from_bids(
    dataset_path=bids_root,
    task_label=task_label,
    space_label=space_label,
    sub_labels=sub_labels,
    derivatives_folder=derivatives_folder,
    img_filters=[("desc", "preproc")],
    smoothing_fwhm=smoothing_fwhm,
    n_jobs=4,
    verbose=1,
    # Use fMRIPrep's cosine regressors for drift (high_pass),
    # so we turn OFF the GLM's own drift model.
    drift_model=None,
    # Confound strategy: motion + high_pass (cosines) + aCompCor (first 5)
    confounds_strategy=("motion", "high_pass", "compcor"),
    confounds_motion="basic",
    confounds_compcor="anat_combined",
    confounds_n_compcor=5,
    # Use fMRIPrep mask if found; otherwise None (Nilearn auto-mask)
    mask_img=mask_img,
)

# Single subject
model = models[0]
run_imgs = models_run_imgs[0]
events_list = models_events[0]
confounds_list = models_confounds[0]

# Implement Jeanette Mumford's ConsDurRTDur model
# and rename trial types to valid Python identifiers:
#   0BACK -> zero_back
#   2BACK -> two_back
cons_dur_rt_dur_events_list = []
for events in events_list:
    events = events.copy()

    # Rename trial_type labels before building design matrix
    if "trial_type" in events.columns:
        events["trial_type"] = events["trial_type"].replace(
            {
                "0BACK": "zero_back",
                "2BACK": "two_back",
            }
        )

    # Create dataframe only containing trials with responses
    response_events = events.loc[~np.isnan(events["response_time"])].copy()
    # Set duration to response time
    response_events.loc[:, "duration"] = response_events.loc[:, "response_time"]
    # Change trial type to a single value (RTDur)
    response_events.loc[:, "trial_type"] = "RTDur"
    # Add new "condition" back into events dataframe
    events = pd.concat((events, response_events))
    events = events.sort_values(by="onset")
    cons_dur_rt_dur_events_list.append(events)

print(f"Number of runs for subject {sub_id}: {len(run_imgs)}")
print("Confounds entries for each run:")
for i, c in enumerate(confounds_list):
    print(f"  Run {i}: {c}")

# ---------- FIT MODEL ----------
model.minimize_memory = False

print(f"\nFitting GLM for subject {sub_id}...")
model = model.fit(
    run_imgs,
    events=cons_dur_rt_dur_events_list,
    confounds=confounds_list,
)

# Inspect design matrix
design_matrix = model.design_matrices_[0]
print("\nDesign matrix columns:")
print(design_matrix.columns)
print(f"\nTotal # regressors in design matrix: {design_matrix.shape[1]}")

save_glm_to_bids(
    model,
    contrasts="two_back - zero_back",
    contrast_types={"two_back - zero_back": "t"},
    out_dir=out_dir,
    threshold=norm.isf(0.001),
    cluster_threshold=10,
    bg_img=(
        "/cbica/projects/executive_function/.cache/templateflow/"
        "tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"
    ),
)

print("\nDone.\n")

