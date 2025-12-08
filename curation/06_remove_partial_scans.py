#!/cbica/home/salot/miniconda3/envs/salot/bin/python
"""Fix BIDS files after heudiconv conversion."""

import os
import shutil
from glob import glob

import nibabel as nb
import pandas as pd


def _crop_run_files(
    run_files,
    scans_file,
    n_noise_vols,
    n_uncropped_vols,
    n_cropped_vols,
):
    scans_df = pd.read_table(scans_file)

    for run_file in run_files:
        run_img = nb.load(run_file)
        n_vols = run_img.shape[3]
        if n_vols == n_uncropped_vols:
            # Split out last two volumes into noise scans
            noise_file = run_file.replace("_bold.nii.gz", "_noRF.nii.gz")
            if os.path.isfile(noise_file):
                print(f"File exists: {os.path.basename(noise_file)}")
                continue

            noise_img = run_img.slicer[..., -n_noise_vols:]
            run_img = run_img.slicer[..., :-n_noise_vols]

            # Overwrite the BOLD scan
            os.remove(run_file)
            run_img.to_filename(run_file)
            noise_img.to_filename(noise_file)

            # Copy the JSON as well
            shutil.copyfile(
                run_file.replace(".nii.gz", ".json"),
                noise_file.replace(".nii.gz", ".json"),
            )

            # Add noise scans to scans DataFrame
            i_row = len(scans_df.index)
            me_bold_fname = os.path.join("func", os.path.basename(run_file))
            noise_fname = os.path.join("func", os.path.basename(noise_file))
            scans_df.loc[i_row] = scans_df.loc[
                scans_df["filename"] == me_bold_fname
            ].iloc[0]
            scans_df.loc[i_row, "filename"] = noise_fname

        elif n_vols == n_cropped_vols:
            noise_file = run_file.replace("_bold.nii.gz", "_noRF.nii.gz")
            noise_img = nb.load(noise_file)
            if noise_img.shape[3] <= n_noise_vols:
                print(f"File already cropped: {os.path.basename(noise_file)}")
                continue

            noise_img = noise_img.slicer[..., :n_noise_vols]

            # Overwrite the noise scan
            os.remove(noise_file)
            noise_img.to_filename(noise_file)

        else:
            print(f"File has {n_vols} volumes: {os.path.basename(run_file)}")

    scans_df = scans_df.sort_values(by=["acq_time", "filename"])
    os.remove(scans_file)
    scans_df.to_csv(scans_file, sep="\t", na_rep="n/a", index=False)


def fix_sub_04_ses_1_task_fracback(dset_dir):
    """Fix sub-04_ses-1_task-fracback.

    sub-04_ses-1_task-fracback_acq-MBME_echo-5_part-phase_bold.nii.gz
    has 218 volumes instead of 219 like the rest of the run's files.
    This run doesn't have noise scans, so I think maybe the scan was ended
    early or failed to reconstruct, which would explain the missing volume.
    I'll just remove the last volume from each of the other files in the
    file collection.
    """
    func_dir = os.path.join(dset_dir, "sub-04/ses-1/func")
    run_files = sorted(
        glob(
            os.path.join(
                func_dir,
                "sub-04_ses-1_task-fracback_acq-MBME_*bold.nii.gz",
            )
        )
    )
    for run_file in run_files:
        run_img = nb.load(run_file)
        n_vols = run_img.shape[3]
        if n_vols == 219:
            run_img = run_img.slicer[:, :, :, :218]

            # Overwrite the BOLD scan
            os.remove(run_file)
            run_img.to_filename(run_file)

        elif n_vols != 218:
            print(f"File has {n_vols} volumes: {os.path.basename(run_file)}")


def fix_sub_04_ses_2_task_fracback(dset_dir):
    """Fix sub-04_ses-2_task-fracback.

    sub-04_ses-2_task-fracback_acq-MBME echoes 3-5 have 239 volumes,
    but the earlier echoes have 240.

    Something went wrong here. Not sure what. I had to reconvert these scans
    and re-fix them with a separate script.
    """
    scans_file = os.path.join(dset_dir, "sub-04/ses-2/sub-04_ses-2_scans.tsv")
    func_dir = os.path.join(dset_dir, "sub-04/ses-2/func")

    run_files = sorted(
        glob(
            os.path.join(
                func_dir,
                "sub-04_ses-2_task-fracback_acq-MBME_*bold.nii.gz",
            )
        )
    )
    _crop_run_files(
        run_files,
        scans_file,
        n_noise_vols=2,
        n_uncropped_vols=239,
        n_cropped_vols=237,
    )


def fix_sub_04_ses_2_task_rest_acq_multiecho_run_02(dset_dir):
    """Fix sub-04_ses-2_task-rest_acq-MBME_run-02.

    I think the later echoes just failed to reconstruct.
    sub-04_ses-2_task-rest_acq-MBME_run-02 echoes 3-5 have 203 volumes,
    but the earlier echoes have 204 volumes.
    I'll just drop the last volume from the other noise scans,
    and split off the two noise volumes from these files.
    """
    scans_file = os.path.join(dset_dir, "sub-04/ses-2/sub-04_ses-2_scans.tsv")
    func_dir = os.path.join(dset_dir, "sub-04/ses-2/func")

    run_files = sorted(
        glob(
            os.path.join(
                func_dir,
                "sub-04_ses-2_task-rest_acq-MBME_run-02_*bold.nii.gz",
            )
        )
    )
    _crop_run_files(
        run_files,
        scans_file,
        n_noise_vols=2,
        n_uncropped_vols=203,
        n_cropped_vols=201,
    )


if __name__ == "__main__":
    dset_dir = "/cbica/projects/executive_function/mebold_trt/dset/"

    # Crop files to shortest (218 vols), without noise scans.
    fix_sub_04_ses_1_task_fracback(dset_dir)
    # Split noise scans out of short files (239 vols) and shorten existing
    # noise scans.
    fix_sub_04_ses_2_task_fracback(dset_dir)
    # Split noise scans out of short files (203 vols) and shorten existing
    # noise scans.
    fix_sub_04_ses_2_task_rest_acq_multiecho_run_02(dset_dir)
