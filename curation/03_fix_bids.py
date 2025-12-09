#!/cbica/home/salot/miniconda3/envs/salot/bin/python
"""Fix BIDS files after heudiconv conversion.

This script should deal with steps 1-6 below.

The necessary steps are:

1.  Deal with duplicates.
2.  Rename multi-echo magnitude BOLD files to part-mag_bold.
3.  Rename phase files to part-phase_bold.
4.  Split out noRF noise scans from multi-echo BOLD scans.
    -   Also copy the JSON.
5.  Copy first echo of each multi-echo field map without echo entity.
6.  Update filenames in the scans.tsv files.
7.  Remove events files.

Here is what the dupe situation looks like.

1.  sub-03_ses-1_task-rest_acq-multiecho_run-02_echo-*_bold|phase|sbref.nii.gz.
    It looks like this scan ends early (82 volumes).
    Need to rename the dupes to run-03.
    -   This script didn't do this correctly the first time around.
2.  sub-03_ses-1_task-frackack_acq-multiecho_bold__echo-1_dup-01.nii.gz.
    The *dupe* is the shorter one here (17 volumes).
    I'm guessing it was started by accident.
    Just remove the frackack dupes.
3.  sub-05_ses-2_task-frackack_acq-multiecho_sbref__echo-1_part-mag_dup-01.json.
    The dupe is the shorter one here as well.
    Just remove all dupes in func dir.
4.  sub-07_ses-2_T1w__dup-01.json
    This subject/session has a whole bunch of duplicate T1ws.
    They're all the same AFAICT, so just remove the dupes.
5.  sub-06_ses-1_echo-1_MEGRE.nii.gz.
    It looks like there are 3 sets of dups.
    I think the orig and even numbered one are phase volumes associated with
    the scans.
    The first set has 6 echoes. The second has 3.
    The scan numbers aren't in order, so I would need to group the images by
    image type and scan number.
    I don't know if I need to retain these scans.
    Ted may not want to share them at all.
    I'm just going to add these to the bidsignore for now.
"""

import os
import shutil
from glob import glob

import nibabel as nb
import pandas as pd

N_NOISE_VOLS = 3


if __name__ == "__main__":
    dset_dir = "/cbica/projects/executive_function/mebold_trt/dset/"
    subject_dirs = sorted(glob(os.path.join(dset_dir, "sub-*")))
    for subject_dir in subject_dirs:
        sub_id = os.path.basename(subject_dir)
        session_dirs = sorted(glob(os.path.join(subject_dir, "ses-*")))
        for session_dir in session_dirs:
            ses_id = os.path.basename(session_dir)
            anat_dir = os.path.join(session_dir, "anat")
            fmap_dir = os.path.join(session_dir, "fmap")
            func_dir = os.path.join(session_dir, "func")

            # Remove events files
            events_files = sorted(glob(os.path.join(func_dir, "*_events.tsv")))
            for events_file in events_files:
                os.remove(events_file)

            # Load scans file
            scans_file = os.path.join(session_dir, f"{sub_id}_{ses_id}_scans.tsv")
            assert os.path.isfile(scans_file), f"Scans file DNE: {scans_file}"
            scans_df = pd.read_table(scans_file)

            if "sub-01/ses-1" in session_dir:
                # Rename "frackack" scans to "rest" since no task was performed.
                frackacks = sorted(
                    glob(os.path.join(func_dir, "*task-frackack_acq-multiecho*"))
                )
                for frackack in frackacks:
                    frackack_fname = os.path.join("func", os.path.basename(frackack))
                    out_file = frackack.replace("acq-multiecho_", "acq-multiecho_run-03_")
                    out_fname = os.path.join("func", os.path.basename(out_file))
                    os.rename(frackack, out_file)
                    scans_df = scans_df.replace({"filename": {frackack_fname: out_fname}})

            if "sub-03/ses-1" in session_dir:
                # Remove duplicate frackack functional scans.
                dupe_frackacks = sorted(glob(os.path.join(func_dir, "*task-frackack*_dup-*")))
                for dupe_frackack in dupe_frackacks:
                    dupe_frackack_fname = os.path.join("func", os.path.basename(dupe_frackack))
                    print(f"Removing {dupe_frackack_fname}")
                    os.remove(dupe_frackack)
                    scans_df = scans_df.loc[scans_df["filename"] != dupe_frackack_fname]
                    scans_df = scans_df.reset_index(drop=True)

                # Rename rest dupes to run-03.
                run2_rests = sorted(
                    glob(os.path.join(func_dir, "*task-rest_acq-multiecho_run-02*_dup-*"))
                )
                for run2_rest in run2_rests:
                    run2_rest_fname = os.path.join("func", os.path.basename(run2_rest))
                    echo = run2_rest_fname.split("echo-")[1].split("_")[0]
                    prerun = run2_rest.split("_run-02_")[0]
                    postrun = run2_rest.split("_run-02_")[1].split("__")[0]
                    if run2_rest_fname.endswith(".nii.gz"):
                        ext = ".nii.gz"
                    else:
                        ext = ".json"

                    out_file = f"{prerun}_run-03_echo-{echo}_{postrun}{ext}"
                    out_fname = os.path.join("func", os.path.basename(out_file))
                    os.rename(run2_rest, out_file)
                    scans_df = scans_df.replace({"filename": {run2_rest_fname: out_fname}})

            if "sub-05/ses-2" in session_dir:
                # Delete duplicate functional scans.
                dupes = sorted(glob(os.path.join(func_dir, "*_dup-*")))
                for dupe in dupes:
                    dupe_fname = os.path.join("func", os.path.basename(dupe))
                    print(f"Removing {dupe_fname}")
                    scans_df = scans_df.loc[scans_df["filename"] != dupe_fname]
                    scans_df = scans_df.reset_index(drop=True)
                    os.remove(dupe)

            if "sub-07/ses-2" in session_dir:
                # Delete duplicate T1w scans.
                dupes = sorted(glob(os.path.join(anat_dir, "*T1w*_dup-*")))
                for dupe in dupes:
                    dupe_fname = os.path.join("anat", os.path.basename(dupe))
                    print(f"Removing {dupe_fname}")
                    scans_df = scans_df.loc[scans_df["filename"] != dupe_fname]
                    scans_df = scans_df.reset_index(drop=True)
                    os.remove(dupe)

            # Rename magnitude files from _bold to _part-mag_bold.
            mag_files = sorted(glob(os.path.join(func_dir, "*echo-*_bold.*")))
            for mag_file in mag_files:
                if "part-" in mag_file:
                    print(f"Skipping {mag_file}")
                    continue

                new_mag_file = mag_file.replace("_bold.", "_part-mag_bold.")
                os.rename(mag_file, new_mag_file)

                mag_filename = os.path.join("func", os.path.basename(mag_file))
                new_mag_filename = os.path.join("func", os.path.basename(new_mag_file))

                # Replace the filename in the scans.tsv file
                scans_df = scans_df.replace({"filename": {mag_filename: new_mag_filename}})

            # Rename phase files from _phase to _part-phase_bold.
            phase_files = sorted(glob(os.path.join(func_dir, "*_phase.*")))
            for phase_file in phase_files:
                new_phase_file = phase_file.replace("_phase.", "_part-phase_bold.")
                os.rename(phase_file, new_phase_file)

                phase_filename = os.path.join("func", os.path.basename(phase_file))
                new_phase_filename = os.path.join("func", os.path.basename(new_phase_file))

                # Replace the filename in the scans.tsv file
                scans_df = scans_df.replace({"filename": {phase_filename: new_phase_filename}})

            # Rename acq-multiecho files to acq-MBME
            me_files = sorted(glob(os.path.join(func_dir, "*acq-multiecho*")))
            for me_file in me_files:
                new_me_file = me_file.replace("acq-multiecho", "acq-MBME")
                os.rename(me_file, new_me_file)

                me_filename = os.path.join("func", os.path.basename(me_file))
                new_me_filename = os.path.join("func", os.path.basename(new_me_file))

                # Replace the filename in the scans.tsv file
                scans_df = scans_df.replace({"filename": {me_filename: new_me_filename}})

            # Rename acq-singleecho files to acq-MBSE
            se_files = sorted(glob(os.path.join(func_dir, "*acq-singleecho*")))
            for se_file in se_files:
                new_se_file = se_file.replace("acq-singleecho", "acq-MBSE")
                os.rename(se_file, new_se_file)

                se_filename = os.path.join("func", os.path.basename(se_file))
                new_se_filename = os.path.join("func", os.path.basename(new_se_file))

                # Replace the filename in the scans.tsv file
                scans_df = scans_df.replace({"filename": {se_filename: new_se_filename}})

            # Rename task-frackack files to task-fracback
            fracback_files = sorted(glob(os.path.join(func_dir, "*task-frackack*")))
            for fracback_file in fracback_files:
                new_fracback_file = fracback_file.replace("task-frackack", "task-fracback")
                os.rename(fracback_file, new_fracback_file)

                fracback_filename = os.path.join("func", os.path.basename(fracback_file))
                new_fracback_filename = os.path.join("func", os.path.basename(new_fracback_file))

                # Replace the filename in the scans.tsv file
                scans_df = scans_df.replace(
                    {"filename": {fracback_filename: new_fracback_filename}},
                )

            # Split out noise scans from all multi-echo BOLD files.
            me_bolds = sorted(glob(os.path.join(func_dir, "*acq-MBME*_bold.nii.gz")))
            for me_bold in me_bolds:
                noise_scan = me_bold.replace("_bold.nii.gz", "_noRF.nii.gz")
                if os.path.isfile(noise_scan):
                    print(f"File exists: {os.path.basename(noise_scan)}")
                    continue

                img = nb.load(me_bold)
                n_vols = img.shape[-1]
                if n_vols not in (240, 204, 200):
                    print(f"File is a partial scan: {os.path.basename(me_bold)}")
                    continue

                noise_img = img.slicer[..., -N_NOISE_VOLS:]
                bold_img = img.slicer[..., :-N_NOISE_VOLS]

                # Overwrite the BOLD scan
                os.remove(me_bold)
                bold_img.to_filename(me_bold)
                noise_img.to_filename(noise_scan)

                # Copy the JSON as well
                shutil.copyfile(
                    me_bold.replace(".nii.gz", ".json"),
                    noise_scan.replace(".nii.gz", ".json"),
                )

                # Add noise scans to scans DataFrame
                i_row = len(scans_df.index)
                me_bold_fname = os.path.join("func", os.path.basename(me_bold))
                noise_fname = os.path.join("func", os.path.basename(noise_scan))
                scans_df.loc[i_row] = scans_df.loc[scans_df["filename"] == me_bold_fname].iloc[0]
                scans_df.loc[i_row, "filename"] = noise_fname

            # Copy first echo's sbref of multi-echo field maps without echo entity.
            me_fmaps = sorted(glob(os.path.join(fmap_dir, "*_acq-ME*_echo-1_sbref.*")))
            for me_fmap in me_fmaps:
                out_fmap = me_fmap.replace("_echo-1_", "_").replace("_sbref", "_epi")
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
