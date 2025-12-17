"""Plot the correlation matrices for the XCP-D outputs."""

import nibabel as nb
import numpy as np
from nilearn import plotting
from nilearn.image import get_data
from nilearn.image.resampling import reorder_img
from scipy.ndimage import binary_fill_holes


if __name__ == "__main__":
    in_file = (
        "/cbica/projects/executive_function/mebold_trt/"
        "derivatives/fracback/group-all/group/"
        "model-onesample_contrast-twobackminuszeroback_stat-z_statmap.nii.gz"
    )
    bg_img = (
        "/cbica/projects/executive_function/.cache/templateflow/"
        "tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_desc-brain_T1w.nii.gz"
    )
    bg_img = nb.load(bg_img)
    bg_img = reorder_img(bg_img)
    data = get_data(bg_img)
    data = data.astype(np.float64)
    anat_mask = binary_fill_holes(data > np.finfo(float).eps)
    data = np.ma.masked_array(data, np.logical_not(anat_mask))
    bg_img = nb.Nifti1Image(data, bg_img.affine, bg_img.header)
    plotting.plot_stat_map(
        in_file,
        bg_img=bg_img,
        display_mode="mosaic",
        threshold=1.96,
        colorbar=True,
        symmetric_cbar=True,
        draw_cross=False,
        black_bg=False,
        output_file="../figures/nback_second_level_onesample.png",
    )
