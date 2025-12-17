"""Plot the correlation matrices for the XCP-D outputs."""

from nilearn import plotting


if __name__ == "__main__":
    in_file = (
        "/cbica/projects/executive_function/mebold_trt/"
        "derivatives/fracback/group-all/group/"
        "model-onesample_contrast-twobackminuszeroback_stat-effect_statmap.nii.gz"
    )
    bg_img = (
        "/cbica/projects/executive_function/.cache/templateflow/"
        "tpl-MNI152NLin6Asym/tpl-MNI152NLin6Asym_res-02_T1w.nii.gz"
    )
    plotting.plot_stat_map(
        in_file,
        bg_img=bg_img,
        display_mode="mosaic",
        threshold=0.01,
        colorbar=True,
        symmetric_cbar=True,
        draw_cross=False,
        black_bg=False,
        output_file="../figures/nback_second_level_onesample.png",
    )
