"""Plot the correlation matrices for the XCP-D outputs."""

from glob import glob

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __name__ == "__main__":
    dseg_file = (
        "/cbica/projects/executive_function/mebold-trt/derivatives/xcp_d/atlases/atlas-4S156Parcels/"
        "atlas-4S156Parcels_dseg.tsv"
    )
    dseg_df = pd.read_table(dseg_file)

    atlas_mapper = {
        "CIT168Subcortical": "Subcortical",
        "ThalamusHCP": "Thalamus",
        "SubcorticalHCP": "Subcortical",
    }
    network_labels = dseg_df["network_label"].fillna(dseg_df["atlas_name"]).tolist()
    network_labels = [atlas_mapper.get(network, network) for network in network_labels]

    # Determine order of nodes while retaining original order of networks
    unique_labels = []
    for label in network_labels:
        if label not in unique_labels:
            unique_labels.append(label)

    mapper = {label: f"{i:03d}_{label}" for i, label in enumerate(unique_labels)}
    mapped_network_labels = [mapper[label] for label in network_labels]
    community_order = np.argsort(mapped_network_labels)

    # Get the community name associated with each network
    labels = np.array(network_labels)[community_order]
    unique_labels = sorted(set(labels))
    unique_labels = []
    for label in labels:
        if label not in unique_labels:
            unique_labels.append(label)

    # Find the locations for the community-separating lines
    break_idx = [0]
    end_idx = None
    for label in unique_labels:
        start_idx = np.where(labels == label)[0][0]
        if end_idx:
            break_idx.append(np.nanmean([start_idx, end_idx]))

        end_idx = np.where(labels == label)[0][-1]

    break_idx.append(len(labels))
    break_idx = np.array(break_idx)

    # Find the locations for the labels in the middles of the communities
    label_idx = np.nanmean(np.vstack((break_idx[1:], break_idx[:-1])), axis=0)

    corrmats = sorted(
        glob(
            "/cbica/projects/executive_function/mebold-trt/derivatives/xcp_d/sub-*/ses-*/func/"
            "*seg-4S156Parcels_stat-pearsoncorrelation_relmat.tsv"
        )
    )
    for acq in ["MBME", "MBSE"]:
        selected_corrmats = [cm for cm in corrmats if f"acq-{acq}" in cm]
        arrs = []
        for cm in selected_corrmats:
            arrs.append(pd.read_table(cm, index_col="Node").to_numpy())
            arr_3d = np.dstack(arrs)
            arr_3d_z = np.arctanh(arr_3d)

            # First mean
            mean_arr_z = np.nanmean(arr_3d_z, axis=2)

            # Sort parcels by community
            mean_arr_z = mean_arr_z[community_order, :]
            mean_arr_z = mean_arr_z[:, community_order]
            np.fill_diagonal(mean_arr_z, 0)

            mean_arr_r = np.tanh(mean_arr_z)

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(mean_arr_r, cmap="seismic", vmin=-1, vmax=1)

            # Add lines separating networks
            for idx in break_idx[1:-1]:
                ax.axes.axvline(idx, color="black")
                ax.axes.axhline(idx, color="black")

            # Add network names
            ax.axes.set_yticks(label_idx)
            ax.axes.set_xticks(label_idx)
            ax.axes.set_yticklabels(unique_labels)
            ax.axes.set_xticklabels(unique_labels, rotation=90)
            fig.tight_layout()
            fig.savefig(f"../figures/XCPD_acq-{acq}_Mean.png")
            plt.close()

            # Now standard deviation
            sd_arr_z = np.nanstd(arr_3d_z, axis=2)

            # Sort parcels by community
            sd_arr_z = sd_arr_z[community_order, :]
            sd_arr_z = sd_arr_z[:, community_order]
            np.fill_diagonal(sd_arr_z, 0)

            sd_arr_r = np.tanh(sd_arr_z)
            # vmax1 = np.round(np.max(sd_arr_r), 2)
            # hardcoded based on previous checks
            vmax1 = 0.6

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(sd_arr_r, cmap="Reds", vmin=0, vmax=vmax1)

            # Add lines separating networks
            for idx in break_idx[1:-1]:
                ax.axes.axvline(idx, color="black")
                ax.axes.axhline(idx, color="black")

            # Add network names
            ax.axes.set_yticks(label_idx)
            ax.axes.set_xticks(label_idx)
            ax.axes.set_yticklabels(unique_labels)
            ax.axes.set_xticklabels(unique_labels, rotation=90)
            fig.tight_layout()
            fig.savefig(f"../figures/XCPD_acq-{acq}_StandardDeviation.png")
            plt.close()

            # Plot the colorbars
            fig, axs = plt.subplots(2, 1, figsize=(10, 1.5))

            norm = mpl.colors.Normalize(vmin=-1, vmax=1)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.seismic),
                cax=axs[0],
                orientation="horizontal",
            )
            cbar.set_ticks([-1, 0, 1])

            norm = mpl.colors.Normalize(vmin=0, vmax=vmax1)
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Reds),
                cax=axs[1],
                orientation="horizontal",
            )
            cbar.set_ticks([0, np.mean([0, vmax1]), vmax1])

            fig.tight_layout()
            fig.savefig(
                f"../figures/XCPD_acq-{acq}_colorbar.png",
                bbox_inches="tight",
            )
            plt.close()
