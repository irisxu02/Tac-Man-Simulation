import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_contact_deviations(
    marker_data, save_path=None, view="xz", magnification=10.0
):
    if len(marker_data) == 0:
        print("No data found.")
        return

    # Find all sequences where state is PROC, RECV, PROC
    sequences = []
    for i in range(1, len(marker_data) - 1):
        if (
            marker_data[i - 1]["state"] == "PROC"
            and marker_data[i]["state"] == "RECV"
            and marker_data[i + 1]["state"] == "PROC"
        ):
            sequences.append((i - 1, i, i + 1))

    if not sequences:
        print("No sequences found.")
        return

    axes_dict = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    labels = {
        "xy": ("X (m)", "Y (m)"),
        "xz": ("X (m)", "Z (m)"),
        "yz": ("Y (m)", "Z (m)"),
    }
    ax_ix = axes_dict[view]

    # Save a separate plot for the INIT state
    for idx, frame in enumerate(marker_data):
        if frame.get("state", None) == "INIT":
            fig, ax = plt.subplots(figsize=(6, 6))
            init_pos = frame["init_pos"].numpy()[:, :3]
            curr_pos = frame["curr_pos"].numpy()[:, :3]
            disp = curr_pos - init_pos
            U = disp[:, ax_ix[0]]
            V = disp[:, ax_ix[1]]
            if "unlocked_pos" in frame:
                unlocked = frame["unlocked_pos"].numpy()[:, :3]
            else:
                unlocked = np.zeros((0, 3))
            if len(unlocked) > 0:
                ax.scatter(
                    unlocked[:, ax_ix[0]],
                    unlocked[:, ax_ix[1]],
                    c="lightgray",
                    s=15,
                    alpha=0.3,
                    zorder=1,
                )
            ax.scatter(
                init_pos[:, ax_ix[0]],
                init_pos[:, ax_ix[1]],
                c="black",
                s=20,
                alpha=0.5,
                zorder=2,
            )
            q = ax.quiver(
                init_pos[:, ax_ix[0]],
                init_pos[:, ax_ix[1]],
                U,
                V,
                color="tab:blue",
                angles="xy",
                scale_units="xy",
                scale=1 / (magnification),
                width=0.003,
                headwidth=4,
                headlength=5,
                zorder=3,
            )
            ref_len = 0.001
            ax.set_aspect("equal")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if save_path:
                base, ext = save_path.rsplit(".", 1)
                out_path = f"{base}_init_step_{idx}.{ext}"
                plt.savefig(out_path, dpi=150)
                print(f"Saved INIT state to {out_path}")
            break

    # Sort sequences by the step index of the RECV frame (middle)
    sequences.sort(key=lambda t: t[1])

    for triple_num, (idx_proc1, idx_recv, idx_proc2) in enumerate(sequences):
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        for j, idx in enumerate([idx_proc1, idx_recv, idx_proc2]):
            frame = marker_data[idx]
            init_pos = frame["init_pos"].numpy()[:, :3]
            curr_pos = frame["curr_pos"].numpy()[:, :3]
            disp = curr_pos - init_pos
            U = disp[:, ax_ix[0]]
            V = disp[:, ax_ix[1]]
            if "unlocked_pos" in frame:
                unlocked = frame["unlocked_pos"].numpy()[:, :3]
            else:
                unlocked = np.zeros((0, 3))
            ax = axs[j]
            if len(unlocked) > 0:
                ax.scatter(
                    unlocked[:, ax_ix[0]],
                    unlocked[:, ax_ix[1]],
                    c="lightgray",
                    s=15,
                    alpha=0.3,
                    zorder=1,
                )
            ax.scatter(
                init_pos[:, ax_ix[0]],
                init_pos[:, ax_ix[1]],
                c="black",
                s=20,
                alpha=0.5,
                zorder=2,
            )
            q = ax.quiver(
                init_pos[:, ax_ix[0]],
                init_pos[:, ax_ix[1]],
                U,
                V,
                color="tab:blue",
                angles="xy",
                scale_units="xy",
                scale=1 / (magnification),
                width=0.003,
                headwidth=4,
                headlength=5,
                zorder=3,
            )
            ref_len = 0.001
            # Only show the scale unit on the first subplot
            if j == 0:
                ax.quiverkey(
                    q,
                    0.9,
                    1.02,
                    ref_len,
                    f"{ref_len*1000:.0f} mm scale",
                    labelpos="E",
                    coordinates="axes",
                    fontproperties={"weight": "bold"},
                )
            ax.set_aspect("equal")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        if save_path:
            base, ext = save_path.rsplit(".", 1)
            out_path = f"{base}_triple_{triple_num}_step_{idx_recv}.{ext}"
            plt.savefig(out_path, dpi=150)
            print(f"Saved to {out_path}")


if __name__ == "__main__":
    import argparse

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--path", type=str, required=True)
    argument_parser.add_argument("--mag", type=float, default=5.0)
    args = argument_parser.parse_args()
    data = torch.load(args.path)
    visualize_contact_deviations(
        data, save_path="deviations.png", view="xz", magnification=args.mag
    )
