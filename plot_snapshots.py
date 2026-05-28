#!/usr/bin/env python3

import pathlib
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from natsort import natsorted

from dedalus.extras.plot_tools import plot_bot

def classify_field(dset):
    """
    Returns: 'scalar', 'vector', or 'reduced'
    """

    shape = dset.shape

    if len(shape) == 4:
        return "vector"   # (t, comp, x, z)
    elif len(shape) == 3:
        return "scalar"   # (t, x, z)
    else:
        return "reduced"  # (t, x) or (t,)


def compute_global_bounds(dset, field_type):
    vmin, vmax = np.inf, -np.inf

    for i in range(dset.shape[0]):

        data = dset[i]

        if field_type == "vector":
            mag = np.sqrt(np.sum(data**2, axis=0))
            arr = mag
        else:
            arr = data

        vmin = min(vmin, np.nanmin(arr))
        vmax = max(vmax, np.nanmax(arr))

    return vmin, vmax


class FieldData:
    def __init__(self, name, dset):
        self.name = name
        self.dset = dset
        self.type = classify_field(dset)
        self.vmin, self.vmax = compute_global_bounds(dset, self.type)

    def get_frame(self, i, mode="mag"):
        data = self.dset[i]

        if self.type == "vector":
            if mode == "mag":
                return np.sqrt(np.sum(data**2, axis=0))
            elif mode == "x":
                return data[0]
            elif mode == "z":
                return data[1]
        else:
            return data


def get_all_writes(open_files, task_name):
    """
    Collect all writes from multiple Dedalus snapshot files.

    Returns:
        list of (dataset, write_index)
    """

    writes = []

    for file in open_files:

        if task_name not in file['tasks']:
            print(f"Task '{task_name}' not found in {file.filename}")
            continue

        dset = file['tasks'][task_name]

        for i in range(dset.shape[0]):
            writes.append((dset, i))

    return writes


def main():

    if len(sys.argv) < 2:
        print("Usage:")
        print("python3 plot_snapshots.py snapshots/*.h5")
        sys.exit(1)

    # Expand and sort filenames naturally
    h5_files = natsorted(sys.argv[1:])

    print("Files:")
    for f in h5_files:
        print(f"  {f}")

    open_files = [h5py.File(f, 'r') for f in h5_files]

    task_names = list(open_files[0]['tasks'].keys())

    fields = {}

    Lx, Lz = 2, 8
    Nx, Nz = 256, 64

    x = np.linspace(0, Lx, Nx)
    z = np.linspace(0, Lz, Nz)

    X, Z = np.meshgrid(x, z)

    stride = 4

    for name in task_names:
        fields[name] = {
            "files": [],
            "type": None,
            "vmin": None,
            "vmax": None
        }

        for f in open_files:
            if name in f["tasks"]:
                dset = f["tasks"][name]
                fields[name]["files"].append(dset)

        # Determine field type from first dataset
        sample_dset = fields[name]["files"][0]

        field_type = classify_field(sample_dset)

        fields[name]["type"] = field_type

        # Compute global bounds across ALL files
        vmin = np.inf
        vmax = -np.inf

        for dset in fields[name]["files"]:

            local_min, local_max = compute_global_bounds(dset, field_type)

            vmin = min(vmin, local_min)
            vmax = max(vmax, local_max)

        fields[name]["vmin"] = vmin
        fields[name]["vmax"] = vmax

    timeline = {}

    for name in task_names:
        frames = []

        for dset in fields[name]["files"]:
            for i in range(dset.shape[0]):
                frames.append((dset, i))

        timeline[name] = frames

    num_frames = len(timeline[task_names[0]])

    # Create subplot grid
    n_tasks = len(task_names)

    fig, axes = plt.subplots(
        n_tasks,
        1,
        figsize=(8, 4 * n_tasks),
        squeeze=False
    )

    axes = axes.flatten()

    def update(frame):
        
        print(f"\rRendering frame {frame+1}/{num_frames}", end="")

        fig.clf()
        axes = fig.subplots(n_tasks, 1).flatten()

        for i, name in enumerate(task_names):

            field = fields[name]
            ax = axes[i]

            dset, idx = timeline[name][frame]
            data = dset[idx]

            # VECTOR FIELD
            if field["type"] == "vector":

                ux = data[1]
                uz = data[0]

                speed = np.sqrt(ux**2 + uz**2)

                im = ax.imshow(
                    speed,
                    origin="lower",
                    aspect="auto",
                    extent=[0, Lx, 0, Lz],
                    cmap="viridis"
                )

                ax.quiver(
                    X[::stride, ::stride],
                    Z[::stride, ::stride],
                    ux[::stride, ::stride],
                    uz[::stride, ::stride],
                    color='white',
                    scale=25
                )

            # SCALAR FIELD
            else:

                arr = data

                im = ax.imshow(
                    arr,
                    origin="lower",
                    aspect="auto",
                    extent=[0, Lx, 0, Lz],
                    cmap="RdBu_r",
                    vmin=field["vmin"],
                    vmax=field["vmax"]
                )

            ax.set_title(f"{name} | frame {frame}")

            plt.colorbar(im, ax=ax, fraction=0.046)


        return axes

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=50,
        blit=False
    )

    writer = FFMpegWriter(fps=20, codec='libx264', extra_args=['-pix_fmt', 'yuv420p'])

    output_name = str(input("Enter name for file: "))

    print(f"\nSaving movie: {output_name}")

    anim.save(output_name, writer=writer)

    print("\nDone.")

    for file in open_files:
        file.close()


if __name__ == "__main__":
    main()