
"""
Read and display ply files. Manipulate pointcloud data and store the result.
"""
# %%
import json

import numpy as np
import open3d as o3d

from highway.pointcloud import read_ply, to_pcd, write_ply
from pathlib import Path

# %%
def filename_to_pcd(input_file):
    # read ply file
    points, colors = read_ply(input_file)

    # subtract center as GPU cannot work on coordinates that are too large
    with open('highway_data/splitted/center.txt') as f:
        center = np.array(json.load(f))
    points_centered = points - center

    # display (rendered on GPU)
    #  * rotate by holding left mouse button
    #  * move by pressing middle mouse button OR STRG + left mouse button
    #  * zoom in and out with the scroll wheel
    pcd_centered = to_pcd(points_centered, colors)

    # save pointcloud as ply for viewing in Meshlab
    write_ply(f"example_1_{Path(input_file).stem}.ply", points_centered, colors)

    return pcd_centered

# %%

def main():
    # define ply path - please change accordingly
    input_files = [
        "highway_data/splitted/reduced_22_intensity.ply",
    ]

    pcds = []

    for input_file in input_files:
        pcds.append(filename_to_pcd(input_file))

    o3d.visualization.draw_geometries(pcds)

if __name__ == '__main__':
    main()


# %%
