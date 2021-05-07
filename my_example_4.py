
"""
Create 2d projections of the pointclouds and render them.
"""

# %%
from highway.camera import load_poses, show_image
from highway.pointcloud import read_ply, render_points
from highway.helpers import match_camera_pose_to_lidar_section, match_lidar_section_to_camera_poses
import pandas as pd
import numpy as np
from filecache import filecache
import json
from pathlib import Path
import os
from skimage.io import *
import open3d as o3d

from highway.pointcloud import read_ply, to_pcd, write_ply
from pathlib import Path

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
ROOT_DIR = Path(os.getcwd())

lidar_data_file = ROOT_DIR / "highway_data" / "splitted" / "dataset.json"
camera_data_dir = ROOT_DIR / "highway_data" / "planar2"
camera_data_file = camera_data_dir / "reference.json"

with open(str(lidar_data_file), "r") as f:
    lidar_data = json.load(f)
camera_data = load_poses(str(camera_data_file))
camera_pose_idx = 33

lidar_section = match_camera_pose_to_lidar_section(
    camera_pose_idx)
camera_pose = camera_data[camera_pose_idx]

camera_poses_of_section = match_lidar_section_to_camera_poses(0)

img0 = imread(camera_data_dir / camera_pose["filename"])

# %%
input_files = [
    "highway_data/splitted/reduced_0_intensity.ply",
]
pcds = []

for input_file in input_files:
    pcds.append(filename_to_pcd(input_file))

o3d.visualization.draw_geometries(pcds)

# %%
points, colors = read_ply(ply_file)

xyz = np.array([(pose['x'], pose['y'], pose['z']) for pose in poses])

# define vector orthogonal to the driving direction
origin = xyz[index]



vec_forward = xyz[index + 1] - xyz[index - 1]
vec_forward /= np.linalg.norm(vec_forward)
vec_up = np.array([0, 0, 1])
vec_right = np.cross(vec_forward, vec_up)

# get points within 2 meter of driving vector
d = np.abs((points - origin) @ vec_right)
sel = d < 2
points_close = points[sel]
intensity_close = colors[sel, 0]

# project points onto driving direction
x = (points_close - origin) @ vec_forward
y = (points_close - origin) @ vec_up

print(x)
# create image coordinates
img_width = 2000
img_height = 1200
scale = 100
y_offset = 3  # meter

x_img = x * scale
y_img = img_height - (y + y_offset) * scale
points_2d = np.c_[x_img, y_img]

# render points
img = render_points(points_2d, intensity_close * 255, (img_width, img_height))
show_image(img)


