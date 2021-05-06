# %%
import numpy as np

from highway.camera import load_poses, show_image
from highway.pointcloud import read_ply, render_points
import pandas as pd
import numpy as np
from filecache import filecache
import json

def load_cp_lp_poses(lidar_data, camera_data):
    camera_poses = np.array([[p["x"],p["y"],p["z"]] for p in camera_data])
    lidar_poses = np.array([pose for section in lidar_data["sections"] for pose in section["poses"]])
    return lidar_poses, camera_poses

@filecache(365 * 24 * 60 * 60)
def find_best_matches(lidar_poses, camera_poses):
    best_matches_lp_cp = {}
    best_matches_cp_lp = {}
    for lp_idx, lp in enumerate(lidar_poses):
        best_diff = np.Infinity
        for cp_idx, cp in enumerate(camera_poses):
            diff = np.abs(np.sum(lp - cp))
            if diff < best_diff:
                best_diff = diff
            else:
                best_matches_lp_cp[lp_idx] = cp_idx
                best_matches_cp_lp[lp_idx] = cp_idx
                break
    return best_matches_lp_cp, best_matches_cp_lp

def get_lidar_section_by_pose_idx(lidar_data, pose_idx):
    nr_poses = 0
    for section in lidar_data["sections"]:
        nr_poses += len(section["poses"])
        if nr_poses > pose_idx:
            return section
    return False

def get_pose_indices_of_lidar_section(lidar_data, section_idx):
    nr_poses = 0
    for s_idx, section in enumerate(lidar_data["sections"]):
        nr_poses += len(section["poses"])
        if s_idx == section_idx:
            return list(range(nr_poses, nr_poses+len(section["poses"])))

with open("highway_data/splitted/dataset.json", "r") as f:
    lidar_data = json.load(f)

if __name__ == '__main__':
    camera_data = load_poses("highway_data/planar2/reference.json")
    lidar_poses, camera_poses = load_cp_lp_poses(lidar_data, camera_data)
    best_matches_lp_cp, best_matches_cp_lp = find_best_matches(lidar_poses, camera_poses)

    matching_lidar_pose_idx = best_matches_cp_lp[750]
    lidar_section = get_lidar_section_by_pose_idx(lidar_data, matching_lidar_pose_idx)

    pose_indices_of_section = get_pose_indices_of_lidar_section(lidar_data, 22)
    matching_camera_pose_indices = [best_matches_lp_cp[p_idx] for p_idx in pose_indices_of_section]

# %%
