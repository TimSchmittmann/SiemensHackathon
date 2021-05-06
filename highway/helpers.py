# %%
import numpy as np

from .camera import load_poses, show_image
from .pointcloud import read_ply, render_points
import pandas as pd
import numpy as np
from filecache import filecache
import json
from pathlib import Path

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

def match_camera_pose_to_lidar_section(camera_pose: int,
        camera_data_file = "highway_data/planar2/reference.json",
        lidar_data_file = "highway_data/splitted/dataset.json"):
    with open(lidar_data_file, "r") as f:
        lidar_data = json.load(f)
    camera_data = load_poses(camera_data_file)
    lidar_poses, camera_poses = load_cp_lp_poses(lidar_data, camera_data)
    best_matches_lp_cp, best_matches_cp_lp =  find_best_matches(lidar_poses, camera_poses)

    matching_lidar_pose_idx = best_matches_cp_lp[camera_pose]
    return get_lidar_section_by_pose_idx(lidar_data, matching_lidar_pose_idx)

def match_lidar_section_to_camera_poses(lidar_section: int,
        camera_data_file= "highway_data/planar2/reference.json",
        lidar_data_file = "highway_data/splitted/dataset.json"):
    with open(lidar_data_file, "r") as f:
        lidar_data = json.load(f)
    camera_data = load_poses(camera_data_file)
    lidar_poses, camera_poses = load_cp_lp_poses(lidar_data, camera_data)
    best_matches_lp_cp, best_matches_cp_lp =  find_best_matches(lidar_poses, camera_poses)

    pose_indices_of_section = get_pose_indices_of_lidar_section(lidar_data, lidar_section)
    matching_camera_pose_indices = [best_matches_lp_cp[p_idx] for p_idx in pose_indices_of_section]
    return [d for i, d in enumerate(camera_data) if i in matching_camera_pose_indices]

if __name__ == '__main__':
    camera_poses_of_section = match_lidar_section_to_camera_poses(21)
    lidar_section_of_camera_pose = match_camera_pose_to_lidar_section(750)

# %%
