"""
Display images & project 3D points into camera images.
"""
# %%
import json
import pathlib

import numpy as np
import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import open3d as o3d
from highway.pointcloud import read_ply, to_pcd, write_ply
from pathlib import Path
import pandas as pd
from highway.helpers import match_camera_pose_to_lidar_section, match_lidar_section_to_camera_poses
import os
import math
import cv2

def load_poses(filename):
    """ Load camera poses """
    with open(str(filename)) as f:
        return json.load(f)


def show_image(img):
    """ Show given image in GUI """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation="bilinear")
    plt.axis('off')
    plt.gcf().subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    plt.show()


# %%
ROOT_DIR = Path(os.getcwd())

SPLITTED_DIR = ROOT_DIR / ".." / "highway_data" / "splitted"
lidar_data_file = SPLITTED_DIR / "dataset.json"
camera_data_dir = ROOT_DIR / ".." / "highway_data" / "planar2"
camera_data_file = camera_data_dir / "reference.json"

with open(str(lidar_data_file), "r") as f:
    lidar_data = json.load(f)
camera_data = load_poses(str(camera_data_file))

camera_poses_of_section = match_lidar_section_to_camera_poses(1, camera_data_file, lidar_data_file)
camara_pose_idx, camera_pose = camera_poses_of_section[0]
# camera pose index - number between 0 and len(proj.poses) - 1


# %%
class Projector:
    """ Project 3d points into camera images"""
    def __init__(self, planar_folder):
        """
        :param planar_folder: folder with planar data
        """
        self.folder = pathlib.Path(planar_folder)

        # load camera extrinsic poses
        self.poses = load_poses(self.folder / "reference.json")

        # load camera intrinsic model
        #    Comes from a trdip sqlite file that was attached to the data from Oslo:
        #    s3://smaragd-storage/DTC/UC_dyn_lichtraumprofil/Oslo_Pointcloud/TMX9318091201-000030.tridb
        self.camera_img_width = 2454
        self.camera_img_height = 2056
        K = np.identity(3)
        K[0, 0] = 2470
        K[1, 1] = 2470
        K[0, 2] = (self.camera_img_width/2) - (0.05765e-3 / 3.45e-6)
        K[1, 2] = (self.camera_img_height/2) - (0.02778e-3 / 3.45e-6)
        self.K = K
        self.distortion = np.array([0.00027041, -1.79468e-05, 0, 0, 0])

    def render(self, index, points, colors, size=4, view_distance=50):
        """
        Project given points into the image with given index. Returns rendered image

        :param index: index of the image / camera pose. See self.poses to select an image / pose.
        :param points: 3d points as n x 3 numpy array in image coordinates
        :param colors: colors of each point as n x 3 numpy array. Defined between 0 and 255
        :param size: size of each point in. points further away are drawn more small
        :param view_distance: only points that are within this distance in meter to the camera origin are drawn.
        :return: rendered color image
        """
        assert 0 <= index < len(self.poses)
        img_path = self.folder / self.poses[index]['filename']
        img = cv2.imread(str(img_path))
        # project_and_draw(img, points, colors, self.poses[index]['full-pose'], size, max_view_distance)

        pose = self.poses[index]['full-pose']
        rot_vec = -np.array([pose['rx'], pose['ry'], pose['rz']])
        t_vec = -np.array([pose['tx'], pose['ty'], pose['tz']]) @ cv2.Rodrigues(rot_vec)[0].T

        # select points which are close
        cam_pos = -np.matmul(cv2.Rodrigues(rot_vec)[0].T, t_vec)
        distances = np.linalg.norm(points - cam_pos, axis=1)
        view_mask = distances < view_distance

        # select points which are in front of camera
        cam_points3d = points @ cv2.Rodrigues(rot_vec)[0].T + t_vec
        view_mask = view_mask & (cam_points3d[:, 2] > 0)

        view_points3d = points[view_mask]
        view_distances = distances[view_mask]
        view_colors = colors[view_mask]
        if len(view_points3d) == 0:
            return
        view_points2d = cv2.projectPoints(view_points3d, rot_vec, t_vec, self.K, self.distortion)[0].reshape(-1, 2)

        p = view_points2d
        selection = np.all((p[:, 0] >= 0, p[:, 0] < img.shape[1], p[:, 1] >= 0, p[:, 1] < img.shape[0]), axis=0)
        p = p[selection]

        # closest points are at 4 meter distance
        norm_distances = view_distances[selection] / 4.0
        shift = 3
        factor = (1 << shift)
        def I(x_):
            return int(x_ * factor + 0.5)
        for i in range(0, len(p)):
            cv2.circle(img, (I(p[i][0]), I(p[i][1])), I(size / norm_distances[i]), view_colors[i], -1, shift=shift)

        return img

    def colorize(self, index, points, size=4, view_distance=100):
        """
        Project given points into the image with given index. Returns rendered image

        :param index: index of the image / camera pose. See self.poses to select an image / pose.
        :param points: 3d points as n x 3 numpy array in image coordinates
        :param colors: colors of each point as n x 3 numpy array. Defined between 0 and 255
        :param size: size of each point in. points further away are drawn more small
        :param view_distance: only points that are within this distance in meter to the camera origin are drawn.
        :return: rendered color image
        """
        assert 0 <= index < len(self.poses)
        img_path = self.folder / self.poses[index]['filename']
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # project_and_draw(img, points, colors, self.poses[index]['full-pose'], size, max_view_distance)

        # theta = np.radians(90)
        # c, s = np.cos(theta), np.sin(theta)
        # R = np.array(((c, -s, 0), (s, c, 0), (0,0,1)))
        # print(R)

        pose = self.poses[index]['full-pose']
        rot_vec = -np.array([pose['rx'], pose['ry'], pose['rz']])
        t_vec = -np.array([pose['tx'], pose['ty'], pose['tz']]) @ cv2.Rodrigues(rot_vec)[0].T

        # select points which are close
        cam_pos = -np.matmul(cv2.Rodrigues(rot_vec)[0].T, t_vec)
        distances = np.linalg.norm(points - cam_pos, axis=1)
        view_mask = distances < view_distance

        # select points which are in front of camera
        cam_points3d = points @ cv2.Rodrigues(rot_vec)[0].T + t_vec
        view_mask = view_mask & (cam_points3d[:, 2] > 0)

        view_points3d = points[view_mask]
        view_cam_points3d = cam_points3d[view_mask]
        view_distances = distances[view_mask]
        view_colors = colors[view_mask]
        if len(view_points3d) == 0:
            return

        nx = 1
        ny = 0
        dim_norm_x = view_cam_points3d[:, nx] - view_cam_points3d[:, nx].min()
        dim_norm_x /= (view_cam_points3d[:, nx].max() - view_cam_points3d[:, nx].min())
        # dim_norm_x = 1 - dim_norm_x
        dim_norm_y = view_cam_points3d[:, ny] - view_cam_points3d[:, ny].min()
        dim_norm_y /= (view_cam_points3d[:, ny].max() - view_cam_points3d[:, ny].min())
        # dim_norm_y = 1 - dim_norm_y

        scaled_x = np.round((self.camera_img_height-1) * dim_norm_x).astype(np.uint32)
        scaled_y = np.round((self.camera_img_width-1) * dim_norm_y).astype(np.uint32)

        dist_dict = { "x": [], "y": [], "idx": [], "dist": []}

        for idx in range(len(view_distances)):
            dist_dict["x"].append(scaled_x[idx])
            dist_dict["y"].append(scaled_y[idx])
            dist_dict["idx"].append(idx)
            dist_dict["dist"].append(view_distances[idx])

        min_df = pd.DataFrame(dist_dict)
        # min_df = min_df.groupby(["dist"]).agg(min).reset_index("dist")
        # min_df.to_csv("../highway_data/distances.csv")

        # min_df = pd.read_csv("../highway_data/distances.csv")

        # min_df = min_df.sort_values(by=["x", "y"])
        # min_df["rolling"] = min_df["dist"].rolling(50).mean()
        # min_df = min_df[min_df["dist"] < min_df["rolling"]]
        for row in min_df.itertuples():
            if row.x > 1700 and row.y < 2000:
                continue
            # if view_distances[row.idx] < 5:
            #     continue
            x_offset = 100
            y_offset = -220
            if row.x+x_offset < self.camera_img_height and row.y >= y_offset:
                view_colors[row.idx] = img[row.x+y_offset, row.y+y_offset] / 255.0

        # cv3 = o3d.utility.Vector3dVector(view_colors)
        # print(cv3)
        # for idx in range(len(view_distances)):
        #     key = (scaled_x[idx],scaled_y[idx])
        #     if key not in best_distances:
        #         best_distances[key] = {}
        #         best_distances[key]["min"] = furthest
        #         best_distances[key]["idx"] = idx
        #     elif view_distances[idx] < best_distances[key]["min"]:
        #         best_distances[key]["idx"] = idx
        #         best_distances[key]["min"] = view_distances[idx]
        # best_distances = {}
        # for idx in range(len(view_distances)):
        #     key = f"({scaled_x[idx]},{scaled_y[idx]})"
        #     if key not in best_distances:
        #         best_distances[key] = {}
        #         best_distances[key]["min"] = np.Infinity
        #         best_distances[key]["idx"] = idx
        #     elif view_distances[idx] < best_distances[key]["min"]:
        #         best_distances[key]["idx"] = idx
        #         best_distances[key]["min"] = view_distances[idx]
        # key = f"({scaled_x[idx]},{scaled_y[idx]})"
        # if best_distances[key]["idx"] == idx:

        # for idx in range(len(view_colors)):
        #     key = (scaled_x[idx],scaled_y[idx])
        #     if view_distances[idx] < min_df[min_df.x == ]:
        #         view_colors[idx] = img[scaled_x[idx], scaled_y[idx]] / 255.0

        with open('../highway_data/splitted/center.txt') as f:
            center = np.array(json.load(f))
        points_centered = view_points3d - center
        # tr = np.copy(view_colors[:,0])
        # tg = np.copy(view_colors[:,1])
        # tb = np.copy(view_colors[:,2])
        # view_colors[:,0] = tb
        # view_colors[:,1] = tg
        # view_colors[:,2] = tr
        pcd_centered = to_pcd(points_centered, view_colors)
        # tx = np.copy(points_centered[:,0])
        # ty = np.copy(points_centered[:,1])
        # tz = np.copy(points_centered[:,2])
        # points_centered[:,0] = -ty
        # points_centered[:,1] = tx
        # points_centered[:,2] = -tz
        # print(points_centered)
        return points_centered, pcd_centered
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_centered)
        # R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
        # pcd_r = pcd.rotate(R, center=(0, 0, 0))
        # pcd_r.colors = o3d.utility.Vector3dVector(view_colors)


# %%
planar_folder = "../highway_data/planar2"
proj = Projector(planar_folder)
pcds = []
for section_idx in range(15, 17):
    print(section_idx)
    lidar_filename = f"reduced_{section_idx}_intensity.ply"
    ply_file = SPLITTED_DIR / lidar_filename
    points, colors = read_ply(ply_file)

    camera_poses_of_section = match_lidar_section_to_camera_poses(section_idx, camera_data_file, lidar_data_file)

    camera_pose_idx = camera_poses_of_section[0][0]

    points_centered, pcd_centered = proj.colorize(camera_pose_idx, points, colors * 255, view_distance=100)

    write_ply(f"{camera_pose_idx}_output.ply", points_centered, colors)
    pcds.append(pcd_centered)

o3d.visualization.draw_geometries(pcds)