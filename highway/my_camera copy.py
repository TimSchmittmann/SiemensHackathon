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

    def colorize(self, index, points, size=4, view_distance=50):
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
        view_cam_points3d = cam_points3d[view_mask]
        view_distances = distances[view_mask]
        view_colors = colors[view_mask]
        if len(view_points3d) == 0:
            return

        view_cam_points3d[:, 0] -= 5
        dim_norm_x = view_cam_points3d[:, 1] - view_cam_points3d[:, 1].min()
        dim_norm_x /= (view_cam_points3d[:, 1].max() - view_cam_points3d[:, 1].min())
        dim_norm_y = view_cam_points3d[:, 0] - view_cam_points3d[:, 0].min()
        dim_norm_y /= (view_cam_points3d[:, 0].max() - view_cam_points3d[:, 0].min())

        scaled_x = np.round((self.camera_img_height-1) * dim_norm_x).astype(np.uint32)
        scaled_y = np.round((self.camera_img_width-1) * dim_norm_y).astype(np.uint32)

        furthest = max(view_distances)
        distances = { "x": [], "y": [], "idx": [], "dist": []}

        for idx in range(len(view_distances)):
            distances["x"].append(scaled_x[idx])
            distances["y"].append(scaled_y[idx])
            distances["idx"].append(idx)
            distances["dist"].append(view_distances[idx])

        # pd.to_csv("../highway_data/distances.csv")
        df = pd.DataFrame(distances)

        min_df = df.groupby(["dist"]).agg(min).reset_index("dist")
        min_df = min_df.sort_values(by=["x", "y"])
        min_df["rolling"] = min_df["dist"].rolling(10).mean()
        min_df = min_df[min_df["dist"] < min_df["rolling"]]
        for row in min_df.itertuples():
            view_colors[row.idx] = img[row.x, row.y] / 255.0

        # for idx in range(len(view_distances)):
        #     key = (scaled_x[idx],scaled_y[idx])
        #     if key not in best_distances:
        #         best_distances[key] = {}
        #         best_distances[key]["min"] = furthest
        #         best_distances[key]["idx"] = idx
        #     elif view_distances[idx] < best_distances[key]["min"]:
        #         best_distances[key]["idx"] = idx
        #         best_distances[key]["min"] = view_distances[idx]
        best_distances = {}
        for idx in range(len(view_distances)):
            key = f"({scaled_x[idx]},{scaled_y[idx]})"
            if key not in best_distances:
                best_distances[key] = {}
                best_distances[key]["min"] = np.Infinity
                best_distances[key]["idx"] = idx
            elif view_distances[idx] < best_distances[key]["min"]:
                best_distances[key]["idx"] = idx
                best_distances[key]["min"] = view_distances[idx]
        key = f"({scaled_x[idx]},{scaled_y[idx]})"
        # if best_distances[key]["idx"] == idx:

        for idx in range(len(view_colors)):
            key = (scaled_x[idx],scaled_y[idx])
            if view_distances[idx] < min_df[min_df.x == scaled_x[idx]]:
                view_colors[idx] = img[scaled_x[idx], scaled_y[idx]] / 255.0

        with open('../highway_data/splitted/center.txt') as f:
            center = np.array(json.load(f))
        points_centered = view_points3d - center
        pcd_centered = to_pcd(points_centered, view_colors)
        o3d.visualization.draw_geometries([pcd_centered])


ply_file = "../highway_data/splitted/reduced_0_intensity.ply"
planar_folder = "../highway_data/planar2"

points, colors = read_ply(ply_file)
proj = Projector(planar_folder)

# camera pose index - number between 0 and len(proj.poses) - 1
index = 33

img = proj.colorize(index, points, colors * 255, view_distance=100)
# show_image(img)

# %%