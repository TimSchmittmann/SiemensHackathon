
"""
Project 3d points, like we have it for pointcloud, into the images.
"""
# %%
from highway.pointcloud import read_ply
from highway.camera import Projector, show_image
import pandas as pd
import numpy as np
import json

# %%
# trajectory_df = pd.read_csv("highway_data/trajectory.csv", sep="\t")

with open("highway_data/splitted/dataset.json", "r") as f:
    lidar_data = json.load(f)
lidar_data["sections"][0]

# %%
with open("highway_data/planar1/reference.json", "r") as f:
    planar1 = json.load(f)

planar1
print(planar1[0]["x"])
print(planar1[0]["y"])
print(planar1[0]["z"])

# %%

ply_file = "highway_data/splitted/reduced_22_intensity.ply"
planar_folder = "highway_data/planar2"

points, colors = read_ply(ply_file)
proj = Projector(planar_folder)

# camera pose index - number between 0 and len(proj.poses) - 1
index = 750

img = proj.render(index, points, colors * 255)
show_image(img)



# %%
