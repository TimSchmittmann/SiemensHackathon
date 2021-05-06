
"""
Convert pointcloud coordinates into GPS and visualize them with OpenStreetMaps
"""

# %%
import numpy as np
import folium
import folium.features
from folium.plugins import MarkerCluster
import skimage.io
from highway.camera import load_poses
from highway.transform import to_gps
from random import randrange
# %%
pose_file = "highway_data/planar1/reference.json"

poses = load_poses(pose_file)
xyz = np.array([(pose['x'], pose['y'], pose['z']) for pose in poses])

print()
print("XYZ coordinates of the camera trajectory")
print(xyz)
print(len(xyz))

long_lat_height = np.array(to_gps(xyz[:, 0], xyz[:, 1], xyz[:, 2])).T

print()
print("GPS coordinates of the camera trajectory")
print(long_lat_height)

# %%
def add_random_marker(el, long_lat_height,
    icon_url, icon_size=(20,20),repeat=3):
    for _ in range(repeat):
        stop_sign_icon = folium.features.CustomIcon(
            skimage.io.imread(icon_url), icon_size=icon_size)
        pos = long_lat_height[randrange(len(long_lat_height))]
        folium.Marker(
            [pos[0], pos[1]], popup="<i>Mt. Hood Meadows</i>", tooltip="Test",
            icon=stop_sign_icon
        ).add_to(el)

# %%
# export to maps
map = folium.Map(location=long_lat_height[0, :2], zoom_start=13)
folium.PolyLine(
    locations=long_lat_height[:, :2],
    popup="Planar 1 Reference",
    color="#000000"
).add_to(map)

stop_signs = folium.FeatureGroup(name='Stop signs', show=True)
trees = folium.FeatureGroup(name='Trees', show=True)
bridges = folium.FeatureGroup(name='Bridges', show=True)

map.add_child(stop_signs)
map.add_child(trees)
map.add_child(bridges)

stop_sign_cluster = MarkerCluster().add_to(stop_signs)
tree_cluster = MarkerCluster().add_to(trees)
bridge_cluster = MarkerCluster().add_to(bridges)

folium.LayerControl().add_to(map)

add_random_marker(stop_sign_cluster, long_lat_height,
    icon_url="icons/1024px-Stopsign.png", icon_size=(20,20), repeat=3)
add_random_marker(tree_cluster, long_lat_height,
    icon_url="icons/bridge.svg.png", icon_size=(80,30), repeat=3)
add_random_marker(bridge_cluster, long_lat_height,
    icon_url="icons/tree.png", icon_size=(20,20), repeat=3)

map.save("example_3_index.html")



# %%
