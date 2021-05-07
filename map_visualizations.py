
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
pose_file = "../highway_data/planar1/reference.json"

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

# export to maps
map = folium.Map(location=long_lat_height[0, :2], zoom_start=13)
folium.PolyLine(
    locations=long_lat_height[:, :2],
    popup="Planar 1 Reference",
    color="#000000"
).add_to(map)

exit_signs = folium.FeatureGroup(name='Exit signs', show=True)
trees = folium.FeatureGroup(name='Trees', show=True)
bridges = folium.FeatureGroup(name='Bridges', show=True)

map.add_child(exit_signs)
map.add_child(trees)
map.add_child(bridges)

exit_sign_cluster = MarkerCluster().add_to(exit_signs)
tree_cluster = MarkerCluster().add_to(trees)
bridge_cluster = MarkerCluster().add_to(bridges)

folium.LayerControl().add_to(map)

exit_sign_locations = [488, 1764, 333]
bridge_locations = [38, 1565, 888]
bridge_heights = [4.5, 5, 5.25]

exit_i = skimage.io.imread("icons/abfahrtsschild.png")
for loc in exit_sign_locations:
    exit_sign_icon = folium.features.CustomIcon(
        exit_i, icon_size=(40,40))
    pos = long_lat_height[loc]
    folium.Marker(
        [pos[0], pos[1]], popup="<i>height: 4.50 m</i>", tooltip="Exit sign",
        icon=exit_sign_icon
    ).add_to(exit_sign_cluster)

bridge_i = skimage.io.imread("icons/bridge.svg.png")
for loc, height in zip(bridge_locations, bridge_heights):
    bridge_icon = folium.features.CustomIcon(bridge_i, icon_size=(80,30))
    pos = long_lat_height[loc]
    folium.Marker(
        [pos[0], pos[1]], popup=f"<i>height: {height} m</i>", tooltip="Bridge",
        icon=bridge_icon
    ).add_to(bridge_cluster)

map.save("../map.html")



# %%
