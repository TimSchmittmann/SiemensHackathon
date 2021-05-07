
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
tree_locations = [125,222, 277,  881, 1171]
tree_locations += list(range(255, 500, 5))
tree_locations += list(range(670, 875, 5))
tree_locations += list(range(900, 1150, 5))
tree_locations += list(range(1325, 1355, 5))


exit_i = skimage.io.imread("../icons/abfahrtsschild.png")
for loc in exit_sign_locations:
    exit_sign_icon = folium.features.CustomIcon(
        exit_i, icon_size=(90,90))
    pos = long_lat_height[loc]
    folium.Marker(
        [pos[0], pos[1]], popup="<i>Mt. Hood Meadows</i>", tooltip="Test",
        icon=exit_sign_icon
    ).add_to(exit_sign_cluster)

bridge_i = skimage.io.imread("../icons/bridge.svg.png")
for loc in bridge_locations:
    bridge_icon = folium.features.CustomIcon(bridge_i, icon_size=(80,30))
    pos = long_lat_height[loc]
    folium.Marker(
        [pos[0], pos[1]], popup="<i>Mt. Hood Meadows</i>", tooltip="Test",
        icon=bridge_icon
    ).add_to(bridge_cluster)

tree_i = skimage.io.imread("../icons/tree.png")
for loc in tree_locations:
    tree_icon = folium.features.CustomIcon(
        tree_i, icon_size=(20,20))
    pos = long_lat_height[loc]
    folium.Marker(
        [pos[0], pos[1]], popup="<i>Mt. Hood Meadows</i>", tooltip="Test",
        icon=tree_icon
    ).add_to(tree_cluster)

map.save("../map.html")



# %%
