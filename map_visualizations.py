
"""
Convert pointcloud coordinates into GPS and visualize them with OpenStreetMaps
"""

# %%
from folium.map import LayerControl
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
m = folium.Map(location=long_lat_height[0, :2], zoom_start=13)
folium.PolyLine(
    locations=long_lat_height[:, :2],
    popup="Planar 1 Reference",
    color="#000000"
).add_to(m)

m.get_root().header.add_child(folium.CssLink('./map.css'))

exit_signs = folium.FeatureGroup(name='''
    <span class="label-btn strong" id="label1">Exit signs:</span>
    <span class="total">Total signs on route: 5</span>, <span >Blocking signs: 1</span><br/> Estimated removal cost: <span id="cost0" class="strong">25000</span><span class="strong">€</span></span>
    <hr>'''
    , show=True)
# trees = folium.FeatureGroup(name='''
#     <span class="label-btn  strong">Trees:</span>
#     <span class="total">Total trees on route: 1437</span>, <span >Blocking tree: 63</span><br/> Estimated removal cost: <span id="cost1" class="strong">237432</span><span class="strong">€</span></span>
#     <hr>'''
#     , show=True)
bridges = folium.FeatureGroup(name='''
    <span class="label-btn strong" id="label2">Bridges:</span>
    <span class="total">Total bridges on route: 3</span>, <span >Blocking bridges: 3</span><br/> Estimated removal cost: <span  id="cost1" class="strong">237432</span><span class="strong">€</span></span>
    <hr><br/><span class="strong">Final cost estimation: <span  id="costfinal strong">262432</span><span class="strong">€</span></span>'''
    , show=True)

m.add_child(exit_signs)
# m.add_child(trees)
m.add_child(bridges)

exit_sign_cluster = MarkerCluster().add_to(exit_signs)
# tree_cluster = MarkerCluster().add_to(trees)
bridge_cluster = MarkerCluster().add_to(bridges)

layer_control = folium.LayerControl()
layer_control.add_to(m)

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
