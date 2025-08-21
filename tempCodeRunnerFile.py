import open3d as o3d
import numpy as np
import cv2

# Load sparse point cloud
pcd = o3d.io.read_point_cloud(r"C:\Users\hp\Desktop\ugpp\test2.ply")  

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
pcd.orient_normals_consistent_tangent_plane(k=10)

# Apply Poisson surface reconstruction to densify the point cloud
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Sample more points from the reconstructed mesh
dense_pcd = mesh.sample_points_uniformly(number_of_points=500000)  # Adjust point count as needed

# Ensure the new point cloud has normals
dense_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Create an Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  # Offscreen rendering
vis.add_geometry(dense_pcd)

# Set camera parameters for front-face projection
ctr = vis.get_view_control()
ctr.set_zoom(0.7)
ctr.set_front([0, 0, -1])  # Looking at the front
ctr.set_lookat([0, 0, 0])  # Centering the face
ctr.set_up([0, -1, 0])  # Keep orientation correct

# Capture the depth buffer
vis.poll_events()
vis.update_renderer()
depth_image = np.asarray(vis.capture_depth_float_buffer(True))  # Get depth buffer
vis.destroy_window()

# Convert normals from [-1, 1] to [0, 255] for RGB mapping
normals = np.asarray(dense_pcd.normals)
normals_rgb = ((normals + 1) / 2 * 255).astype(np.uint8)

# Get point coordinates
points = np.asarray(dense_pcd.points)

# Project 3D points to 2D using a simple camera projection
img_size = depth_image.shape  # Match the captured depth image size
x_min, y_min = np.min(points[:, :2], axis=0)
x_max, y_max = np.max(points[:, :2], axis=0)

# Normalize and map points to 2D image coordinates
points_2d = ((points[:, :2] - np.array([x_min, y_min])) / (np.array([x_max, y_max]) - np.array([x_min, y_min])) * (img_size[1] - 1)).astype(int)

# Clamp points to ensure they stay within bounds of the image size
points_2d[:, 0] = np.clip(points_2d[:, 0], 0, img_size[1] - 1)  # x coordinates should be between 0 and img_width-1
points_2d[:, 1] = np.clip(points_2d[:, 1], 0, img_size[0] - 1)  # y coordinates should be between 0 and img_height-1

# Create an empty image for storing normal colors
image = np.zeros((img_size[0], img_size[1], 3), dtype=np.uint8)

# Map normal colors to the 2D projection
for (x, y), color in zip(points_2d, normals_rgb):
    image[y, x] = color  # Note: OpenCV uses row-major order (y, x)

# Save and show the normal map image
cv2.imwrite(r"C:\Users\hp\Desktop\ugpp\dense_normals_colormap.png", image)
cv2.imshow("Normal Map", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final dense point cloud
o3d.io.write_point_cloud(r"C:\Users\hp\Desktop\ugpp\dense_test1.ply", dense_pcd)
o3d.visualization.draw_geometries([dense_pcd], point_show_normal=True)
