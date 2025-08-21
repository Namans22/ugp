import open3d as o3d
import numpy as np
import cv2

# Load sparse point cloud
pcd = o3d.io.read_point_cloud(r"C:\Users\hp\Desktop\ugpp\test2.ply")  

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Normalize normals (optional but recommended)
pcd.orient_normals_consistent_tangent_plane(k=10)

# Apply Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Sample more points from the reconstructed mesh
dense_pcd = mesh.sample_points_uniformly(number_of_points=500000)  # Adjust the number of points as needed

# Ensure the new point cloud has normals
dense_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Get the points and normals of the dense point cloud
points = np.asarray(dense_pcd.points)
normals = np.asarray(dense_pcd.normals)

# Convert normals from [-1, 1] to [0, 255] for RGB mapping
normals_rgb = ((normals + 1) / 2 * 255).astype(np.uint8)

# Project points onto a 2D plane (XY view)
img_size = 512  # Adjust for resolution
image = np.zeros((img_size, img_size, 3), dtype=np.uint8)

# Normalize points to fit in the image
x_min, y_min = np.min(points[:, :2], axis=0)
x_max, y_max = np.max(points[:, :2], axis=0)
points_2d = ((points[:, :2] - np.array([x_min, y_min])) / (np.array([x_max, y_max]) - np.array([x_min, y_min])) * (img_size - 1)).astype(int)


# Place normal colors at projected 2D positions
for (x, y), color in zip(points_2d, normals_rgb):
    image[y, x] = color  # Note: OpenCV uses row-major order (y, x)

# Save and show image
cv2.imwrite("dense_normals_colormap1.png", image)
cv2.imshow("Normal Map", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save and visualize the new dense point cloud
o3d.io.write_point_cloud(r"C:\Users\hp\Desktop\ugpp\dense_test1.ply", dense_pcd)
o3d.visualization.draw_geometries([dense_pcd], point_show_normal=True)
