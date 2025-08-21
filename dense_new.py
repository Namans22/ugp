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
pcd = mesh.sample_points_uniformly(number_of_points=500000)  # Adjust point count as needed

# Ensure the new point cloud has normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Convert normals from [-1, 1] to [0, 255] for color mapping
normals = np.asarray(pcd.normals)
normals_rgb = ((normals + 1) / 2 * 255).astype(np.uint8) / 255.0  # Normalize for Open3D

# Assign colors to point cloud
pcd.colors = o3d.utility.Vector3dVector(normals_rgb)

# Create an Open3D visualizer
vis = o3d.visualization.Visualizer()
vis.create_window(visible=False)  # Offscreen rendering
vis.add_geometry(pcd)

# Set camera parameters for proper perspective rendering
ctr = vis.get_view_control()
ctr.set_zoom(0.7)  # Adjust zoom
ctr.set_front([0, 0, -1])  # Keep the face forward
ctr.set_lookat([0, 0, 0])  # Centered
ctr.set_up([0, -1, 0])  # Maintain orientation

# Capture the rendered image
vis.poll_events()
vis.update_renderer()
image = vis.capture_screen_float_buffer(True)  # Get image
vis.destroy_window()

# Convert Open3D image to NumPy format
image_np = (np.asarray(image) * 255).astype(np.uint8)

# Save and show
cv2.imwrite(r"C:\Users\hp\Desktop\ugpp\normal_map_fixed2.png", image_np)
#cv2.imshow("Normal Map", image_np)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
