import open3d as o3d

# Load sparse point cloud
pcd = o3d.io.read_point_cloud(r"C:\Users\hp\Desktop\ugpp\test2.ply")  

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Normalize normals (optional but recommended)
pcd.orient_normals_consistent_tangent_plane(k=10)

# Apply Poisson surface reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

# Sample more points from the reconstructed mesh
dense_pcd = mesh.sample_points_uniformly(number_of_points=10000)  # Adjust as needed

# Save and visualize the new dense point cloud
o3d.io.write_point_cloud(r"C:\Users\hp\Desktop\ugpp\dense_test2.ply", dense_pcd)

# Visualize the dense point cloud
o3d.visualization.draw_geometries([dense_pcd], point_show_normal=True)
