import open3d as o3d

# Load point cloud
pcd = o3d.io.read_point_cloud(r"C:\Users\hp\Desktop\ugpp\test2.ply")  # Replace with your file

# Estimate normals
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

# Normalize normals (optional but recommended)
pcd.orient_normals_consistent_tangent_plane(k=10)

# Visualize
o3d.visualization.draw_geometries([pcd], point_show_normal=True)
#o3d.visualization.draw(pcd)