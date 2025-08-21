import trimesh
import pyrender
import numpy as np
import matplotlib.pyplot as plt

# Load the mesh with vertex colors
mesh = trimesh.load("test2.ply")

# Ensure vertex colors are passed into pyrender
color_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)  # smooth=False preserves vertex color shading

# Create a scene and add the mesh
scene = pyrender.Scene()
scene.add(color_mesh)

# Add a camera
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],  # Camera 2 units in front
    [0.0, 0.0, 0.0, 1.0]
])
scene.add(camera, pose=camera_pose)

# Add light
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
scene.add(light, pose=camera_pose)

# Render the image
renderer = pyrender.OffscreenRenderer(640, 480)
color, _ = renderer.render(scene)

# Display
plt.imshow(color)
plt.axis("off")
plt.show()
