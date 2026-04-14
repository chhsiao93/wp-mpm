from pathlib import Path
import sys
import tqdm
ROOT = Path(__file__).resolve().parent         # repo root
WARP_MPM = ROOT / "warp-mpm"
sys.path.insert(0, str(WARP_MPM))

import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
wp.init()
wp.config.verify_cuda = True


dvc = "cuda:0"

mpm_solver = MPM_Simulator_WARP(10) # initialize with whatever number is fine. it will be reintialized
cube_size = (0.05, 0.05, 0.05)
lower_corner = (0.0, 0.0, 0.0)
n_grid = 128
grid_lim = 1.0
grid_dx = grid_lim / n_grid
particle_per_cell = 8 # 8 particles per cell
particle_volume = (grid_lim / n_grid) **3 / particle_per_cell

# ==== Create a cube of particles based on size, lower corner, and grid =====

vol = cube_size[0] * cube_size[1] * cube_size[2]
n_particles_per_cube = int(vol / ( (grid_lim / n_grid) **3 ) * particle_per_cell)
unit_cube_tensor = torch.rand(n_particles_per_cube, 3) * torch.tensor(cube_size) + torch.tensor(lower_corner)

cube1_tensor = unit_cube_tensor.clone() + torch.tensor([0.3, 0.44, 0.1])
cube2_tensor = unit_cube_tensor.clone() + torch.tensor([0.3, 0.51, 0.1])
cube3_tensor = unit_cube_tensor.clone() + torch.tensor([0.3, 0.47, 0.16])

# ==== Create a sphere ====
radius = 0.05
# sample points from a cube and then filter out outsiders
vol = (2*radius)**3
n_particles_per_sphere = int(vol / ( (grid_lim / n_grid) **3 ) * particle_per_cell)
unit_cube_tensor = torch.rand(n_particles_per_sphere, 3 ) * (2*radius) + torch.tensor([-radius, -radius, -radius]) # centered at origin
dist_from_center = torch.norm(unit_cube_tensor, dim=1)
sphere_tensor = unit_cube_tensor[dist_from_center <= radius]
sphere_tensor += torch.tensor([0.7, 0.5, 0.3]) # move to the center of the ramp
n_particles_per_sphere = sphere_tensor.shape[0] # correct number

# ==== Combine cubes and sphere ====
position_tensor = torch.cat([cube1_tensor, cube2_tensor, cube3_tensor, sphere_tensor], dim=0)
n_particles = position_tensor.shape[0]
volume_tensor = torch.ones(n_particles) * particle_volume

print(f"total particles: {n_particles}")
mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor, n_grid=n_grid, device=dvc, grid_lim=grid_lim)

material_params = {
        "E": 2_000_000,           # Young's modulus (stiff) 
        "nu": 0.3,             # Poisson's ratio
        "material": "metal",
        "density": 7800.0,     # steel-like density
        "yield_stress": 200_000.0, # onset of plastic deformation 
        "hardening": 1.0,      # enable strain hardening
        "xi": 0.1,             # hardening rate
        'g': [0.0, 0.0, -9.8],
        
}
mpm_solver.set_parameters_dict(material_params)

# velocity_tensor = mpm_solver.export_particle_v_to_torch()
# velocity_tensor[n_particles*0:n_particles*1, :] = torch.tensor([0.0, 0.0, 0.0])
# velocity_tensor[n_particles*1:n_particles*2, :] = torch.tensor([-0.2, 0.0, 0.0])
# velocity_tensor[n_particles*2:n_particles*3, :] = torch.tensor([0.2, 0.0, 0.0])

# add bounding box for fluid
n_padding_cells = 3
# restitution = 0.01
ground_friction = 0.5
ramp_friction = 0.1
padding_dist = n_padding_cells * grid_dx
mpm_solver.add_surface_collider((0.0, 0.0, 0.1), (0.0,0.0,1.0), 'slip', friction=ground_friction)
mpm_solver.add_surface_collider((padding_dist, 0.0, 0.0), (1.0,0.0,0.0), 'slip', friction=ground_friction)
mpm_solver.add_surface_collider((grid_lim-padding_dist, 0.0, 0.0), (-1.0,0.0,0.0), 'slip', friction=ground_friction)
mpm_solver.add_surface_collider((0.0, padding_dist, 0.0), (0.0,1.0,0.0), 'slip', friction=ground_friction)
mpm_solver.add_surface_collider((0.0, grid_lim-padding_dist, 0.0), (0.0,-1.0,0.0), 'slip', friction=ground_friction)
# add the ramp collider
plane_center = (0.5, 0.6, 0.1)
plane_normal = (-0.5, 0.0, 1.0)
mpm_solver.add_surface_collider(plane_center, plane_normal, 'slip', friction=ramp_friction)

directory_to_save = './sim_results/ramp'
import os
os.makedirs(directory_to_save, exist_ok=True)

for k in tqdm.tqdm(range(30000), desc="Simulating"):
    mpm_solver.p2g2p(k, 0.0001, device=dvc)

    if k % 100 == 0:
        position = mpm_solver.export_particle_x_to_torch().cpu().numpy()
        num_particles = (position).shape[0]
        position = position.astype(np.float32)
        
        filename = f"{directory_to_save}/sim_{k//100:010d}.ply"
        with open(filename, 'wb') as f: # write binary
            header = f"""ply
format binary_little_endian 1.0
element vertex {num_particles}
property float x
property float y
property float z
end_header
"""
            f.write(str.encode(header))
            f.write(position.tobytes())
            
# ==== Visualization with open3d ====
import open3d as o3d
import imageio.v2 as imageio
import glob
# ==== setting =====
width, height = 800, 608
point_size = 2.0
lookat = [0.5, 0.5, 0.2]
eye = [0.5, 1.0, 0.4]
up = [0, 0, 1]
grid_size = 1.0
grid_divisions = 10
grid_height = 0.1
# frames = []
output_file = 'ani_ramp.mp4'

renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
# Setup material for point cloud
material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultUnlit"
material.point_size = point_size

# Setup grid
# Create grid on XY plane at Z=grid_height
grid_lines = []
step = grid_size / grid_divisions

# Lines parallel to X-axis
for i in range(grid_divisions + 1):
    y = i * step
    grid_lines.append([[0, y, grid_height], [grid_size, y, grid_height]])

# Lines parallel to Y-axis
for i in range(grid_divisions + 1):
    x = i * step
    grid_lines.append([[x, 0, grid_height], [x, grid_size, grid_height]])

# Create LineSet
points = []
lines = []
for i, (start, end) in enumerate(grid_lines):
    points.append(start)
    points.append(end)
    lines.append([2*i, 2*i+1])

line_set = o3d.geometry.LineSet()
line_set.points = o3d.utility.Vector3dVector(points)
line_set.lines = o3d.utility.Vector2iVector(lines)
# Set grid color (light gray)
colors = [[0.7, 0.7, 0.7] for _ in range(len(lines))]
line_set.colors = o3d.utility.Vector3dVector(colors)

grid_material = o3d.visualization.rendering.MaterialRecord()
grid_material.shader = "unlitLine"
grid_material.line_width = 1.0
renderer.scene.add_geometry("grid", line_set, grid_material)

# Create ramp mesh for visualization derived from plane_center and plane_normal
center = np.array(plane_center, dtype=np.float64)
normal = np.array(plane_normal, dtype=np.float64)
normal /= np.linalg.norm(normal)
# Two orthogonal tangent vectors spanning the plane
ref = np.array([0.0, 1.0, 0.0]) if abs(normal[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
tangent1 = np.cross(normal, ref); tangent1 /= np.linalg.norm(tangent1)
tangent2 = np.cross(normal, tangent1); tangent2 /= np.linalg.norm(tangent2)
half_extent = 0.5
ramp_vertices = np.array([
    center - half_extent * tangent1 - half_extent * tangent2,
    center + half_extent * tangent1 - half_extent * tangent2,
    center + half_extent * tangent1 + half_extent * tangent2,
    center - half_extent * tangent1 + half_extent * tangent2,
], dtype=np.float64)
ramp_triangles = np.array([[0, 1, 2], [0, 2, 3]])
ramp_mesh = o3d.geometry.TriangleMesh()
ramp_mesh.vertices = o3d.utility.Vector3dVector(ramp_vertices)
ramp_mesh.triangles = o3d.utility.Vector3iVector(ramp_triangles)
ramp_mesh.compute_vertex_normals()
ramp_material = o3d.visualization.rendering.MaterialRecord()
ramp_material.shader = "defaultLit"
ramp_material.base_color = [0.55, 0.55, 0.55, 1.0]
renderer.scene.add_geometry("ramp", ramp_mesh, ramp_material)

writer = imageio.get_writer(
    output_file,
    fps=30,
    quality=6,
    codec='libx264',
    pixelformat='yuv420p'
    )
# find all ply files and sort by name
ply_files = sorted(glob.glob(f'{directory_to_save}/sim_*.ply'))

for i in tqdm.tqdm(range(len(ply_files)), desc="Writing video frames"):
    # load ply file
    filename = ply_files[i]
    pcd = o3d.io.read_point_cloud(filename)
    # render point cloud
    colors = np.asarray(pcd.points)
    # height coloring
    z_lo, z_hi = 0.00, 0.3
    t_elev = np.clip((colors[:, 2] - z_lo) / (z_hi - z_lo), 0.0, 1.0)[:, None]
    colors = (1 - t_elev) * np.array([0.0, 0.0, 1.0]) + t_elev * np.array([1.0, 0.0, 0.0])
    pcd.colors = o3d.utility.Vector3dVector(colors)
    renderer.scene.add_geometry("pcd", pcd, material)
    
    renderer.setup_camera(60.0, lookat, eye, up)
    image = renderer.render_to_image()
    writer.append_data(np.asarray(image))
    renderer.scene.remove_geometry("pcd")
writer.close()