from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent         # repo root
WARP_MPM = ROOT / "warp-mpm"
sys.path.insert(0, str(WARP_MPM))

import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
import tqdm

def write_ply(position, selection, output_dir, frame):
    # position is (n,3), selection is (n,)
    position = position[selection==0] # only write unselected particles
    filename = f"{output_dir}/frame_{str(frame).zfill(5)}.ply"
    num_particles = (position).shape[0]
    position = position.astype(np.float32)
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

def add_cube(lower_corner, cube_size, particle_per_cell, cell_dx):
    vol = cube_size[0] * cube_size[1] * cube_size[2]
    n_particles = int(vol / ( (cell_dx) **3 ) * particle_per_cell)
    position_tensor = torch.rand(n_particles, 3) * torch.tensor(cube_size) + torch.tensor(lower_corner)
    particle_volume = (cell_dx) **3 / particle_per_cell
    volume_tensor = torch.ones(n_particles) * particle_volume
    return position_tensor, volume_tensor
wp.init()
wp.config.verify_cuda = True
dvc = "cuda:0"

mpm_solver = MPM_Simulator_WARP(10, device=dvc) # initialize with whatever number is fine. it will be reintialized
grid_lim = 4.0 # simulation region is [0, grid_lim]
n_grid = 200
# ==== Create a water cube of particles based on size, lower corner, and grid =====
cube_size = (0.3, 0.3, 0.05)
lower_corner = (2.0, 2.0, 0.5)
particle_per_cell = 8 # 8 particles per cell
padding =  1 # padding for point cloud collider, in number of grid cells. 

# vol = cube_size[0] * cube_size[1] * cube_size[2]
# n_particles = int(vol / ( (grid_lim / n_grid) **3 ) * particle_per_cell)
# unit_cube_tensor = torch.rand(n_particles, 3) * torch.tensor(cube_size) + torch.tensor(lower_corner)
# particle_volume = (grid_lim / n_grid) **3 / particle_per_cell

# copy cube data but set selection to 1
n_copies = 200
position_tensor, volume_tensor = add_cube(lower_corner, cube_size, particle_per_cell, grid_lim/n_grid)
n_particles = position_tensor.shape[0]
for i in range(n_copies-1):
    unit_cube_pos, unit_cube_vol = add_cube(lower_corner, cube_size, particle_per_cell, grid_lim/n_grid)
    position_tensor = torch.cat([position_tensor, unit_cube_pos], dim=0)
    volume_tensor = torch.cat([volume_tensor, unit_cube_vol], dim=0)

print(f"particles per cube: {n_particles}")
mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor, n_grid=n_grid, device=dvc, grid_lim=grid_lim)
# export selection
selection_tensor = mpm_solver.export_particle_selection_to_torch()
selection_tensor[:] = 1 # set all to unselected
# import back the selection
mpm_solver.import_particle_selection_from_torch(selection_tensor)
# initialize velocity with a specific value
velocity_tensor = torch.tile(torch.tensor([-0.5, -0.5, -0.2]), (n_particles*n_copies, 1)) # shape (n_particles*n_copies, 3)
mpm_solver.import_particle_v_from_torch(velocity_tensor)
material_params = {
    'bulk_modulus': 2000.0,
    "material": "fluid",
    'friction_angle': 35, #### Change this to set different friction angle
    'g': [0.0, 0.0, -4.0],
    "density": 1000.0 # this is needed to compute mass from volume not particle count
}
mpm_solver.set_parameters_dict(material_params, device=dvc)

mpm_solver.add_surface_collider((0.0, 0.0, 0.01), (0.0,0.0,1.0), 'slip')  # add ground plane
mpm_solver.add_bounding_box()  # prevent particles from escaping the grid

# Load ply file for point cloud collider
from plyfile import PlyData
plydata = PlyData.read('sarah_prc_edited.ply')

vertex = plydata['vertex']
point_cloud = np.column_stack([vertex['x'], vertex['y'], vertex['z']]).astype(np.float64)


# rescale and shift pcd
point_cloud = point_cloud * 0.5 + np.array([2.0, 2.0, 0.0]) # scale and shift if needed
# remove points outside the grid
point_cloud = point_cloud[(point_cloud[:,0] >= 0) & (point_cloud[:,0] <= grid_lim) & 
                          (point_cloud[:,1] >= 0) & (point_cloud[:,1] <= grid_lim) & 
                          (point_cloud[:,2] >= 0) & (point_cloud[:,2] <= grid_lim)]
# remove points above z=0.2
# point_cloud = point_cloud[point_cloud[:,2] <= 0.2]
print(f"Point cloud collider with {point_cloud.shape[0]} points created.")
print(f"pcd min: {point_cloud.min(axis=0)}, max: {point_cloud.max(axis=0)}")
mpm_solver.add_point_cloud_collider(point_cloud, padding=padding, start_time=0.0, end_time=999.0, device=dvc)

directory_to_save = './sim_results/prc'

# save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)
released_copy = 0 # initialize released copy index
for k in tqdm.tqdm(range(0,40000), desc="Simulating"):
    mpm_solver.p2g2p(k, 0.0001, device=dvc)
    if k % 100 == 0 and released_copy < n_copies:
        # export selection
        selection_tensor = mpm_solver.export_particle_selection_to_torch()
        selection_tensor[released_copy*n_particles:(released_copy+1)*n_particles] = 0 # set the 2nd copy to zero
        mpm_solver.import_particle_selection_from_torch(selection_tensor)
        released_copy += 1
    if k % 200 == 0:
        pos = mpm_solver.export_particle_x_to_torch()
        if torch.isnan(pos).any():
            print(f"NaN detected at step {k}!")
            break
        # save_data_at_frame(mpm_solver, directory_to_save, k//200, save_to_ply=True, save_to_h5=False)
        write_ply(pos.cpu().numpy(), selection_tensor.cpu().numpy(), directory_to_save, k//200)



