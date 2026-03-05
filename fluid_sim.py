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


# ==== Create a sand cube of particles based on size, lower corner, and grid =====
cube_size = (0.1, 0.1, 0.02)
lower_corner = (0.5, 0.5, 0.30)
n_grid = 100
grid_lim = 1.0
particle_per_cell = 8 # 8 particles per cell

vol = cube_size[0] * cube_size[1] * cube_size[2]
n_particles = int(vol / ( (grid_lim / n_grid) **3 ) * particle_per_cell)
unit_cube_tensor = torch.rand(n_particles, 3) * torch.tensor(cube_size) + torch.tensor(lower_corner)
particle_volume = (grid_lim / n_grid) **3 / particle_per_cell

# copy cube data but set selection to 1
n_copies = 10
position_tensor = torch.cat([unit_cube_tensor] * n_copies, dim=0)
volume_tensor = torch.ones(n_particles*n_copies) * particle_volume

print(f"total particles: {n_particles}")
mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor, n_grid=n_grid, device=dvc, grid_lim=grid_lim)
# export selection
selection_tensor = mpm_solver.export_particle_selection_to_torch()
selection_tensor[:] = 1 # set all to unselected
# import back the selection
mpm_solver.import_particle_selection_from_torch(selection_tensor)

material_params = {
    'bulk_modulus': 2000.0,
    "material": "fluid",
    'friction_angle': 35,
    'g': [0.0, 0.0, -9.8],
    "density": 1000.0
}
mpm_solver.set_parameters_dict(material_params)

# add bounding box for fluid
box_length = 0.4
mpm_solver.add_surface_collider((0.0, 0.0, 0.13), (0.0,0.0,1.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5-box_length/2., 0.0, 0.0), (1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.5+box_length/2., 0.0, 0.0), (-1.0,0.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5+box_length/2., 0.0), (0.0,-1.0,0.0), 'cut', 0.0)
mpm_solver.add_surface_collider((0.0, 0.5-box_length/2., 0.0), (0.0,1.0,0.0), 'cut', 0.0)

directory_to_save = './sim_results/fluid'

save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)
released_copy = 0
for k in tqdm.tqdm(range(1,2000), desc="Simulating"):
    mpm_solver.p2g2p(k, 0.001, device=dvc)
    if k % 50 == 0 and released_copy < n_copies:
        # export selection
        selection_tensor = mpm_solver.export_particle_selection_to_torch()
        selection_tensor[released_copy*n_particles:(released_copy+1)*n_particles] = 0 # set the 2nd copy to zero
        mpm_solver.import_particle_selection_from_torch(selection_tensor)
        released_copy += 1

    if k % 10 == 0:
        pos = mpm_solver.export_particle_x_to_torch()
        if torch.isnan(pos).any():
            print(f"NaN detected at step {k}!")
            break
        save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)
