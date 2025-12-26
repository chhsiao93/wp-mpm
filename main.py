from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent         # repo root
WARP_MPM = ROOT / "warp-mpm"
sys.path.insert(0, str(WARP_MPM))

import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
wp.init()
wp.config.verify_cuda = True


dvc = "cuda:1"

mpm_solver = MPM_Simulator_WARP(10, device=dvc) # initialize with whatever number is fine. it will be reintialized

# ==== Create a sand cube of particles based on size, lower corner, and grid =====
cube_size = (0.2, 0.2, 0.2)
lower_corner = (0.4, 0.4, 0.105)
n_grid = 150
particle_per_cell = 8 # 8 particles per cell

vol = cube_size[0] * cube_size[1] * cube_size[2]
n_particles = int(vol / ( (1.0/n_grid) **3 ) * particle_per_cell)
position_tensor = torch.rand(n_particles, 3) * torch.tensor(cube_size) + torch.tensor(lower_corner)
particle_volume = (1.0 / n_grid) **3 / particle_per_cell
print(f"Particle volume: {particle_volume}, total particles: {n_particles}")
volume_tensor = torch.ones(n_particles) * particle_volume
mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor, n_grid=n_grid, device=dvc)


material_params = {
    'E': 2000,
    'nu': 0.2,
    "material": "sand",
    'friction_angle': 35, #### Change this to set different friction angle
    'g': [0.0, 0.0, -9.8],
    "density": 1800.0 # this is needed to compute mass from volume not particle count
}
mpm_solver.set_parameters_dict(material_params, device=dvc)

mpm_solver.finalize_mu_lam_bulk(device=dvc) # set mu and lambda from the E and nu input

mpm_solver.add_surface_collider((0.0, 0.0, 0.1), (0.0,0.0,1.0), 'slip', 0.5)  # add ground plane


directory_to_save = './sim_results/sand'

save_data_at_frame(mpm_solver, directory_to_save, 0, save_to_ply=True, save_to_h5=False)

for k in range(1,1000):
    mpm_solver.p2g2p(k, 0.002, device=dvc)
    if k % 10 == 0:
        save_data_at_frame(mpm_solver, directory_to_save, k, save_to_ply=True, save_to_h5=False)



