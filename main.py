from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent         # repo root
WARP_MPM = ROOT / "warp-mpm"
sys.path.insert(0, str(WARP_MPM))

import warp as wp
from mpm_solver_warp import MPM_Simulator_WARP
from engine_utils import *
import torch
from tqdm import tqdm
wp.init()
wp.config.verify_cuda = True

dvc = "cuda:0"

mpm_solver = MPM_Simulator_WARP(10, device=dvc) # initialize with whatever number is fine. it will be reintialized
grid_lim = 2.0 # this control the simulation domain size. [0, grid_lim] in each dimension.
n_grid = 200 # number of grid cells in each dimension.
particle_per_cell = 8 # 8 particles per cell
particle_volume = (grid_lim / n_grid) **3 / particle_per_cell

# ==== Create a cube of particles based on size, lower corner, and grid =====
cube_size = (0.2, 0.2, 0.15)
lower_corner = (1.0, 1.0, 0.075)
vol = cube_size[0] * cube_size[1] * cube_size[2]
n_particles = int(vol / ( (grid_lim / n_grid) **3 ) * particle_per_cell) # number of particles of a cube
cube_tensor = torch.rand(n_particles, 3) * torch.tensor(cube_size) + torch.tensor(lower_corner)

# ==== Copy cubes for different materials ====
stationary_cube = cube_tensor.clone() + torch.tensor([0.4, -0.4, 0.0])
jelly_cube = cube_tensor.clone() + torch.tensor([0.0, -0.4, 0.0])
metal_cube = cube_tensor.clone() + torch.tensor([-0.4, -0.4, 0.0])
sand_cube = cube_tensor.clone() + torch.tensor([0.4, 0.0, 0.0])
foam_cube = cube_tensor.clone() + torch.tensor([0.0, 0.0, 0.0])
snow_cube = cube_tensor.clone() + torch.tensor([-0.4, 0.0, 0.0])
plasticine_cube = cube_tensor.clone() + torch.tensor([0.4, 0.4, 0.0])
fluid_cube = cube_tensor.clone() + torch.tensor([0.0, 0.4, 0.0])
# concatenate all cubes into one tensor for initialization
position_tensor = torch.cat([stationary_cube, jelly_cube, metal_cube, sand_cube, foam_cube, snow_cube, plasticine_cube, fluid_cube], dim=0)
volume_tensor = torch.ones(position_tensor.shape[0]) * particle_volume # set the same volume for all particles.
mpm_solver.load_initial_data_from_torch(position_tensor, volume_tensor, grid_lim=grid_lim, n_grid=n_grid, device=dvc) # initialize the solver with the particle positions and volumes.

# ==== Set parameter globally ====
# This line will set all particles as fluid with bulk modulus of 2000 and density of 1000. 
mpm_solver.set_parameters_dict({
    'bulk_modulus': 2000.0,
    "material": "fluid",
    "density": 1000.0 # this is needed to compute mass from volume not particle count
}, device=dvc)
mpm_solver.set_gravity((0.0, 0.0, -9.81)) # set gravity
# ==== Set parameter locally ====
# This line will set the first cube in the position tensor to be stationary.
mpm_solver.set_parameters_for_particles(
    start_idx=n_particles*0, 
    end_idx=n_particles*1,
    params_dict={
        "material": "stationary",
        "density": 2000.0 # this is needed
    }, 
    device=dvc)
# Set the second cube to be jelly
mpm_solver.set_parameters_for_particles(
    start_idx=n_particles*1, 
    end_idx=n_particles*2,
    params_dict={
        "E": 2000.0,
        "nu": 0.3,
        "material": "jelly",
        "density": 1200.0 # this is needed
    },
    device=dvc)
# Set the third cube to be metal (StVK elasticity + Von Mises plasticity)
mpm_solver.set_parameters_for_particles(
    start_idx=n_particles*2,
    end_idx=n_particles*3,
    params_dict={
        "E": 200_000,           # Young's modulus (stiff) 
        "nu": 0.3,             # Poisson's ratio
        "material": "metal",
        "density": 7800.0,     # steel-like density
        "yield_stress": 200_000.0, # onset of plastic deformation 
        "hardening": 1.0,      # enable strain hardening
        "xi": 0.1,             # hardening rate
    },
    device=dvc)
# Set the fourth cube to be sand (Drucker-Prager plasticity)
mpm_solver.set_parameters_for_particles(
    start_idx=n_particles*3,
    end_idx=n_particles*4,
    params_dict={
        "E": 2000.0,           # Young's modulus
        "nu": 0.2,             # Poisson's ratio (sand is less compressible laterally)
        "material": "sand",
        "density": 1600.0,     # dry sand density
        "friction_angle": 45.0,# internal friction angle in degrees
        "softening": 0.1,      # softening rate
    },
    device=dvc)
# Set the fifth cube to be foam (viscoplastic, toothpaste-like)
mpm_solver.set_parameters_for_particles(
    start_idx=n_particles*4,
    end_idx=n_particles*5,
    params_dict={
        "E": 1500.0,           # Young's modulus
        "nu": 0.45,             # Poisson's ratio (nearly incompressible)
        "material": "foam",
        "density": 300.0,      # lightweight foam
        "yield_stress": 50.0,  # low yield stress
        "plastic_viscosity": 0.5,  # viscous flow resistance
    },
    device=dvc)
# Set the sixth cube to be snow (FCR elasticity, no plasticity)
mpm_solver.set_parameters_for_particles(
    start_idx=n_particles*5,
    end_idx=n_particles*6,
    params_dict={
        "E": 2000.0,            # Young's modulus (soft)
        "nu": 0.2,             # Poisson's ratio
        "material": "snow",
        "density": 200.0,      # light snow density
    },
    device=dvc)
# Set the seventh cube to be plasticine (FCR + Von Mises with damage)
mpm_solver.set_parameters_for_particles(
    start_idx=n_particles*6,
    end_idx=n_particles*7,
    params_dict={
        "E": 1200.0,           # Young's modulus
        "nu": 0.35,            # Poisson's ratio
        "material": "plasticine",
        "density": 1400.0,     # clay-like density
        "yield_stress": 75.0,  # moderate yield stress
        "softening": 0.15,     # damage softening rate
        "hardening": 0.8,      # enable hardening
        "xi": 0.05,            # hardening rate
    },
    device=dvc)
# The eighth cube stays as fluid (already set globally)

# ==== Boundary =====
# The material point cannot be too close to the boundary
# because the kernel will sample neighboring grid nodes 
# and if the particle is too close to the boundary, 
# it may sample outside the grid which will cause NaN.
# 3 grid cells paddings are safe
dx = grid_lim / n_grid
padding = 3 * dx
mpm_solver.add_surface_collider((0.0, 0.0, padding), (0.0,0.0,1.0), 'sticky')  # add ground plane
mpm_solver.add_surface_collider((padding, 0.0, 0.0), (1.0,0.0,0.0), 'slip')  
mpm_solver.add_surface_collider((grid_lim-padding, 0.0, 0.0), (-1.0,0.0,0.0), 'slip')  
mpm_solver.add_surface_collider((0.0, padding, 0.0), (0.0,1.0,0.0), 'slip')  
mpm_solver.add_surface_collider((0.0, grid_lim-padding, 0.0), (0.0,-1.0,0.0), 'slip')  
# mpm_solver.add_bounding_box()  # You can also use this. add_bounding_box will add a slip collider at 3*dx inside each boundary.
directory_to_save = './sim_results/multi_material'

# ==== selection tensor ====
# After we initialize the particles, we can get the selection tensor which controls whether a particle is active or not.
# If selection is 0 for a particle, it is active. Any value other than 0 will make it inactive.
selection = mpm_solver.export_particle_selection_to_torch()
print("Initial active particles:", (selection == 0).sum().item())
# Let's make all cube inactive at the beginning except the stationary one.
selection[:] = 1.0 # make all inactive
selection[n_particles*0:n_particles*1] = 0.0 # make the first cube (stationary) active
mpm_solver.import_particle_selection_from_torch(selection, device=dvc) # put it back

# ==== Simulation loop =====
n_steps = 10000
dt = 0.001
n_cubes = 8
released_cube = 1
visualize_step = 20
for k in tqdm(range(1, n_steps+1)):
    mpm_solver.p2g2p(k, dt, device=dvc)
    # release the next cube every 800 steps
    if k % 1250 == 0 and k//1250 < n_cubes:
        selection_tensor = mpm_solver.export_particle_selection_to_torch()
        selection_tensor[n_particles*released_cube:n_particles*(released_cube+1)] = 0
        selection_tensor[:n_particles*released_cube] = 1 # deactivate all previous cubes
        mpm_solver.import_particle_selection_from_torch(selection_tensor, device=dvc)
        released_cube += 1
    if k % visualize_step == 0:
        pos = mpm_solver.export_particle_x_to_torch()
        if torch.isnan(pos).any(): # safe guard
            print(f"NaN detected at step {k}!")
            break
        save_data_at_frame(mpm_solver, directory_to_save, k//visualize_step, save_to_ply=True, save_to_h5=False)



