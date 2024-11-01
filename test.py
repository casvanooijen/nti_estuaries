import numpy as np
import ngsolve
import sympy
from scipy.special import erf

from hydrodynamics import Hydrodynamics, select_model_options
import boundary_fitted_coordinates
from spatial_parameter import SpatialParameter
from postprocessing import *
from TruncationBasis import eigbasis_constantAv, unit_harmonic_time_basis
from solve import *
from geometry.geometries import *
from geometry.create_geometry import parametric_geometry, WALLDOWN, WALLUP, RIVER, SEA


# STEP 1: Create geometry and mesh ====================================================================================

# set shape parameters

L = 20e3 # length of estuary in m
B = 3e3 # width of estuary in m

geometrycurves = parametric_rectangle(1, 1) # collection of parametric curves describing the boundary of the estuary 
# we use a unit square because the equations are scaled accordingly

maxh_global = 0.2 # global element size

boundary_maxh_dict = {WALLDOWN: [maxh_global], WALLUP: [maxh_global], RIVER: [maxh_global], SEA: [maxh_global]} # set element size for each partition: allows for variable mesh sizes!
boundary_parameter_partition_dict = {WALLDOWN: [0,1], WALLUP: [0,1], RIVER: [0,1], SEA: [0,1]}  # partition boundary into segments, assuming it starts at 0 and ends at 1

geometry = parametric_geometry(geometrycurves, boundary_parameter_partition_dict, boundary_maxh_dict)

# create mesh

mesh = ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh_global))

# STEP 2: Set parameters =======================================================================================

sem_order = 6 # order of the Spectral Element basis (Dubiner basis)
M = 5 # amount of vertical basis functions to be projected on
imax = 1 # amount of tidal constituents in addition to subtidal flow


# constant physical parameters

g = 9.81 # gravitational acceleration
f = 0
Av = 0.01 # vertical eddy viscosity
Ah = 10
rho0 = 1020 # reference water density
beta = 0 # if density is to be turned off
sigma = 2 / 89428.32720 # M2-frequency (not angular) from Table 3.2 in Gerkema (2019)


# spatially varying physical parameters

bfc = boundary_fitted_coordinates.generate_bfc(mesh, sem_order, "diffusion", alpha=1) # generate coordinates that follow the boundary of the estuary

C = 0.1 # bathymetric steepness parameter 
Hscale = 10 # bathymetric scale parameter
if C > 0:
    Hoffset = (Hscale / 2) * (np.sqrt(np.pi/6) * erf(np.sqrt(6)) - np.sqrt(np.pi/C) * erf(np.sqrt(C))) + 2
else:
    Hoffset = (Hscale / 2) * (np.sqrt(np.pi/6) * erf(np.sqrt(6))) + 2

def make_H(Ho, Hs, steepness): # this is a closure, because if we save this function for later, we don't need to respecify C, Hscale and Hoffset
    def H(xi, eta): # Gaussian bathymetry
        return Ho + Hs * sympy.exp(-steepness * eta**2)
    return H
    
def make_rho(ref_dens, saline_contraction):
    def rho(xi, eta): # water density following the hyperbolic tangent profile from Talke et al. (2009)
        Ssea = 30 # salinity at sea
        xi_L = 0.235
        xi_c = 0.5
        S = Ssea / 2 * (1 - sympy.tanh((xi-xi_c)/xi_L))
        return ref_dens * (1 + saline_contraction * S)
    return rho


def make_ramp(slope, offset):
    def ramp(xi, eta):
        return 0.5 * (sympy.tanh((xi-offset) * slope) + 1)
    return ramp


def R(xi, eta):
    return 0 * xi

H_sp = SpatialParameter(make_H(Hoffset, Hscale, C), bfc)
rho_sp = SpatialParameter(make_rho(rho0, beta), bfc)
R_sp = SpatialParameter(R, bfc)
ramp_sp = SpatialParameter(make_ramp(0, 0.1), bfc)

# STEP 3 : Create Hydrodynamics object ======================================================================

# set options

model_options = select_model_options(bed_bc='no_slip',
                                     surface_in_sigma=False,
                                     veddy_viscosity_assumption='constant', 
                                     horizontal_diffusion=True,
                                     density='depth-independent',
                                     advection_epsilon=1,
                                     advection_influence_matrix=np.array([[False, True],
                                                                          [False, False]]),
                                     x_scaling=L,
                                     y_scaling=B)

# create object

hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, boundary_partition_dict=boundary_parameter_partition_dict, boundary_maxh_dict=boundary_maxh_dict, maxh_global=maxh_global, geometrycurves=geometrycurves)
hydro.set_constant_physical_parameters(Av=Av, sigma=sigma, g=g, f=f, Ah=Ah)
hydro.set_spatial_physical_parameters(H=H_sp, density=rho_sp, R=R_sp, nonlinear_ramp=ramp_sp)

# set tidal forcing

tide_amplitude_list = [0, 1] # per constituent
tide_phase_list = [0, 0]
hydro.set_seaward_boundary_condition(tide_amplitude_list, tide_phase_list)

# set river discharge (only stationary discharges work right now)

discharge_amplitude_list = [0, 0]
discharge_phase_list = [0, 0]
hydro.set_riverine_boundary_condition(discharge_amplitude_list, discharge_phase_list, is_constant=True)

# STEP 5: Solve the equations ===============================================================================

solve(hydro, max_iterations=5, tolerance=1e-12)
hydro.save('test_solution')