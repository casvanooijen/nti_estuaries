import numpy as np
import matplotlib.pyplot as plt
import ngsolve
import sympy
from scipy.special import erf
import pickle
import dill

from hydrodynamics import Hydrodynamics, select_model_options, load_hydrodynamics
import boundary_fitted_coordinates
from spatial_parameter import SpatialParameter
from postprocessing import *
from TruncationBasis import eigbasis_constantAv, unit_harmonic_time_basis, harmonic_time_basis
import mesh_functions
from geometry.geometries import *
from geometry.create_geometry import parametric_geometry, WALLDOWN, WALLUP, RIVER, SEA, BOUNDARY_DICT


# STEP 1: Create geometry and mesh ====================================================================================

# set shape parameters

L = 10e3 # length of estuary in m
B = 3e3 # width of estuary in m

method = 'new'

if method == 'new':
    geometrycurves = parametric_rectangle(1, 1) # collection of parametric curves describing the boundary of the estuary 
    # we use a unit square because the equations are scaled accordingly

    maxh_global = 0.2 # global element size
elif method == 'old':
    geometrycurves = parametric_rectangle(B, L) # collection of parametric curves describing the boundary of the estuary 
    # we use a unit square because the equations are scaled accordingly

    maxh_global = 0.95e3 # global element size

boundary_maxh_dict = {WALLDOWN: [maxh_global], WALLUP: [maxh_global], RIVER: [maxh_global], SEA: [maxh_global]} # set element size for each partition: allows for variable mesh sizes!
boundary_parameter_partition_dict = {WALLDOWN: [0,1], WALLUP: [0,1], RIVER: [0,1], SEA: [0,1]}  # partition boundary into segments, assuming it starts at 0 and ends at 1

geometry = parametric_geometry(geometrycurves, boundary_parameter_partition_dict, boundary_maxh_dict)

# create mesh

mesh = ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh_global))
# mesh.ngmesh.SetGeometry(None)



# STEP 2: Set parameters =======================================================================================

sem_order = 4 # order of the Spectral Element basis (Dubiner basis)
M = 7 # amount of vertical basis functions to be projected on
imax = 1 # amount of tidal constituents in addition to subtidal flow


# constant physical parameters

g = 9.81 # gravitational acceleration
f = 1e-4 # Coriolis acceleration
# f = 0
Av = 0.01 # vertical eddy viscosity
rho0 = 1020 # reference water density
# beta = 7.6e-4 # coefficient of saline contraction
beta = 0 # if density is to be turned off
sigma = 2 / 89428.32720 # M2-frequency (not angular) from Table 3.2 in Gerkema (2019)


# spatially varying physical parameters

bfc = boundary_fitted_coordinates.generate_bfc(mesh, sem_order, "diffusion", alpha=1) # generate coordinates that follow the boundary of the estuary

C = 6 # bathymetric steepness parameter 
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
        return sympy.tanh((xi-offset) * slope) + 1
    return ramp


def R(xi, eta):
    return 0 * xi

H_sp = SpatialParameter(make_H(Hoffset, Hscale, C), bfc)
rho_sp = SpatialParameter(make_rho(rho0, beta), bfc)
R_sp = SpatialParameter(R, bfc)
ramp_sp = SpatialParameter(make_ramp(10, 0.2), bfc)

# STEP 3: Define expansion bases ============================================================================

vertical_basis = eigbasis_constantAv
if method == 'new':
    time_basis = unit_harmonic_time_basis
elif method == 'old':
    time_basis = harmonic_time_basis(sigma)



# STEP 4 : Create Hydrodynamics object ======================================================================

# set options

if method == 'new':
    x_scale = L
    y_scale = B
elif method == 'old':
    x_scale = 1.
    y_scale = 1.

model_options = select_model_options(bed_bc='no_slip',
                                     surface_in_sigma=False,
                                     veddy_viscosity_assumption='constant', 
                                     density='depth-independent',
                                     advection_epsilon=0.5,
                                     advection_influence_matrix=np.array([[True, True],
                                                                          [False, False]]),
                                     x_scaling=x_scale,
                                     y_scaling=y_scale)

# create object

hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, time_basis, vertical_basis)
hydro.set_constant_physical_parameters(Av=Av, sigma=sigma, g=g, f=f)
hydro.set_spatial_physical_parameters(H=H_sp, density=rho_sp, R=R_sp, nonlinear_ramp=ramp_sp)




# refine mesh based on bathymetry gradient

hydro.hrefine(1.2, 2)

# set tidal forcing

tide_amplitude_list = [0, 1] # per constituent
tide_phase_list = [0, 0]
hydro.set_seaward_boundary_condition(tide_amplitude_list, tide_phase_list)

# set river discharge (only stationary discharges work right now)

discharge_amplitude_list = [0, 0]
discharge_phase_list = [0, 0]
hydro.set_riverine_boundary_condition(discharge_amplitude_list, discharge_phase_list, is_constant=True)



# STEP 5: Solve the equations ===============================================================================

advection_epsilon_list = [0, 0.5] # this is a list to allow for homology methods in the Newton method; the solution procedure could really use a rewrite as well; this is not so intuitive



hydro.solve(advection_epsilon_list, skip_nonlinear=False, maxits=5, tol=1e-7, method='pardiso')
# hydro.save('test_solution', format='npy')

# STEP 6: Postprocessing =====================================================================================

postpro = PostProcessing(hydro) # by creating this object, velocity fields are created from the 'raw' solution

# set matplotlib font parameters
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
plt.rcParams["font.family"] = 'serif'

# endpoints of cross-section plots

if method == 'new':
    p1 = np.array([0.6,0.5])
    p2 = np.array([0.6,-0.5])
elif method == 'old':
    p1 = np.array([L/2, B/2])
    p2 = np.array([L/2,-B/2])



# # create plots
postpro.plot_colormap(ramp_sp.cf, refinement_level=4, title='Non-linear ramp', clabel='-', figsize=(7,4))
postpro.plot_colormap(postpro.u_DA_abs(1), refinement_level=4, show_mesh=False, title='Amplitude of depth-averaged semidiurnal along-channel velocity', clabel='Velocity [m/s]', figsize=(7,4))
postpro.plot_colormap(postpro.v_DA_abs(1), refinement_level=4, show_mesh=False, title="Amplitude of depth-averaged semidiurnal cross-channel velocity", clabel="Velocity [m/s]", figsize=(7,4))
postpro.plot_colormap(postpro.gamma_abs(1), refinement_level=4, show_mesh=False, title="Amplitude of semidiurnal surface elevation", clabel="Surface elevation [m]", figsize=(7,4))
postpro.plot_colormap(postpro.u_DA(0), refinement_level=4, show_mesh=False, title="Residual along-channel velocity", clabel="Velocity [m/s]", center_range=True, cmap='RdBu', figsize=(7,4))

# postpro.plot_vertical_cross_section(lambda sig: postpro.u(1, sig), "Semidiurnal positive component along-channel velocity in central cross-section", 'Velocity [m/s]', p1, p2, 1000, 1000, figsize=(7,4), center_range=True, cmap='RdBu')
postpro.plot_vertical_cross_section(lambda sig: postpro.u(0, sig), "Residual along-channel velocity in central cross-section", 'Velocity [m/s]', p1, p2, 1000, 1000, figsize=(7,4), center_range=True, cmap='RdBu')

postpro.plot_mesh2d('Mesh', color='k', linewidth=0.5)

plt.show()











