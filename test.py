import numpy as np
import matplotlib.pyplot as plt
import ngsolve
import sympy
from scipy.special import erf

from hydrodynamics import Hydrodynamics
from modeloptions import ModelOptions
import boundary_fitted_coordinates
from spatial_parameter import SpatialParameter
from postprocessing import *
from TruncationBasis import eigbasis_constantAv, harmonic_time_basis
import mesh_functions
from geometry.geometries import *
from geometry.create_geometry import parametric_geometry, WALLDOWN, WALLUP, RIVER, SEA, BOUNDARY_DICT


# STEP 1: Create geometry and mesh ====================================================================================

# set shape parameters

L = 10e3 # length of estuary in m
B = 3e3 # width of estuary in m

# create geometry

geometrycurves = parametric_rectangle(B, L) # collection of parametric curves describing the boundary of the estuary

maxh_global = 1e3 # global element size

boundary_parameter_partition_dict = {WALLDOWN: [0,1], WALLUP: [0,1], RIVER: [0,1], SEA: [0,1]}  # partition boundary into segments, assuming it starts at 0 and ends at 1
boundary_maxh_dict = {WALLDOWN: [maxh_global], WALLUP: [maxh_global], RIVER: [maxh_global], SEA: [maxh_global]} # set element size for each partition: allows for variable mesh sizes!

geometry = parametric_geometry(geometrycurves, boundary_parameter_partition_dict, boundary_maxh_dict)

# create mesh

mesh = ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh_global))



# STEP 2: Set parameters =======================================================================================

sem_order = 6 # order of the Spectral Element basis (Dubiner basis)
M = 7 # amount of vertical basis functions to be projected on
imax = 1 # amount of tidal constituents in addition to subtidal flow


# constant physical parameters

g = 9.81 # gravitational acceleration
f = 1e-4 # Coriolis acceleration
Av = 0.01 # vertical eddy viscosity
rho0 = 1020 # reference water density
beta = 7.6e-4 # coefficient of saline contraction
sigma = 2 / 89428.32720 # M2-frequency (not angular) from Table 3.2 in Gerkema (2019)


# spatially varying physical parameters

bfc = boundary_fitted_coordinates.generate_bfc(mesh, sem_order, "diffusion", alpha=1) # generate coordinates that follow the boundary of the estuary

C = 6 # bathymetric steepness parameter 
Hscale = 10 # bathymetric scale parameter
if C > 0:
    Hoffset = (Hscale / 2) * (np.sqrt(np.pi/6) * erf(np.sqrt(6)) - np.sqrt(np.pi/C) * erf(np.sqrt(C))) + 2
else:
    Hoffset = (Hscale / 2) * (np.sqrt(np.pi/6) * erf(np.sqrt(6))) + 2

def H(xi, eta): # Gaussian bathymetry
    return Hoffset + Hscale * sympy.exp(-C * eta**2)

def rho(xi, eta): # water density following the hyperbolic tangent profile from Talke et al. (2009)
    Ssea = 30 # salinity at sea
    xi_L = 0.235
    xi_c = 0.5
    S = Ssea / 2 * (1 - sympy.tanh((xi-xi_c)/xi_L))
    return rho0 * (1 + beta * S)

H_sp = SpatialParameter(H, bfc)
rho_sp = SpatialParameter(rho, bfc)

# STEP 3: Define expansion bases ============================================================================

vertical_basis = eigbasis_constantAv
time_basis = harmonic_time_basis(sigma)


# STEP 4 : Create Hydrodynamics object ======================================================================

# set options

model_options = ModelOptions(bed_bc='no_slip', leading_order_surface=True, veddy_viscosity_assumption='constant', density='depth-independent', advection_epsilon=0)

# create object

hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, time_basis, vertical_basis)
hydro.set_constant_physical_parameters(Av=Av, sigma=sigma, g=g, f=f)
hydro.set_spatial_physical_parameters(H_sp, rho_sp)

# set tidal forcing

tide_amplitude_list = [0, 1] # per constituent
tide_phase_list = [0, 0]
hydro.set_seaward_boundary_condition(tide_amplitude_list, tide_phase_list)

# set river discharge (only stationary discharges work right now)

discharge_amplitude_list = [0.1, 0]
discharge_phase_list = [0, 0]
hydro.set_riverine_boundary_condition(discharge_amplitude_list, discharge_phase_list, is_constant=True)



# STEP 5: Solve the equations ===============================================================================

advection_epsilon_list = [0] # this is a list to allow for homology methods in the Newton method; the solution procedure could really use a rewrite as well; this is not so intuitive

hydro.solve(advection_epsilon_list, skip_nonlinear=True, maxits=10, tol=1e-9, method='pardiso')


# STEP 6: Postprocessing =====================================================================================

postpro = PostProcessing(hydro) # by creating this object, velocity fields are created from the 'raw' solution

# set matplotlib font parameters
plt.rcParams["mathtext.fontset"] = 'dejavuserif'
plt.rcParams["font.family"] = 'serif'

# endpoints of cross-section plots
p1 = np.array([L/2,B/2])
p2 = np.array([L/2,-B/2])


# create plots
postpro.plot_colormap(postpro.u_DA_abs(1), refinement_level=4, show_mesh=False, title="Amplitude of depth-averaged semidiurnal along-channel velocity", clabel="Velocity [m/s]")
postpro.plot_colormap(postpro.v_DA_abs(1), refinement_level=4, show_mesh=False, title="Amplitude of depth-averaged semidiurnal cross-channel velocity", clabel="Velocity [m/s]")
postpro.plot_colormap(postpro.gamma_abs(1), refinement_level=4, show_mesh=False, title="Amplitude of semidiurnal surface elevation", clabel="Surface elevation [m]")
postpro.plot_colormap(postpro.u_DA(0), refinement_level=4, show_mesh=False, title="Residual along-channel velocity", clabel="Velocity [m/s]", center_range=True, cmap='RdBu')

postpro.plot_vertical_cross_section(lambda sig: postpro.u_abs(1, sig), "Amplitude of semidiurnal along-channel velocity in central cross-section", 'Velocity [m/s]', p1, p2, 1000, 1000)
postpro.plot_vertical_cross_section(lambda sig: postpro.v_abs(1, sig), "Amplitude of semidiurnal cross-channel velocity in central cross-section", 'Velocity [m/s]', p1, p2, 1000, 1000)

plt.show()











