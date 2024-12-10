import numpy as np
import ngsolve
import sympy
from scipy.special import erf

from hydrodynamics import Hydrodynamics, select_model_options
import boundary_fitted_coordinates
from spatial_parameter import SpatialParameter
from postprocessing import *
from solve import *
from geometry.geometries import *
from geometry.create_geometry import parametric_geometry, WALLDOWN, WALLUP, RIVER, SEA
from geometry.meshing import generate_mesh



# STEP 1: Create geometry and mesh ====================================================================================

# set shape parameters

L = 50e3 # length of estuary in m
B = 3e3 # width of estuary in m

geometrycurves = parametric_rectangle(1, 1) # collection of parametric curves describing the boundary of the estuary 
# we use a unit square because the equations are scaled accordingly
geometry = parametric_geometry(geometrycurves)

# create mesh
# maxh_global = 0.04
# maxh_global = 0.11 # global element size
# maxh_global = 0.3
# mesh = ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh_global))

num_cells = (18,18)
ngmesh = generate_mesh(geometry, method='structured_quads', num_cells=num_cells)
mesh = ngsolve.Mesh(ngmesh)


# STEP 2: Set parameters =======================================================================================

sem_order = 1 # order of the Spectral Element basis (Dubiner basis)
M = 5 # amount of vertical basis functions to be projected on
imax = 1 # amount of tidal constituents in addition to subtidal flow

# constant physical parameters

g = 9.81 # gravitational acceleration
f = 0
Av = 0.01 / 12 # vertical eddy viscosity
sf = 10
Ah = 10
rho0 = 1020 # reference water density
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
        return 0.5 * (sympy.tanh((xi-offset) * slope) + 1)
    return ramp


def R(xi, eta):
    return 0 * xi

H_sp = SpatialParameter(make_H(Hoffset, Hscale, C), bfc)
# H_sp.cf = Hoffset
# H_sp.gradient_cf = (0,0)
rho_sp = SpatialParameter(make_rho(rho0, beta), bfc)
R_sp = SpatialParameter(R, bfc)
ramp_sp = SpatialParameter(make_ramp(10, 0.1), bfc)

# STEP 3 : Create Hydrodynamics object ======================================================================

# set options

model_options = select_model_options(bed_bc='no_slip',
                                     surface_in_sigma=False,
                                     veddy_viscosity_assumption='depth-scaled&constantprofile', 
                                     horizontal_diffusion=True,
                                     density='depth-independent',
                                     advection_epsilon=1,
                                     advection_influence_matrix=np.array([[False, True],[False, False]]),
                                     x_scaling=L,
                                     y_scaling=B,
                                     element_type='taylor-hood',
                                     mesh_generation_method='structured_quads')

# create object
hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, geometrycurves=geometrycurves, num_cells=num_cells)
# hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, geometrycurves=geometrycurves, maxh_global=maxh_global)
hydro.set_constant_physical_parameters(Av=Av, sigma=sigma, g=g, f=f, Ah=Ah, sf=sf)
hydro.set_spatial_physical_parameters(H=H_sp, density=rho_sp, R=R_sp, nonlinear_ramp=ramp_sp)

# set tidal forcing

tide_amplitude_list = [0, 1] # per constituent
tide_phase_list = [0, 0]
hydro.set_seaward_boundary_condition(tide_amplitude_list, tide_phase_list, enhanced_winant_conforming=True)

# set river discharge (only stationary discharges wo
# rk right now)

discharge = Hscale * B * 0.00
hydro.set_riverine_boundary_condition(discharge)

# STEP 5: Solve the equations ===============================================================================

# cont_parameters = {'advection_epsilon': [1, 1], 'Av': [10, 1]}
solve(hydro, max_iterations=5, tolerance=1e-12, linear_solver='pardiso', matrix_analysis=False)
hydro.save('test')

p1 = np.array([0.5, 0.5])
p2 = np.array([0.5, -0.5])

pp = PostProcessing(hydro)

lon_advec_forcing = lambda sig: H_sp.cf * (pp.u(1, sig)*pp.vx(1, sig) + pp.u(-1, sig)*pp.vx(-1, sig))
lat_advec_forcing = lambda sig: H_sp.cf * (pp.v(1, sig)*pp.vy(1, sig) + pp.v(-1, sig)*pp.vy(-1, sig))
vert_advec_forcing = lambda sig: pp.w(1, sig)*pp.vsig(1, sig) + pp.w(-1, sig)*pp.vsig(-1, sig)

# pp.plot_mesh2d('Mesh', linewidth=0.5, color='k')

pp.plot_colormap(pp.u_DA(1), refinement_level=6, title='Depth-averaged M2 along-channel velocity (cosine)', clabel='Velocity [m/s]', figsize=(7,4), contourlines=True, cmap='RdBu', center_range=True)
# pp.plot_colormap(pp.u_DA(-1), refinement_level=6, title='Depth-averaged M2 along-channel velocity (sine)', clabel='Velocity [m/s]', figsize=(7,4), cmap='RdBu', center_range=True)
# pp.plot_colormap(pp.v_DA(1), refinement_level=10, title='Depth-averaged M2 cross-channel velocity (cosine)', clabel='Velocity [m/s]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=False)
# pp.plot_colormap(pp.v_DA(-1), refinement_level=10, title='Depth-averaged M2 cross-channel velocity (sine)', clabel='Velocity [m/s]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=False, show_mesh=True)

# pp.plot_colormap(pp.u_DA_abs(1), refinement_level=5, title='Depth-averaged M2 along-channel velocity (amp)', clabel='Velocity [m/s]', figsize=(7,4), center_range=False, cmap='viridis', contourlines=True, show_mesh=False)
# pp.plot_colormap(pp.v_DA_abs(1), refinement_level=8, title='Depth-averaged M2 cross-channel velocity (amp)', clabel='Velocity [m/s]', figsize=(7,4), center_range=False, cmap='viridis', contourlines=True, show_mesh=False)
# pp.plot_colormap(pp.v_DA_abs(2), refinement_level=8, title='Amplitude of depth-averaged M4 cross-channel velocity', clabel='Velocity [m/s]', figsize=(7,4), center_range=False, cmap='viridis', contourlines=True, show_mesh=False)
# pp.plot_colormap(pp.u_DA_abs(2), refinement_level=8, title='Amplitude of depth-averaged M4 along-channel velocity', clabel='Velocity [m/s]', figsize=(7,4), center_range=False, cmap='viridis', contourlines=True, show_mesh=False)

# pp.plot_colormap(pp.gamma_abs(1), refinement_level=5, title='M2 surface elevation (amp)', clabel='Velocity [m/s]', figsize=(7,4), center_range=False, cmap='viridis', contourlines=True, show_mesh=False)


# pp.plot_colormap(pp.u_DA_x(1), refinement_level=4, title='Depth-averaged x-gradient of M2 along-channel velocity (cosine)', clabel="Gradient [s^-1]", figsize=(7,4), center_range=True, cmap='RdBu', contourlines=False)
# pp.plot_colormap(pp.v_DA(-1), refinement_level=4, title='Depth-averaged M2 cross-channel velocity (sine)', clabel='Velocity [m/s]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=False, show_mesh=True)
# pp.plot_colormap(pp.gamma(1), refinement_level=10, title='Semidiurnal surface elevation (cosine)', clabel='Elevation [m]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=False)
# pp.plot_colormap(pp.gamma(-1), refinement_level=10, title='Semidiurnal surface elevation (sine)', clabel='Elevation [m]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=False)

# pp.plot_colormap(pp.gammax(1), refinement_level=4, title='Semidiurnal surface elevation X-gradient (cosine)', clabel='Gradient [-]', figsize=(7,4))
# pp.plot_colormap(pp.gammay(1), refinement_level=6, title='Semidiurnal surface elevation Y-gradient (cosine)', clabel='Gradient [-]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=True, show_mesh=False)
# 
# pp.plot_colormap(pp.gammax(-1), refinement_level=4, title='Semidiurnal surface elevation X-gradient (sine)', clabel='Gradient [-]', figsize=(7,4))
# pp.plot_colormap(pp.gammay(-1), refinement_level=6, title='Semidiurnal surface elevation Y-gradient (sine)', clabel='Gradient [-]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=True, show_mesh=False)


# pp.plot_colormap(pp.gamma(0), refinement_level=5, title='Residual surface elevation', clabel='Elevation [m]', figsize=(7,4), center_range=True, cmap='RdBu')
# pp.plot_colormap(pp.u_DA(0), refinement_level=5, title='Residual along-channel velocity', clabel='Velocity [m/s]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=True)
# pp.plot_colormap(pp.v_DA(0), refinement_level=5, title='Residual cross-channel velocity', clabel='Velocity [m/s]', figsize=(7,4), center_range=True, cmap='RdBu', contourlines=True)
# # 
# pp.plot_vertical_cross_section(lambda sig: pp.u(1, sig), 'Cosine M2 along-channel velocity', '', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.u_abs(2, sig), 'M4 along-channel velocity (amp)', 'Velocity [m/s]', p1, p2, 1000, 1000, vmin=0, cmap='viridis', figsize=(7,4))

# pp.plot_vertical_cross_section(lambda sig: pp.ux(1, sig), 'X-gradient of semidiurnal along-channel velocity (cosine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.uy(1, sig), 'Y-gradient of semidiurnal along-channel velocity (cosine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.u_abs(1, sig), 'Semidiurnal along-channel velocity (amp)', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=False, cmap='viridis', figsize=(7,4), vmin=0)
# pp.plot_vertical_cross_section(lambda sig: pp.ux(-1, sig), 'X-gradient of semidiurnal along-channel velocity (sine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.uy(-1, sig), 'Y-gradient of semidiurnal along-channel velocity (sine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))

# pp.plot_vertical_cross_section(lambda sig: pp.v_abs(1, sig), 'Semidiurnal cross-channel velocity (amp)', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=False, cmap='viridis', figsize=(7,4))

# pp.plot_vertical_cross_section(lambda sig: pp.u_abs(2, sig), 'Amplitude of M4 along-channel velocity', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=False, cmap='viridis', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.v_abs(2, sig), 'Amplitude of M4 cross-channel velocity', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=False, cmap='viridis', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.w_abs(2, sig), 'Amplitude of M4 vertical velocity (amp)', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=False, cmap='viridis', figsize=(7,4))

# pp.plot_vertical_cross_section(lambda sig: pp.vx(1, sig), 'X-gradient of semidiurnal cross-channel velocity (cosine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.vy(1, sig), 'Y-gradient of semidiurnal cross-channel velocity (cosine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.v(-1, sig), 'Semidiurnal cross-channel velocity (sine)', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.vx(-1, sig), 'X-gradient of semidiurnal cross-channel velocity (sine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.vy(-1, sig), 'Y-gradient of semidiurnal cross-channel velocity (sine)', 'Gradient', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))

# pp.plot_vertical_cross_section(lambda sig: pp.u(0, sig), 'Residual along-channel velocity', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))

# pp.plot_vertical_cross_section(lambda sig: pp.v(0, sig), 'Residual cross-channel velocity', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.v_abs(2, sig), 'M4 cross-channel velocity (amp)', 'Velocity [m/s]', p1, p2, 1000, 1000, vmin=0, cmap='viridis', figsize=(7,4))


# pp.plot_vertical_cross_section(lambda sig: pp.w_abs(1, sig), 'Semidiurnal vertical velocity (amp)', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=False, figsize=(7,4), vmin=0)
# pp.plot_vertical_cross_section(lambda sig: pp.w(0, sig), 'Residual vertical velocity', 'Velocity [m/s]', p1, p2, 1000, 1000, center_range=True, cmap='RdBu', figsize=(7,4))


# pp.plot_vertical_cross_section(lambda sig: pp.u_abs(1, sig), "Semidiurnal along-channel velocity amplitude", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.v_abs(1, sig), "Semidiurnal cross-channel velocity amplitude", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.w_abs(1, sig), "Semidiurnal vertical velocity amplitude", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4))
# pp.plot_vertical_cross_section(lambda sig: pp.ux(1, sig), "Semidiurnal along-channel velocity cosine gradient x ", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4), center_range=True, cmap='RdBu')
# pp.plot_vertical_cross_section(lambda sig: pp.uy(1, sig), "Semidiurnal along-channel velocity cosine gradient y", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4), center_range=True, cmap='RdBu')
# pp.plot_vertical_cross_section(lambda sig: pp.usig(1, sig), "Semidiurnal along-channel velocity sine gradient sig", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4), center_range=True, cmap='RdBu')
# pp.plot_vertical_cross_section(lambda sig: pp.ux(-1, sig), "Semidiurnal along-channel velocity sine gradient x ", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4), center_range=True, cmap='RdBu')
# pp.plot_vertical_cross_section(lambda sig: pp.uy(-1, sig), "Semidiurnal along-channel velocity sine gradient y", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4), center_range=True, cmap='RdBu')
# pp.plot_vertical_cross_section(lambda sig: pp.usig(-1, sig), "Semidiurnal along-channel velocity sine gradient sig", 'Velocity [m/s]', p1, p2, 500, 500, figsize=(7,4), center_range=True, cmap='RdBu')

# pp.plot_vertical_cross_section(lambda sig: lon_advec_forcing(sig) + lat_advec_forcing(sig) + vert_advec_forcing(sig), "Total advective forcing", '-', p1, p2, 500, 500, figsize=(7,4), center_range=True, cmap='RdBu')

# pp.plot_vertical_cross_section(lambda sig: pp.w_abs(2, sig), 'M4 vertical velocity (amp)', 'Velocity [m/s]', p1, p2, 1000, 1000, vmin=0, cmap='viridis', figsize=(7,4))

# pp.plot_cross_section_circulation(p1, p2, 1000, 1000, 100, constituent=0)
# pp.plot_cross_section_residual_forcing_mechanisms(p1, p2, 1000, 1000, component='v')

plt.show()