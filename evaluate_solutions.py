import numpy as np
import matplotlib.pyplot as plt
import ngsolve
from hydrodynamics import Hydrodynamics
import mesh_functions
from TruncationBasis import TruncationBasis
from copy import copy
from minusonepower import minusonepower

# Construct vertical structure and vertical velocities at points and point ranges

def get_vertical_UVstructure_at_point(hydro: Hydrodynamics, p, num_vertical_gridpoints, constituent_index):
    sigma_range = np.linspace(-1, 0, num_vertical_gridpoints)

    if constituent_index == 0:
        u = np.zeros_like(sigma_range)
        v = np.zeros_like(sigma_range)
    elif constituent_index > 0:
        u_real = np.zeros_like(sigma_range)
        v_real = np.zeros_like(sigma_range)
        u_imag = np.zeros_like(sigma_range)
        v_imag = np.zeros_like(sigma_range)
    else:
        raise ValueError('Invalid index for tidal constituent. Please enter one of 0 (subtidal), 1 (M2), 2 (M4), ...')
    
    if constituent_index == 0:
        for m in range(hydro.M + 1):
            alpha_val = mesh_functions.evaluate_gridfunction_point(hydro.alpha_solution[m][0], hydro.mesh, p[0], p[1])
            beta_val = mesh_functions.evaluate_gridfunction_point(hydro.beta_solution[m][0], hydro.mesh, p[0], p[1])

            u += alpha_val * hydro.vertical_basis.evaluation_function(sigma_range, m)
            v += beta_val * hydro.vertical_basis.evaluation_function(sigma_range, m)
    elif constituent_index > 0:
        for m in range(hydro.M + 1):
            alpha_val_real = mesh_functions.evaluate_gridfunction_point(hydro.alpha_solution[m][constituent_index], hydro.mesh, p[0], p[1])
            alpha_val_imag = mesh_functions.evaluate_gridfunction_point(hydro.alpha_solution[m][-constituent_index], hydro.mesh, p[0], p[1])
            u_real += alpha_val_real * hydro.vertical_basis.evaluation_function(sigma_range, m)
            u_imag += alpha_val_imag * hydro.vertical_basis.evaluation_function(sigma_range, m)

            beta_val_real = mesh_functions.evaluate_gridfunction_point(hydro.beta_solution[m][constituent_index], hydro.mesh, p[0], p[1])
            beta_val_imag = mesh_functions.evaluate_gridfunction_point(hydro.beta_solution[m][-constituent_index], hydro.mesh, p[0], p[1])
            v_real += beta_val_real * hydro.vertical_basis.evaluation_function(sigma_range, m)
            v_imag += beta_val_imag * hydro.vertical_basis.evaluation_function(sigma_range, m)
    
    if constituent_index == 0:
        return u, v
    elif constituent_index > 0:
        return u_real, u_imag, v_real, v_imag
    

def get_vertical_UVstructure_at_pointrange(hydro: Hydrodynamics, coords, num_vertical_gridpoints, constituent_index):
    sigma_range = np.linspace(-1, 0, num_vertical_gridpoints)

    if constituent_index == 0:
        u = np.zeros((coords.shape[0], num_vertical_gridpoints))
        v = np.zeros((coords.shape[0], num_vertical_gridpoints))
    elif constituent_index > 0:
        u_real = np.zeros((coords.shape[0], num_vertical_gridpoints))
        v_real = np.zeros((coords.shape[0], num_vertical_gridpoints))
        u_imag = np.zeros((coords.shape[0], num_vertical_gridpoints))
        v_imag = np.zeros((coords.shape[0], num_vertical_gridpoints))
    else:
        raise ValueError('Invalid index for tidal constituent. Please enter one of 0 (subtidal), 1 (M2), 2 (M4), ...')
    if constituent_index == 0:
        for m in range(hydro.M + 1):
        
            alpha_val = mesh_functions.evaluate_gridfunction_range(hydro.alpha_solution[m][0], hydro.mesh, coords)
            beta_val = mesh_functions.evaluate_gridfunction_point(hydro.beta_solution[m][0], hydro.mesh, coords)

            u += alpha_val * hydro.vertical_basis.evaluation_function(sigma_range, m)
            v += beta_val * hydro.vertical_basis.evaluation_function(sigma_range, m)
    elif constituent_index > 0:
        for m in range(hydro.M + 1):
            alpha_val_real = mesh_functions.evaluate_gridfunction_range(hydro.alpha_solution[m][constituent_index], hydro.mesh, coords)
            alpha_val_imag = mesh_functions.evaluate_gridfunction_range(hydro.alpha_solution[m][-constituent_index], hydro.mesh, coords)
            u_real += alpha_val_real * hydro.vertical_basis.evaluation_function(sigma_range, m)
            u_imag += alpha_val_imag * hydro.vertical_basis.evaluation_function(sigma_range, m)

            beta_val_real = mesh_functions.evaluate_gridfunction_range(hydro.beta_solution[m][constituent_index], hydro.mesh, coords)
            beta_val_imag = mesh_functions.evaluate_gridfunction_range(hydro.beta_solution[m][-constituent_index], hydro.mesh, coords)
            v_real += beta_val_real * hydro.vertical_basis.evaluation_function(sigma_range, m)
            v_imag += beta_val_imag * hydro.vertical_basis.evaluation_function(sigma_range, m)
    
    if constituent_index == 0:
        return u, v
    elif constituent_index > 0:
        return u_real, u_imag, v_real, v_imag
    

def construct_W_at_point(hydro: Hydrodynamics, p, num_vertical_gridpoints, constituent_index, u=None, v=None, u_real=None, v_real=None, u_imag=None, v_imag=None):
    sigma_range = np.linspace(-1, 0, num_vertical_gridpoints)

    if constituent_index == 0:
        omegatilde = np.zeros_like(sigma_range)
    elif constituent_index > 0:
        omegatilde_real = np.zeros_like(sigma_range)
        omegatilde_imag = np.zeros_like(sigma_range)
    else:
        raise ValueError('Invalid index for tidal constituent. Please enter one of 0 (subtidal), 1 (M2), 2 (M4), ...')
    
    Hval = mesh_functions.evaluate_coefficient_function_point(hydro.spatial_physical_parameters['H'].cf, hydro.mesh, p[0], p[1])
    Hxval = mesh_functions.evaluate_coefficient_function_point(hydro.spatial_physical_parameters['H'].gradient_cf[0], hydro.mesh, p[0], p[1])
    Hyval = mesh_functions.evaluate_coefficient_function_point(hydro.spatial_physical_parameters['H'].gradient_cf[1], hydro.mesh, p[0], p[1])

    if constituent_index == 0:
        for m in range(hydro.M + 1):
            # calculate div(H*alpha) 
            F_cf = hydro.spatial_physical_parameters['H'].cf * (ngsolve.grad(hydro.alpha_solution[m][0])[0] + 
                                                                ngsolve.grad(hydro.beta_solution[m][0])[1]) + \
                   hydro.alpha_solution[m][0] * hydro.spatial_physical_parameters['H'].gradient_cf[0] + \
                   hydro.beta_solution[m][0] * hydro.spatial_physical_parameters['H'].gradient_cf[1]
            
            F_val = mesh_functions.evaluate_coefficient_function_point(F_cf, hydro.mesh, p[0], p[1])

            omegatilde += -1/Hval * 1/((m+0.5)*np.pi) * F_val * (np.sin((m+0.5)*np.pi*sigma_range) + minusonepower(m) * np.ones_like(sigma_range))
        
        w = Hval*omegatilde + sigma_range*Hxval*u + sigma_range*Hyval*v

        return w

    elif constituent_index > 0:
        for m in range(hydro.M + 1):
            F_cf_real = hydro.spatial_physical_parameters['H'].cf * (ngsolve.grad(hydro.alpha_solution[m][constituent_index])[0] + 
                                                                     ngsolve.grad(hydro.beta_solution[m][constituent_index])[1]) + \
                        hydro.alpha_solution[m][constituent_index] * hydro.spatial_physical_parameters['H'].gradient_cf[0] + \
                        hydro.beta_solution[m][constituent_index] * hydro.spatial_physical_parameters['H'].gradient_cf[1]
            F_val_real = mesh_functions.evaluate_coefficient_function_point(F_cf_real, hydro.mesh, p[0], p[1])

            F_cf_imag = hydro.spatial_physical_parameters['H'].cf * (ngsolve.grad(hydro.alpha_solution[m][-constituent_index])[0] + 
                                                                     ngsolve.grad(hydro.beta_solution[m][-constituent_index])[1]) + \
                        hydro.alpha_solution[m][-constituent_index] * hydro.spatial_physical_parameters['H'].gradient_cf[0] + \
                        hydro.beta_solution[m][-constituent_index] * hydro.spatial_physical_parameters['H'].gradient_cf[1]

            F_val_imag = mesh_functions.evaluate_coefficient_function_point(F_cf_imag, hydro.mesh, p[0], p[1])

            omegatilde_real += -1/Hval * 1/((m+0.5)*np.pi) * F_val_real * (np.sin((m+0.5)*np.pi*sigma_range) + minusonepower(m) * np.ones_like(sigma_range))
            omegatilde_imag += -1/Hval * 1/((m+0.5)*np.pi) * F_val_imag * (np.sin((m+0.5)*np.pi*sigma_range) + minusonepower(m) * np.ones_like(sigma_range))

        w_real = Hval*omegatilde_real + sigma_range*Hxval*u_real + sigma_range*Hyval*v_real
        w_imag = Hval*omegatilde_imag + sigma_range*Hxval*u_imag + sigma_range*Hyval*v_imag

        return w_real, w_imag
    

## Evaluate on certain ranges ##
    
def evaluate_on_cross_section(hydro: Hydrodynamics, quantity, x_range, y_range, sigma_range):
    num_horizontal_points = x_range.shape[0]
    num_vertical_points = sigma_range.shape[0]

    Q = np.zeros((num_horizontal_points, num_vertical_points))
    for i in range(num_horizontal_points):
        Q[i, :] = mesh_functions.evaluate_gridfunction_range(quantity, hydro.mesh, x_range, y_range)
    
    return Q


## Create functions that return coefficient functions/grid functions for a given value of sigma ##

def construct_vertical_structure(hydro: Hydrodynamics):
    """Returns dictionary of sigma-dependent flow velocities u, v, w, with tidal constituents as keys (positive=real part, negative=imaginary part)."""

    u = dict()
    v = dict()
    w = dict()

    H = hydro.spatial_physical_parameters['H'].cf
    Hx = hydro.spatial_physical_parameters['H'].gradient_cf[0]
    Hy = hydro.spatial_physical_parameters['H'].gradient_cf[1]

    # subtidal flow
    def u_M0(sigma):
        return sum([hydro.alpha_solution[m][0] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M + 1)])
    
    def v_M0(sigma):
        return sum([hydro.beta_solution[m][0] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M + 1)])
    
    u[0] = u_M0
    v[0] = v_M0

    F_list = [H * (ngsolve.grad(hydro.alpha_solution[m][0])[0] + ngsolve.grad(hydro.beta_solution[m][0])[1]) + \
              hydro.alpha_solution[m][0] * Hx + hydro.beta_solution[m][0] * Hy for m in range(hydro.M + 1)]

    def w_M0(sigma):
        return -sum([1/((m+0.5)*np.pi) * F_list[m] * (np.sin((m+0.5)*np.pi*sigma) + minusonepower(m) * np.ones_like(sigma)) for m in range(hydro.M + 1)]) + \
               sigma * u_M0(sigma) * Hx + sigma * v_M0(sigma) * Hy
    
    w[0] = w_M0

    for q in range(1, hydro.qmax + 1):
        
        def u_tidal_real(sigma):
            return sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M + 1)])
        
        def v_tidal_real(sigma):
            return sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M + 1)])
    
        F_list_real = [H * (ngsolve.grad(hydro.alpha_solution[m][q])[0] + ngsolve.grad(hydro.beta_solution[m][q])[1]) + \
                       hydro.alpha_solution[m][q] * Hx + hydro.beta_solution[m][q] * Hy for m in range(hydro.M + 1)]

        def w_tidal_real(sigma):
            return -sum([1/((m+0.5)*np.pi) * F_list_real[m] * (np.sin((m+0.5)*np.pi*sigma) + minusonepower(m) * np.ones_like(sigma)) for m in range(hydro.M + 1)]) + \
                   sigma * u_tidal_real(sigma) * Hx + sigma * v_tidal_real(sigma) * Hy
        
        u[q] = copy(u_tidal_real)
        v[q] = copy(v_tidal_real)
        w[q] = copy(w_tidal_real)

        def u_tidal_imag(sigma):
            return sum([hydro.alpha_solution[m][-q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M + 1)])
        
        def v_tidal_imag(sigma):
            return sum([hydro.beta_solution[m][-q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M + 1)])
    
        F_list_imag = [H * (ngsolve.grad(hydro.alpha_solution[m][-q])[0] + ngsolve.grad(hydro.beta_solution[m][-q])[1]) + \
                       hydro.alpha_solution[m][-q] * Hx + hydro.beta_solution[m][-q] * Hy for m in range(hydro.M + 1)]

        def w_tidal_imag(sigma):
            return -sum([1/((m+0.5)*np.pi) * F_list_imag[m] * (np.sin((m+0.5)*np.pi*sigma) + minusonepower(m) * np.ones_like(sigma)) for m in range(hydro.M + 1)]) + \
                   sigma * u_tidal_imag(sigma) * Hx + sigma * v_tidal_imag(sigma) * Hy
        
        u[-q] = copy(u_tidal_imag)
        v[-q] = copy(v_tidal_imag)
        w[-q] = copy(w_tidal_imag)

    return u, v, w


def evaluate_vertical_structure_at_point(hydro, quantity_function, p, num_vertical_points):
    sigma_range = np.linspace(-1, 0, num_vertical_points)
    Q = np.zeros_like(sigma_range)

    for i in range(num_vertical_points):
        Q[i] = mesh_functions.evaluate_coefficient_function_point(quantity_function(sigma_range[i]), hydro.mesh, p[0], p[1])

    return Q


def evaluate_vertical_structure_at_cross_section(hydro, quantity_function, p1, p2, num_horizontal_points, num_vertical_points):
    sigma_range = np.linspace(-1, 0, num_vertical_points)
    x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
    y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

    Q = np.zeros((num_horizontal_points, num_vertical_points))
    
    for i in range(num_vertical_points):
        Q[i, :] = mesh_functions.evaluate_gridfunction_range(quantity_function(sigma_range[i]), hydro.mesh, x_range, y_range)

    return Q