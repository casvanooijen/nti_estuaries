import numpy as np
import scipy.sparse as sp
import timeit
import ngsolve
import copy

from hydrodynamics import *
import define_weak_forms as weakforms


def solve(hydro: Hydrodynamics, max_iterations: int = 10, tolerance: float = 1e-9, linear_solver = 'pardiso', 
          continuation_parameters: dict = {'advection_epsilon': [1], 'Av': [1]}, stopcriterion = 'scaled_2norm'):

    """
    
    Compute the solution of the model given by a Hydrodynamics-object and add it as an attribute to the Hydrodynamics-object. Solution is computed using a Newton-Raphson
    method, combined with a continuation (homology) method to guide the Newton method towards the correct solution.
    
    Arguments:
    
        - hydro (Hydrodynamics):            object containing the (weak) model equations and options;
        - max_iterations (int):             maximum number of Newton iterations per continuation step;
        - tolerance (float):                if the stopping criterion is less than this value, the Newton method terminates and the procedure moves to the next continuation step;
        - linear_solver:                    choice of linear solver; options: 'pardiso', ..
        - continuation_parameters (dict):   dictionary with keys 'advection_epsilon' and 'Av', with values indicating what the default value of these parameters should be multiplied by in each continuation step;
        - stopcriterion:                    choice of stopping criterion; options: 'matrix_norm', 'scaled_2norm', 'relative_newtonstepsize';
    
    """

    # Quick argument check

    if len(continuation_parameters['advection_epsilon']) == len(continuation_parameters['Av']):
        num_continuation_steps = len(continuation_parameters['advection_epsilon'])
    else:
        raise ValueError(f"Length of both continuation parameter lists must be equal; now the lenghts are {len(continuation_parameters['advection_epsilon'])} and {len(continuation_parameters['Av'])}")
    
    # Report that solution procedure is about to start.

    print(f"Initiating solution procedure for hydrodynamics-model with {hydro.M} vertical components and {hydro.imax + 1} tidal constituents (including residual).\nIn total, there are {(2*hydro.M+1)*(2*hydro.imax+1)} equations. The total number of free degrees of freedom is {hydro.nfreedofs}.")

    # Set initial guess
    sol = ngsolve.GridFunction(hydro.femspace)
    sol.components[2*(hydro.M)*(2*hydro.imax+1)].Set(hydro.seaward_forcing.boundaryCFdict[0], ngsolve.BND)
    for q in range(1, hydro.imax + 1):
        sol.components[2*(hydro.M)*(2*hydro.imax+1) + q].Set(hydro.seaward_forcing.boundaryCFdict[-q], ngsolve.BND)
        sol.components[2*(hydro.M)*(2*hydro.imax+1) + hydro.imax + q].Set(hydro.seaward_forcing.boundaryCFdict[q], ngsolve.BND)

    hydro.solution_gf = sol
    

    # Save true values of advection_epsilon and Av before modifying them in the continuation (homology) method

    true_epsilon = copy.copy(hydro.model_options['advection_epsilon'])
    true_Av = copy.copy(hydro.constant_physical_parameters['Av'])

    for continuation_counter in range(num_continuation_steps):
        hydro.model_options['advection_epsilon'] = true_epsilon * continuation_parameters['advection_epsilon'][continuation_counter]
        hydro.constant_physical_parameters['Av'] = true_Av * continuation_parameters['Av'][continuation_counter]

        print(f"\nCONTINUATION STEP {continuation_counter}: Epsilon = {hydro.model_options['advection_epsilon']}, Av = {hydro.constant_physical_parameters['Av']}.\n")
        print("Setting up full weak form\n")

        hydro.setup_forms(skip_nonlinear = (hydro.model_options['advection_epsilon'] == 0))

        # Start the Newton method

        previous_iterate = copy.copy(hydro.solution_gf)

        for newton_counter in range(max_iterations):
            print(f"Newton-Raphson iteration {newton_counter}")
            hydro.restructure_solution() # restructure solution so that hydro.alpha_solution, hydro.beta_solution, and hydro.gamma_solution are specified.
            # Set-up weak forms for linearisation

            forms_start = timeit.default_timer()
            a = ngsolve.BilinearForm(hydro.femspace)
            weakforms.add_weak_form(a, hydro.model_options, hydro.alpha_trialfunctions, hydro.beta_trialfunctions, hydro.gamma_trialfunctions,
                                    hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions, hydro.M, hydro.imax,
                                    hydro.constant_physical_parameters, hydro.spatial_physical_parameters, hydro.vertical_basis, hydro.time_basis,
                                    hydro.riverine_forcing.normal_alpha, only_linear=True)
            if hydro.model_options['advection_epsilon'] != 0:
                weakforms.add_linearised_nonlinear_terms(a, hydro.model_options, hydro.alpha_trialfunctions, hydro.alpha_solution, hydro.beta_trialfunctions, hydro.beta_solution,
                                                         hydro.gamma_trialfunctions, hydro.gamma_solution, 
                                                         hydro.umom_testfunctions, hydro.vmom_testfunctions, hydro.DIC_testfunctions, hydro.M, hydro.imax,
                                                         hydro.constant_physical_parameters, hydro.spatial_physical_parameters, hydro.vertical_basis, hydro.time_basis,
                                                         hydro.riverine_forcing.normal_alpha)
            forms_time = timeit.default_timer() - forms_start
            print(f"    Weak form construction took {forms_time} seconds")

            # Assemble system matrix
            assembly_start = timeit.default_timer()
            a.Assemble()
            assembly_time = timeit.default_timer() - assembly_start
            print(f"    Assembly took {assembly_time} seconds")

            # Solve linearisation
            rhs = hydro.solution_gf.vec.CreateVector()
            hydro.total_bilinearform.Apply(hydro.solution_gf.vec, rhs)
            du = ngsolve.GridFunction(hydro.femspace)
            for i in range(hydro.femspace.dim):
                du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions

            inversion_start = timeit.default_timer()
            if linear_solver == 'pardiso':
                du.vec.data = a.mat.Inverse(freedofs=hydro.femspace.FreeDofs(), inverse='pardiso') * rhs
            else:
                raise ValueError(f"Linear solver '{linear_solver}' not known to the system.")
            inversion_time = timeit.default_timer() - inversion_start
            print(f"    Inversion took {inversion_time} seconds")

            hydro.solution_gf.vec.data = hydro.solution_gf.vec.data - du.vec.data

            # Compute stopping criterion
            residual = hydro.solution_gf.vec.CreateVector()
            apply_start = timeit.default_timer()
            hydro.total_bilinearform.Apply(hydro.solution_gf.vec, residual)
            apply_time = timeit.default_timer() - apply_start
            print(f"    Evaluating weak form at current Newton iterate took {apply_time} seconds.")

            homogenise_essential_Dofs(residual, hydro.femspace.FreeDofs())

            if stopcriterion == 'matrix_norm':
                stopcriterion_value = abs(ngsolve.InnerProduct(hydro.solution_gf.vec - previous_iterate.vec, residual))
            elif stopcriterion == 'scaled_2norm':
                stopcriterion_value = ngsolve.sqrt(ngsolve.InnerProduct(residual, residual) / hydro.nfreedofs)
            else:
                raise ValueError(f"Stopping criterion '{stopcriterion}' not known to the system.")

            print(f"    Stopping criterion value is {stopcriterion_value}\n")

            if stopcriterion_value < tolerance:
                print('Newton-Raphson method converged')
                break
            else:
                previous_iterate = copy.copy(hydro.solution_gf)

    print('\nSolution process complete.')


