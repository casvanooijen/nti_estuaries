import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import timeit
import ngsolve
import copy
import matplotlib.pyplot as plt

from hydrodynamics import *
import define_weak_forms as weakforms
from mesh_functions import plot_CF_colormap


# Main function


def solve(hydro: Hydrodynamics, max_iterations: int = 10, tolerance: float = 1e-9, linear_solver = 'pardiso', 
          continuation_parameters: dict = {'advection_epsilon': [1], 'Av': [1]}, stopcriterion = 'scaled_2norm',
          reduced_hydro: Hydrodynamics=None, plot_intermediate_results='none', parallel=True):

    """
    
    Compute the solution of the model given by a Hydrodynamics-object and add it as an attribute to the Hydrodynamics-object. Solution is computed using a Newton-Raphson
    method, combined with a continuation (homology) method to guide the Newton method towards the correct solution.
    
    Arguments:
    
        - hydro (Hydrodynamics):            object containing the (weak) model equations and options;
        - max_iterations (int):             maximum number of Newton iterations per continuation step;
        - tolerance (float):                if the stopping criterion is less than this value, the Newton method terminates and the procedure moves to the next continuation step;
        - linear_solver:                    choice of linear solver; options: 'pardiso', 'scipy_direct', 'bicgstab'
        - continuation_parameters (dict):   dictionary with keys 'advection_epsilon' and 'Av', with values indicating what the default value of these parameters should be multiplied by in each continuation step;
        - stopcriterion:                    choice of stopping criterion; options: 'matrix_norm', 'scaled_2norm', 'relative_newtonstepsize';
        - reduced_hydro (Hydrodynamics):    reduced version of the hydrodynamics object that can be used as a preconditioner for iterative solvers; Note: M and imax must be identical
        - plot_intermediate_results:        indicates whether intermediate results should be plotted and saved; options: 'none' (default), 'all' and 'overview'.
        - parallel:                         flag indicating whether time-costly operations should be performed in parallel (see https://docu.ngsolve.org/latest/how_to/howto_parallel.html)
    
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
        # tidal waterlevel
    sol.components[2*(hydro.M)*(2*hydro.imax+1)].Set(hydro.seaward_forcing.boundaryCFdict[0], ngsolve.BND)
    for q in range(1, hydro.imax + 1):
        sol.components[2*(hydro.M)*(2*hydro.imax+1) + q].Set(hydro.seaward_forcing.boundaryCFdict[-q], ngsolve.BND)
        sol.components[2*(hydro.M)*(2*hydro.imax+1) + hydro.imax + q].Set(hydro.seaward_forcing.boundaryCFdict[q], ngsolve.BND)
        # river discharge
    if hydro.model_options['horizontal_diffusion']:
        for m in range(hydro.M):
            sol.components[m * (2*hydro.imax + 1)].Set(hydro.riverine_forcing.normal_alpha_boundaryCF[m][0], ngsolve.BND)
            
    hydro.solution_gf = sol
    

    # Save true values of advection_epsilon and Av before modifying them in the continuation (homology) method

    true_epsilon = copy.copy(hydro.model_options['advection_epsilon'])
    true_Av = copy.copy(hydro.constant_physical_parameters['Av'])

    for continuation_counter in range(num_continuation_steps):
        hydro.model_options['advection_epsilon'] = true_epsilon * continuation_parameters['advection_epsilon'][continuation_counter]
        hydro.constant_physical_parameters['Av'] = true_Av * continuation_parameters['Av'][continuation_counter]

        print(f"\nCONTINUATION STEP {continuation_counter}: Epsilon = {hydro.model_options['advection_epsilon']}, Av = {hydro.constant_physical_parameters['Av']}.\n")
        print("Setting up full weak form\n")

        if parallel:
            with ngsolve.TaskManager():
                hydro.setup_forms(skip_nonlinear = (hydro.model_options['advection_epsilon'] == 0))
        else:
            hydro.setup_forms(skip_nonlinear = (hydro.model_options['advection_epsilon'] == 0))


        # Start the Newton method

        previous_iterate = copy.copy(hydro.solution_gf)

        for newton_counter in range(max_iterations):
            print(f"Newton-Raphson iteration {newton_counter}")
            hydro.restructure_solution() # restructure solution so that hydro.alpha_solution, hydro.beta_solution, and hydro.gamma_solution are specified.
            # Set-up weak forms for linearisation

            forms_start = timeit.default_timer()
            if parallel:
                with ngsolve.TaskManager():
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
            else:
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

            if reduced_hydro is not None:
                forms_start_reduced = timeit.default_timer()
                a_reduced = ngsolve.BilinearForm(reduced_hydro.femspace)
                weakforms.add_weak_form(a_reduced, reduced_hydro.model_options, reduced_hydro.alpha_trialfunctions, reduced_hydro.beta_trialfunctions, reduced_hydro.gamma_trialfunctions,
                                        reduced_hydro.umom_testfunctions, reduced_hydro.vmom_testfunctions, reduced_hydro.DIC_testfunctions, hydro.M, hydro.imax,
                                        reduced_hydro.constant_physical_parameters, reduced_hydro.spatial_physical_parameters, hydro.vertical_basis, hydro.time_basis,
                                        reduced_hydro.riverine_forcing.normal_alpha, only_linear=True)
                if reduced_hydro.model_options['advection_epsilon'] != 0:
                    weakforms.add_linearised_nonlinear_terms(a, reduced_hydro.model_options, reduced_hydro.alpha_trialfunctions, hydro.alpha_solution, reduced_hydro.beta_trialfunctions, hydro.beta_solution,
                                                             reduced_hydro.gamma_trialfunctions, hydro.gamma_solution, 
                                                             reduced_hydro.umom_testfunctions, reduced_hydro.vmom_testfunctions, reduced_hydro.DIC_testfunctions, hydro.M, hydro.imax,
                                                             reduced_hydro.constant_physical_parameters, reduced_hydro.spatial_physical_parameters, hydro.vertical_basis, hydro.time_basis,
                                                             reduced_hydro.riverine_forcing.normal_alpha)
                forms_time_reduced = timeit.default_timer() - forms_start_reduced
                print(f"    Weak form construction for reduced model preconditioner took {forms_time} seconds")

            # Assemble system matrix
            assembly_start = timeit.default_timer()
            if parallel:
                with ngsolve.TaskManager():
                    a.Assemble()
            else:
                a.Assemble()
            assembly_time = timeit.default_timer() - assembly_start
            print(f"    Assembly took {assembly_time} seconds")

            if reduced_hydro is not None:
                assembly_start_reduced = timeit.default_timer()
                a_reduced.Assemble()
                assembly_time_reduced = timeit.default_timer() - assembly_start_reduced
                print(f"    Assembly for reduced model preconditioner took {assembly_time_reduced} seconds")

            # Solve linearisation
            rhs = hydro.solution_gf.vec.CreateVector()
            hydro.total_bilinearform.Apply(hydro.solution_gf.vec, rhs)
            du = ngsolve.GridFunction(hydro.femspace)
            for i in range(hydro.femspace.dim):
                du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions

            inversion_start = timeit.default_timer()
            if linear_solver == 'pardiso':
                if parallel:
                    with ngsolve.TaskManager():
                        du.vec.data = a.mat.Inverse(freedofs=hydro.femspace.FreeDofs(), inverse='pardiso') * rhs
                else:
                    du.vec.data = a.mat.Inverse(freedofs=hydro.femspace.FreeDofs(), inverse='pardiso') * rhs
            elif linear_solver == 'scipy_direct':
                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                rhs_arr = rhs.FV().NumPy()[freedof_list]

                sol = spsolve(mat, rhs_arr)
                du.vec.FV().NumPy()[freedof_list] = sol
            elif linear_solver == 'bicgstab':
                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                if reduced_hydro is not None:
                    freedof_list_reduced = get_freedof_list(reduced_hydro.femspace.FreeDofs())
                    reduced_mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a_reduced.mat), freedof_list_reduced)
                rhs_arr = rhs.FV().NumPy()[freedof_list]

            
                if newton_counter == 0 and reduced_hydro is not None:
                    rhs_arr_reduced = project_arr_to_other_basis(rhs_arr, hydro.femspace, reduced_hydro.femspace, hydro.num_equations)
                    reduced_initial_guess = spsolve(reduced_mat, rhs_arr_reduced)
                    initial_guess = project_arr_to_other_basis(reduced_initial_guess, reduced_hydro.femspace, hydro.femspace, hydro.num_equations)
                else:
                    initial_guess = hydro.solution_gf.vec.FV().NumPy()[freedof_list]

                if reduced_hydro is not None:
                    sol, exitcode = bicgstab(mat, rhs_arr, initial_guess, reduced_A = reduced_mat, reduced_fespace = reduced_hydro.femspace, full_fespace = hydro.femspace, num_equations=hydro.num_equations)
                else:
                    sol, exitcode = bicgstab(mat, rhs_arr, initial_guess)

                if exitcode == 0:
                    print(f"    Bi-CGSTAB did not converge in 500 iterations")
                else:
                    print(f"    Bi-CGSTAB converged in {exitcode} iterations")

                du.vec.FV().NumPy()[freedof_list] = sol
            else:
                raise ValueError(f"Linear solver '{linear_solver}' not known to the system.")
            inversion_time = timeit.default_timer() - inversion_start
            print(f"    Inversion took {inversion_time} seconds")

            hydro.solution_gf.vec.data = hydro.solution_gf.vec.data - du.vec.data

            # Compute stopping criterion
            residual = hydro.solution_gf.vec.CreateVector()
            apply_start = timeit.default_timer()
            if parallel:
                with ngsolve.TaskManager():
                    hydro.total_bilinearform.Apply(hydro.solution_gf.vec, residual)
            else:
                hydro.total_bilinearform.Apply(hydro.solution_gf.vec, residual)
            
            apply_time = timeit.default_timer() - apply_start
            print(f"    Evaluating weak form at current Newton iterate took {apply_time} seconds.")

            homogenise_essential_Dofs(residual, hydro.femspace.FreeDofs())

            if stopcriterion == 'matrix_norm':
                stopcriterion_value = abs(ngsolve.InnerProduct(hydro.solution_gf.vec - previous_iterate.vec, residual))
            elif stopcriterion == 'scaled_2norm':
                stopcriterion_value = ngsolve.sqrt(ngsolve.InnerProduct(residual, residual) / hydro.nfreedofs)
            elif stopcriterion == 'relative_newtonstepsize':
                stopcriterion_value = ngsolve.sqrt(ngsolve.InnerProduct(hydro.solution_gf.vec - previous_iterate.vec, hydro.solution_gf.vec - previous_iterate.vec)) / ngsolve.sqrt(ngsolve.InnerProduct(hydro.solution_gf.vec, hydro.solution_gf.vec))
            else:
                raise ValueError(f"Stopping criterion '{stopcriterion}' not known to the system.")

            print(f"    Stopping criterion value is {stopcriterion_value}\n")

            if plot_intermediate_results == 'all':
                # plotting to test where convergence goes wrong
                for m in range(hydro.M):
                    for i in range(-hydro.imax, hydro.imax+1):
                        plot_CF_colormap(hydro.alpha_solution[m][i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'alpha_({m},{i})', save = f"iteration{newton_counter}_alpha({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.alpha_solution[m][i])[0], hydro.mesh, refinement_level=3, show_mesh=True, title=f'alphax_({m},{i})', save = f"iteration{newton_counter}_alphax({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.alpha_solution[m][i])[1], hydro.mesh, refinement_level=3, show_mesh=True, title=f'alphay_({m},{i})', save = f"iteration{newton_counter}_alphay({m},{i})")
                        plot_CF_colormap(hydro.beta_solution[m][i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'beta_({m},{i})', save = f"iteration{newton_counter}_beta({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.beta_solution[m][i])[0], hydro.mesh, refinement_level=3, show_mesh=True, title=f'betax_({m},{i})', save = f"iteration{newton_counter}_betax({m},{i})")
                        plot_CF_colormap(ngsolve.grad(hydro.beta_solution[m][i])[1], hydro.mesh, refinement_level=3, show_mesh=True, title=f'betay_({m},{i})', save = f"iteration{newton_counter}_betay({m},{i})")

                for i in range(-hydro.imax, hydro.imax+1):
                    plot_CF_colormap(hydro.gamma_solution[i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'gamma_({i})', save = f"iteration{newton_counter}_gamma({i})")
                    u_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.alpha_solution[m][i] for m in range(hydro.M)])
                    plot_CF_colormap(u_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DA_({i})', save = f"iteration{newton_counter}_uDA({i})")
                    v_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.beta_solution[m][i] for m in range(hydro.M)])
                    plot_CF_colormap(v_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DA_({i})', save = f"iteration{newton_counter}_vDA({i})")


                plt.close(fig = 'all')
            elif plot_intermediate_results == 'overview':
                for i in range(-hydro.imax, hydro.imax+1):
                    plot_CF_colormap(hydro.gamma_solution[i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'gamma_({i})', save = f"iteration{newton_counter}_gamma({i})")
                    u_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.alpha_solution[m][i] for m in range(hydro.M)])
                    u_DAx = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.alpha_solution[m][i])[0] for m in range(hydro.M)])
                    u_DAy = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.alpha_solution[m][i])[1] for m in range(hydro.M)])
                    plot_CF_colormap(u_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DA_({i})', save = f"iteration{newton_counter}_uDA({i})")
                    plot_CF_colormap(u_DAx, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DAx_({i})', save = f"iteration{newton_counter}_uDAx({i})")
                    plot_CF_colormap(u_DAy, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DAy_({i})', save = f"iteration{newton_counter}_uDAy({i})")

                    v_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.beta_solution[m][i] for m in range(hydro.M)])
                    v_DAx = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.beta_solution[m][i])[0] for m in range(hydro.M)])
                    v_DAy = sum([hydro.vertical_basis.tensor_dict['G4'](m) * ngsolve.grad(hydro.beta_solution[m][i])[1] for m in range(hydro.M)])
                    plot_CF_colormap(v_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DA_({i})', save = f"iteration{newton_counter}_vDA({i})")
                    plot_CF_colormap(v_DAx, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DAx_({i})', save = f"iteration{newton_counter}_vDAx({i})")
                    plot_CF_colormap(v_DAy, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DAy_({i})', save = f"iteration{newton_counter}_vDAy({i})")

            if stopcriterion_value < tolerance:
                print('Newton-Raphson method converged')
                break
            else:
                previous_iterate = copy.copy(hydro.solution_gf)

    print('\nSolution process complete.')


# Utility functions for sparse matrices


def get_freedof_list(freedof_bitarray):
    """Constructs a list containing the indices of the free degrees of freedom, generated from a bitarray from ngsolve.
    
    Arguments:
    
        - freedof_bitarray:     bitarray indicating which degree of freedom is free.
    """
    freedof_list = []
    for i, isFree in enumerate(freedof_bitarray):
        if isFree:
            freedof_list.append(i)
    return freedof_list


def basematrix_to_csr_matrix(mat: ngsolve.BaseMatrix):
    """Converts an ngsolve BaseMatrix to a scipy sparse matrix in CSR-format. Returns the CSR-matrix
    
    Arguments:

    - mat:      to-be-converted BaseMatrix      
        
    """
    rows, cols, vals = mat.COO()
    return sp.csr_matrix((vals, (rows, cols)))


def remove_fixeddofs_from_csr(mat, freedof_list):
    """Removes all of the fixed degrees of freedom from a scipy sparse matrix in CSR-format.
    
    Arguments:

        - mat:              matrix to be sliced;
        - freedof_list:     list of degrees of freedom to be kept; obtainable via `get_freedof_list`.

    """

    mat = mat[freedof_list, :]
    mat = mat[:, freedof_list]
    return mat


# Properties of matrices

def is_symmetric(mat: sp.csr_matrix, tol=1e-12):
    """Returns True if a sparse matrix (CSR) is symmetric within a certain (absolute) tolerance.
    
    Arguments:

    - mat (sp.csr_matrix):      sparse matrix to be checked;
    - tol (float):              if elements are further apart than this number, the function returns False.
    
    """
    diff = mat - mat.transpose()
    return not np.any(np.absolute(diff.data) >= tol * np.ones_like(diff.data))


def get_eigenvalue(mat, shift_inverse=False, maxits = 100, tol=1e-9):
    """Computes the largest eigenvalue of a matrix (sparse or otherwise) using the power method. If shift_inverse is True, then the method computes the smallest eigenvalue using
    the shift-inverse version of the power method.
    
    Arguments:

        - mat:                  matrix for which the eigenvalue is computed;
        - shift_inverse:        if True, uses shift inverse version of power method to compute the smallest eigenvalue.
        - maxits:               maximum number of iterations
    
    """

    previous_vec = np.random.randn(mat.shape[0]) # starting vector
    previous_eig = 0

    if not shift_inverse:
        for i in range(maxits):
            new_vec = mat @ previous_vec
            new_eig = np.inner(np.conj(previous_vec), new_vec)
            previous_vec = new_vec / np.linalg.norm(new_vec, ord=2)

            stopvalue = abs(new_eig - previous_eig) / abs(new_eig)
            if stopvalue < tol:
                break

            previous_eig = new_eig

            if i == maxits - 1:
                print('Method did not converge')
    else:
        for i in range(maxits):
            new_vec = spsolve(mat, previous_vec)
            new_eig = np.inner(np.conj(previous_vec), new_vec)

            previous_vec = new_vec / np.linalg.norm(new_vec, ord=2)

            stopvalue = abs(new_eig - previous_eig) / abs(new_eig)
            if stopvalue < tol:
                break

            previous_eig = new_eig

            if i == maxits - 1:
                print('Method did not converge')

    if shift_inverse:
        return 1 / new_eig
    else:
        return new_eig


def get_condition_number(mat, maxits = 100, tol=1e-9):
    """Computes 2-condition number of a sparse matrix by approximating the largest and smallest (in modulus) eigenvalues.
    
    Arguments:

    - mat:              sparse matrix;
    - maxits:           maximum number of iterations used in the power method;
    - tol:              tolerance used in the power method.

    """
    large_eig = abs(get_eigenvalue(mat, shift_inverse=False, maxits=maxits, tol=tol))
    small_eig = abs(get_eigenvalue(mat, shift_inverse=True, maxits=maxits, tol=tol))
    return large_eig / small_eig

# Linear solvers


def bicgstab(A, f, u0, tol=1e-12, maxits = 500, reduced_A=None, reduced_fespace=None, full_fespace=None, num_equations=None):
    """Carries out a Bi-CGSTAB solver based on the pseudocode in Van der Vorst (1992). This function has the option for a
    reduced basis preconditioner if reduced_A and transition_mass_matrix are specified. Returns the solution and an exitcode
    indicating how many iterations it took for convergence. If exitcode=0 is returned, then the method did not converge.
    This implementation is heavily based on the scipy-implementation that can be found on https://github.com/scipy/scipy/blob/main/scipy/sparse/linalg/_isolve/iterative.py.
    The function is also included in this file, so that the reduced-basis preconditioner can be used.
    
    Arguments:
    
    - A:                        system matrix;
    - f:                        right-hand side;
    - u0:                       initial guess;
    - tol:                      tolerance for stopping criterion;
    - reduced_A:                reduced basis-version of the system matrix. If this is an ngsolve.BaseMatrix, solves using this matrix are performed with PARDISO; otherwise with sp.spsolve (UMFPACK)
    - reduced_fespace:          finite element space containing the reduced basis;
    - full_fespace:             finite element space containing the full basis;
    - num_equations:            number of equations the system of two-dimensional PDEs consists of; given by hydro.num_equations
    
    """
    # initialising parameters
    r = f - A @ u0
    shadow_r0 = np.copy(r)

    previous_rho = 1
    alpha = 1
    omega = 1

    v = np.zeros_like(u0)
    p = np.zeros_like(u0)

    solution = u0[:]
    f_norm = np.linalg.norm(f, 2) # precompute this so this need not happen every iteration

    for i in range(maxits):
        rho = np.inner(shadow_r0, r)

        beta = (rho / previous_rho) * (alpha / omega)

        p -= omega * v
        p *= beta
        p += r

        # preconditioner
        if reduced_A is not None:
            reduced_p = project_arr_to_other_basis(p, full_fespace, reduced_fespace, num_equations)
            reduced_preconditioned_p = spsolve(reduced_A, reduced_p)
            preconditioned_p = project_arr_to_other_basis(reduced_preconditioned_p, reduced_fespace, full_fespace, num_equations)
        else:
            preconditioned_p = np.copy(p)

        v = A @ preconditioned_p
        alpha = rho / np.inner(shadow_r0, v)
        s = r - alpha * v

        # preconditioner
        if reduced_A is not None:
            reduced_s = project_arr_to_other_basis(s, full_fespace, reduced_fespace, num_equations)
            reduced_z = spsolve(reduced_A, reduced_s)
            z = project_arr_to_other_basis(reduced_z, reduced_fespace, full_fespace, num_equations)
        else:
            z = np.copy(s)

        t = A @ z
        omega = np.inner(t, s) / np.inner(t, t)

        solution += alpha * p + omega * z
        r = s - omega * t
        
        if np.linalg.norm(r, 2) / f_norm < tol:
            return solution, i+1 # return the solution and how many iterations it took for convergence

        previous_rho = np.copy(rho)

    return solution, 0 # return the solution after the final iteration, but with a 0 indicating non-convergence


# Preconditioning

def construct_mass_matrix(fespace: ngsolve.comp.FESpace):
    u, v = fespace.TnT()
    a = ngsolve.BilinearForm(fespace)
    a += u * v * ngsolve.dx
    a.Assemble()
    return basematrix_to_csr_matrix(a.mat)


def construct_transition_mass_matrix(full_fespace: ngsolve.comp.FESpace, reduced_fespace: ngsolve.comp.FESpace):
    """Returns the transition mass matrix for transitioning from the full basis to a reduced basis.
    
    Arguments:
    
    - full_fespace:             finite element space for the full (larger) basis;
    - reduced_fespace:          finite element space for the reduced basis;
    """
    
    M = np.empty((reduced_fespace.ndof, full_fespace.ndof))

    for k in range(reduced_fespace.ndof):
        reduced_gf = ngsolve.GridFunction(reduced_fespace)
        reduced_gf.vec[k] = 1
        for n in range(full_fespace.ndof):
            full_gf = ngsolve.GridFunction(full_fespace)
            full_gf.vec[n] = 1
            M[k, n] = ngsolve.Integrate(reduced_gf * full_gf, full_fespace.mesh)
    return M


def project_arr_to_other_basis(arr, fespace1, fespace2, num_equations):
    """Projects a numpy coefficient array to another finite element basis. Returns projected array
    
    Arguments:

    - arr:              numpy array to project
    - fespace1:         finite element space corresponding to arr
    - fespace2:         finite element space corresponding to the projected arr 
    - num_equations:    how many equations the hydrodynamics-object solves (given by hydro.num_equations)
    
    """

    freedoflist1 = get_freedof_list(fespace1.FreeDofs())
    freedoflist2 = get_freedof_list(fespace2.FreeDofs())

    gridfunction_1 = ngsolve.GridFunction(fespace1)
    gridfunction_1.vec.FV().NumPy()[freedoflist1] = arr

    gridfunction_2 = ngsolve.GridFunction(fespace2)
    for i in range(num_equations):
        gridfunction_2.components[i].Set(gridfunction_1.components[i])

    return gridfunction_2.vec.FV().NumPy()[freedoflist2]



if __name__ == '__main__':
    
    from netgen.geom2d import unit_square
    mesh = ngsolve.Mesh(unit_square.GenerateMesh(maxh=0.3))
    reduced_mesh = ngsolve.Mesh(unit_square.GenerateMesh(maxh=0.4))
    full_fespace = ngsolve.H1(mesh, order=3)

    cf = 8 * ngsolve.x * (1 - ngsolve.x) * ngsolve.y * (1 - ngsolve.y)

    full_gf = ngsolve.GridFunction(full_fespace)
    full_gf.Set(cf, ngsolve.VOL)

    reduced_fespace = ngsolve.H1(reduced_mesh, order=2)

    M = construct_transition_mass_matrix(full_fespace, reduced_fespace)
    reduced_massmatrix = construct_mass_matrix(reduced_fespace)
    reduced_gf = ngsolve.GridFunction(reduced_fespace)

    reduced_gf.Set(cf)
    print(reduced_gf.vec.FV().NumPy())

    rhs_vector = np.empty(reduced_massmatrix.shape[0])
    for i in range(reduced_fespace.ndof):
        basisfunction = ngsolve.GridFunction(reduced_fespace)
        basisfunction.vec[i] = 1
        rhs_vector[i] = ngsolve.Integrate(cf * basisfunction, mesh)

    reduced_arr_manual = sp.linalg.spsolve(reduced_massmatrix, rhs_vector)
    print(reduced_arr_manual)
    reduced_gf_manual = ngsolve.GridFunction(reduced_fespace)
    reduced_gf_manual.vec.FV().NumPy()[:] = reduced_arr_manual

    mesh_functions.plot_CF_colormap(reduced_gf, mesh, refinement_level=3, title='Ngsolve projection')
    mesh_functions.plot_CF_colormap(reduced_gf_manual, mesh, refinement_level=3, title='Manual projection')
    mesh_functions.plot_CF_colormap(cf, mesh, refinement_level=3, title='True')
    mesh_functions.plot_CF_colormap(reduced_gf - reduced_gf_manual, mesh, refinement_level=3, title='Difference')
    plt.show()
