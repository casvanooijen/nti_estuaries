import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve, eigs, qmr, gmres, bicgstab
import timeit
import ngsolve
import copy

from hydrodynamics import *
import define_weak_forms as weakforms
from mesh_functions import plot_CF_colormap


# Main function


def solve(hydro: Hydrodynamics, max_iterations: int = 10, tolerance: float = 1e-9, linear_solver = 'pardiso', 
          continuation_parameters: dict = {'advection_epsilon': [1], 'Av': [1]}, stopcriterion = 'scaled_2norm',
          preconditioner = None, plot_intermediate_results='none'):

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
        - preconditioner:                   choice of preconditioner for an iterative method; options: None, 'Jacobi', 'Gauss-Seidel', 'reduced_basis', ...,
        - plot_intermediate_results:        indicates whether intermediate results should be plotted and saved; options: 'none' (default) and 'all'.
    
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
            elif linear_solver == 'scipy_direct':
                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                rhs_arr = rhs.FV().NumPy()[freedof_list]
                plt.imshow(np.absolute(mat.todense()), cmap='viridis')

                # if newton_counter > 0:
                #     eigs, _ = np.linalg.eig(mat.todense())
                #     abs_eigs = np.absolute(eigs)
                #     print(f'    Condition number is equal to {np.amax(abs_eigs)/ np.amin(abs_eigs)}')
                #     print(f'    NNZ: {mat.nnz}')
                    
                #     fig, ax = plt.subplots()
                #     ax.plot(np.sort(abs_eigs)[::-1])
                #     ax.set_title(f'Spectrum in iteration {newton_counter}')
                #     ax.set_yscale('log')


                sol = spsolve(mat, rhs_arr)
                du.vec.FV().NumPy()[freedof_list] = sol
            elif linear_solver == 'bicgstab':
                freedof_list = get_freedof_list(hydro.femspace.FreeDofs())
                mat = remove_fixeddofs_from_csr(basematrix_to_csr_matrix(a.mat), freedof_list)
                rhs_arr = rhs.FV().NumPy()[freedof_list]


                # construct preconditioner


                if preconditioner == 'Jacobi': # diagonal scaling preconditioner
                    diagonal_entries = mat.diagonal()
                    diagonal_entries_nonzero = np.where(diagonal_entries == 0, np.ones_like(diagonal_entries), diagonal_entries)
                    print(f'    Condition number is {np.linalg.cond(mat.todense())}')
                    prec = sp.diags(np.power(diagonal_entries_nonzero, -1))
                    print(f'    Preconditioned condition number is {np.linalg.cond(mat.todense() @ prec)}')
                
                # plt.imshow(mat.todense(), cmap='RdBu')

                plt.spy(mat.todense())
                if preconditioner is not None:
                    sol, exitcode = bicgstab(mat, rhs_arr, maxiter=500, M=prec)
                else:
                    sol, exitcode = bicgstab(mat, rhs_arr, maxiter=500)


                if exitcode != 0:
                    print('    BiCG-STAB did not converge in 500 iterations')

                du.vec.FV().NumPy()[freedof_list] = sol
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

                for i in range(-hydro.imax, hydro.imax):
                    plot_CF_colormap(hydro.gamma_solution[i], hydro.mesh, refinement_level=3, show_mesh=True, title=f'gamma_({i})', save = f"iteration{newton_counter}_gamma({i})")
                    u_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.alpha_solution[m][i] for m in range(hydro.M)])
                    plot_CF_colormap(u_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'u_DA_({i})', save = f"iteration{newton_counter}_uDA({i})")
                    v_DA = sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.beta_solution[m][i] for m in range(hydro.M)])
                    plot_CF_colormap(v_DA, hydro.mesh, refinement_level=3, show_mesh=True, title=f'v_DA_({i})', save = f"iteration{newton_counter}_vDA({i})")


                plt.close(fig = 'all')

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


def bicgstab(A, f, u0, tol=1e-12, maxits = 500, reduced_A=None, transition_mass_matrix=None):
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
    - transition_mass_matrix:   matrix specifying how coefficient vectors of the full basis transfer to coefficient vectors of the reduced basis;    
    
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

        preconditioned_p = np.copy(p) # Add reduced-basis preconditioner!
        v = A @ preconditioned_p
        alpha = rho / np.inner(shadow_r0, v)
        s = r - alpha * v

        z = np.copy(s) # Add reduced-basis preconditioner!
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
