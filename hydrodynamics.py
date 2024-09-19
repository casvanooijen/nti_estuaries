# import ngsolve
# from ngsolve.solvers import *
# # from ngsolve.la import IdentityMatrix, EigenValues_Preconditioner
# import numpy as np
# import scipy.sparse as sp
# import scipy.sparse.linalg
# import os
# import TruncationBasis
# import matplotlib.pyplot as plt
# from geometry.create_geometry import RIVER, SEA, BOUNDARY_DICT
# import copy
# from modeloptions import ModelOptions
# import define_weak_forms as weakforms
# from minusonepower import minusonepower
# import mesh_functions
# import timeit
# from ngsolve.solvers import GMRes
# import pypardiso

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import copy
import os
import timeit
import json
import dill

import ngsolve
from ngsolve.solvers import *
import pypardiso

import TruncationBasis
from geometry.create_geometry import RIVER, SEA, BOUNDARY_DICT
from boundary_fitted_coordinates import generate_bfc
from spatial_parameter import SpatialParameter

import define_weak_forms as weakforms
from minusonepower import minusonepower
import mesh_functions




def count_free_dofs(fes):
    """
    Returns the number of free degrees of freedom in an ngsolve Finite Element space.

    Arguments:

        - fes:      ngsolve Finite element space.   
    
    """
    i = 0
    for isFree in fes.FreeDofs():
        i = i + isFree
    return i


def homogenise_essential_Dofs(vec: ngsolve.BaseVector, freedofs):
    """
    Sets the essential (non-free) degrees of freedom of an ngsolve BaseVector to zero.

    Arguments:
        
        - vec:          ngsolve BaseVector;
        - freedofs:     bitarray indicating the free degrees of freedom, obtainable via calling the method FiniteElementSpace.FreeDofs();
    
    
    """
    for i, free in enumerate(freedofs):
        if not free:
            vec[i] = 0.


def select_model_options(bed_bc:str = 'no-slip', surface_in_sigma:bool = True, veddy_viscosity_assumption:str = 'constant', density:str = 'depth-independent',
                 advection_epsilon:float = 1, advection_influence_matrix: np.ndarray = None, x_scaling: float = 1., y_scaling: float = 1):
    
    """
    
    Returns a dictionary of all the available model options. Safer than manually creating this dictionary.
    
    Arguments: ('...'  means that there will possibly be future options added)
        
        - bed_bc:                                   indicates what type of boundary condition is used at the river bed ('no_slip' or 'partial_slip');
        - surface_in_sigma (bool):                  flag to indicate whether non-linear effects stemming from presence of the surface in the sigma-coordinates are included;
        - veddy_viscosity_assumption:               structure of the vertical eddy viscosity parameter ('constant' or 'depth-scaled&constantprofile' or ...);
        - density:                                  indicates what type of water density field is used ('depth-independent' or ...);
        - advection_epsilon (float):                scalar by which the advective terms in the momentum equations are multiplied; if set to zero, advective terms are skipped;     
                                                    if set to one, advective terms are fully included;
        - advection_influence_matrix (np.ndarray):  (imax+1) x (imax+1) - boolean matrix where element (i,j) indicates whether constituent i is influenced by constituent j through momentum advection (if possible);
                                                    more precisely, in the equations for constituent i, any product of constituents that includes constituent j will not be present in the advective terms
                                                    if element (i, j) is False, even if that product *should* physically be present;    
        - x_scaling (float):                        factor [m] by which the input geometry should be scaled in the x-direction; this variable adds scaling factors in the equations to compensate for this; default = 1
        - y_scaling (float):                        factor [m] by which the input geometry should be scaled in the y-direction; default = 1
        
        """
    
    if bed_bc == 'partial_slip' and veddy_viscosity_assumption == 'constant':
        raise ValueError("Partial-slip condition and constant vertical eddy viscosity are incompatible")

    options = {
            'bed_bc': bed_bc,
            'surface_in_sigma': surface_in_sigma,
            'veddy_viscosity_assumption': veddy_viscosity_assumption,
            'density': density,
            'advection_epsilon': advection_epsilon,
            'advection_influence_matrix': advection_influence_matrix, # the validity of this matrix is checked when imax is know, i.e. when the hydrodynamics object is initialised
            'x_scaling': x_scaling,
            'y_scaling': y_scaling
        }
    

    return options


class Hydrodynamics(object):

    """
    Class that collects all necessary components to solve the hydrodynamic model and sets up the matrix. Also contains solution methods, but this
    will later be separate.

    Attributes:

        - mesh (ngsolve.Mesh):                                  mesh on which the model is defined;
        - model_options (dict):                                 dictionary of model options, created using select_model_options;
        - imax (int):                                           number of tidal constituents taken into account, excluding subtidal;
        - M (int):                                              number of vertical basis functions taken into account;
        - order (int):                                          order of the spectral element basis (Dubiner basis);
        - time_basis (TruncationBasis):                         harmonic time basis;
        - vertical_basis (TruncationBasis):                     vertical basis;
        - vertical_basis_name (str):                            name of the vertical basis if it is a default one from TruncationBasis.py
        - constant_physical_parameters (dict):                  dictionary containing values of constant physical parameters;
        - spatial_physical_parameters (dict):                   dictionary containing spatially varying physical parameters, such as
                                                                bathymetry, in the form of SpatialParameter objects;
        - loaded_from_files (bool):                             flag indicating whether this hydrodynamics-object was manually created or loaded in;
        - nfreedofs (int):                                      number of free degrees of freedom;
        - femspace (ngsolve.comp.H1):                           finite element space associated with this mesh and parameter set;
        - n:                                                    ngsolve normal vector to the boundary of the mesh;
        - alpha_trialfunctions (dict):                          dictionary of dictionaries of trial functions (ngsolve.comp.ProxyFunction) representing all the coefficients alpha_{mi} that form along-channel velocity u;
        - umom_testfunctions (dict):                            dictionary of dictionaries of test functions (ngsolve.comp.ProxyFunction) for the along-channel momentum equation;
        - beta_trialfunctions (dict):                           dictionary of dictionaries of trial functions representing all the coefficients beta_{mi} that form cross-channel velocity v;
        - vmom_testfunctions (dict):                            dictionary of dictionaries of test functions for the cross-channel momentum equation;
        - gamma_trialfunctions (dict):                          dictionary of trial functions representing all the coefficients gamma_{i} that form the water surface zeta;
        - DIC_trialfunctions (dict):                            dictionary of test functions for the Depth-Integrated Continuity equation;
        - total_bilinearform (ngsolve.comp.BilinearForm):       bilinear form of the entire system of projected PDEs; not necessarily bilinear, but ngsolve refers to any weak form as a bilinear form;
        - solution_gf (ngsolve.GridFunction):                   gridfunction that contains the entire vector solution of the system of PDEs;
        - alpha_solution (dict):                                dictionary of dictionaries of the solutions for the coefficients alpha_{mi};
        - beta_solution (dict):                                 dictionary of dictionaries of the solutions for the coefficients beta_{mi};
        - gamma_solution (dict):                                dictionary of the solutions for the coefficients gamma_{i};
        - seaward_forcing (SeawardForcing):                     SeawardForcing-object containing the details of the seaward boundary condition;
        - riverine_forcing (RiverineForcing):                   RiverineForcing-object containing the details of the riverine (landward) boundary condition;
        - scaling (bool):                                       flag indicating whether the geometry and hence the equations are scaled;
    
    """

    def __init__(self, mesh: ngsolve.Mesh, model_options:dict, imax:int, M:int, order:int, 
                 time_basis:TruncationBasis.TruncationBasis, vertical_basis:TruncationBasis.TruncationBasis):
        
        self.mesh = mesh
        self.model_options = model_options
        self.imax = imax
        self.M = M
        self.num_equations = (2*M+1)*(2*imax + 1)
        self.order = order
        self.time_basis = time_basis
        self.vertical_basis = vertical_basis
        self.constant_physical_parameters = dict()
        self.spatial_physical_parameters = dict()
        
        self.loaded_from_files = False # flag that indicates whether the object is loaded from a saved solution; if it is, the spatial parameters are stored differently
        if self.vertical_basis is TruncationBasis.eigbasis_constantAv: # not so pretty but it works
            self.vertical_basis_name = "eigbasis_constantAv"

        # check/generate advection_influence_matrix

        if self.model_options['advection_influence_matrix'] is None:
            self.model_options['advection_influence_matrix'] = np.full((self.imax + 1, self.imax + 1), True) # in this case, every constituent affects every other constituent through advection, as would be physical
        elif self.model_options['advection_influence_matrix'].shape != (self.imax+1, self.imax+1):
            raise ValueError(f"Invalidly shaped advection influence matrix, please provide a square boolean matrix of size imax+1: {self.imax+1}")
        elif self.model_options['advection_influence_matrix'].dtype != bool:
            raise ValueError(f"Invalid advection influence matrix, please provide a *boolean* matrix")

        self._setup_fem_space()
        self.nfreedofs = count_free_dofs(self.femspace)

        self._setup_TnT()
        self._get_normalvec()

        if abs(self.model_options['x_scaling'] - 1) > 1e-12 or abs(self.model_options['y_scaling'] - 1) > 1e-12:
            self.scaling = True
        else:
            self.scaling = False
        
        # Get norms of the basis functions

        # zero_gridfunction = ngsolve.GridFunction(self.femspace)
        # for k in range(self.num_equations):
        #     zero_gridfunction.components[k].Set(0)

        # self.fem_basis_norms = np.zeros(self.nfreedofs)
        # # for n in range(self.nfreedofs):
        # for n in range(1):
        #     r = zero_gridfunction.vec.CreateVector()
        #     r[n] = 1
        #     self.fem_basis_norms[n] = ngsolve.InnerProduct(r, r)
            




    # Private methods

    def _setup_fem_space(self):
        U = ngsolve.H1(self.mesh, order=self.order) 
        G = ngsolve.H1(self.mesh, order=self.order, dirichlet=BOUNDARY_DICT[SEA])

        list_of_spaces = [U for _ in range(2*self.M*(2*self.imax + 1))]
        for _ in range(2*self.imax+1):
            list_of_spaces.append(G)

        X = ngsolve.FESpace(list_of_spaces) # tensor product of all spaces
        self.femspace = X
        # self.Idmat = IdentityMatrix(X.ndof, complex=False) # save identity matrix for use later when computing condition numbers
        




    def _get_normalvec(self):
        X = self.femspace
        self.n = ngsolve.specialcf.normal(2)
    

    def _setup_TnT(self):


        """Sorts the ngsolve Trial and Test functions into intuitive dictionaries"""

        trialtuple = self.femspace.TrialFunction()
        testtuple = self.femspace.TestFunction()

        alpha_trialfunctions = [dict() for _ in range(self.M)]
        umom_testfunctions = [dict() for _ in range(self.M)] # test functions for momentum equation u

        beta_trialfunctions = [dict() for _ in range(self.M)]
        vmom_testfunctions = [dict() for _ in range(self.M)] # test functions for momentum equation v

        gamma_trialfunctions = dict()
        DIC_testfunctions = dict() # test functions for Depth-Integrated Continuity equation

        for m in range(self.M):
            alpha_trialfunctions[m][0] = trialtuple[m * (2*self.imax + 1)]
            umom_testfunctions[m][0] = testtuple[m * (2*self.imax + 1)]

            beta_trialfunctions[m][0] = trialtuple[(self.M + m) * (2*self.imax + 1)]
            vmom_testfunctions[m][0] = testtuple[(self.M + m) * (2*self.imax + 1)]
            for q in range(1, self.imax + 1):
                alpha_trialfunctions[m][-q] = trialtuple[m * (2*self.imax + 1) + q]
                alpha_trialfunctions[m][q] = trialtuple[m * (2*self.imax + 1) + self.imax + q]

                umom_testfunctions[m][-q] = testtuple[m * (2*self.imax + 1) + q]
                umom_testfunctions[m][q] = testtuple[m * (2*self.imax + 1) + self.imax + q]

                beta_trialfunctions[m][-q] = trialtuple[(self.M + m) * (2*self.imax + 1) + q]
                beta_trialfunctions[m][q] = trialtuple[(self.M + m) * (2*self.imax + 1) + self.imax + q]

                vmom_testfunctions[m][-q] = testtuple[(self.M + m) * (2*self.imax + 1) + q]
                vmom_testfunctions[m][q] = testtuple[(self.M + m) * (2*self.imax + 1) + self.imax + q]
        
        gamma_trialfunctions[0] = trialtuple[2*(self.M)*(2*self.imax+1)]
        DIC_testfunctions[0] = testtuple[2*(self.M)*(2*self.imax+1)]

        for q in range(1, self.imax + 1):
            gamma_trialfunctions[-q] = trialtuple[2*(self.M)*(2*self.imax+1) + q]
            gamma_trialfunctions[q] = trialtuple[2*(self.M)*(2*self.imax+1) + self.imax + q]

            DIC_testfunctions[-q] = testtuple[2*(self.M)*(2*self.imax+1) + q]
            DIC_testfunctions[q] = testtuple[2*(self.M)*(2*self.imax+1) + self.imax + q]

        self.alpha_trialfunctions = alpha_trialfunctions
        self.umom_testfunctions = umom_testfunctions
        self.beta_trialfunctions = beta_trialfunctions
        self.vmom_testfunctions = vmom_testfunctions
        self.gamma_trialfunctions = gamma_trialfunctions
        self.DIC_testfunctions = DIC_testfunctions


    def _setup_forms(self, skip_nonlinear=False):

        self._get_normalvec()

        a_total = ngsolve.BilinearForm(self.femspace)

        if self.scaling:
            weakforms.add_weak_form(a_total, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
                                self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis,
                                self.riverine_forcing.normal_alpha, only_linear=(not skip_nonlinear))
        else:
            weakforms.add_bilinear_part(a_total, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                        self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
                                        self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis,
                                        self.riverine_forcing.normal_alpha, forcing=True)
            if not skip_nonlinear:
                weakforms.add_nonlinear_part(a_total, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                            self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
                                            self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis,
                                            self.riverine_forcing.normal_alpha, self.n)

        
        
        self.total_bilinearform = a_total


    def _restructure_solution(self):

        """Associates each part of the solution gridfunction vector to a Fourier and vertical eigenfunction pair."""

        self.alpha_solution = [dict() for _ in range(self.M)]
        self.beta_solution = [dict() for _ in range(self.M)]
        self.gamma_solution = dict()

        for m in range(self.M):
            self.alpha_solution[m][0] = self.solution_gf.components[m * (2*self.imax + 1)]
            self.beta_solution[m][0] = self.solution_gf.components[(self.M + m) * (2*self.imax + 1)]
            
            for q in range(1, self.imax + 1):
                self.alpha_solution[m][-q] = self.solution_gf.components[m * (2*self.imax + 1) + q]
                self.alpha_solution[m][q] = self.solution_gf.components[m * (2*self.imax + 1) + self.imax + q]

                self.beta_solution[m][-q] = self.solution_gf.components[(self.M + m) * (2*self.imax + 1) + q]
                self.beta_solution[m][q] = self.solution_gf.components[(self.M + m) * (2*self.imax + 1) + self.imax + q]
        
        self.gamma_solution[0] = self.solution_gf.components[2*(self.M)*(2*self.imax+1)]

        for q in range(1, self.imax + 1):
            self.gamma_solution[-q] = self.solution_gf.components[2*(self.M)*(2*self.imax+1) + q]
            self.gamma_solution[q] = self.solution_gf.components[2*(self.M)*(2*self.imax+1) + self.imax + q]


    # def _construct_velocities(self): # THIS FUNCTIONALITY IS PERFORMED BY postprocessing.py
    #     """Still dependent on precise form of orthogonal basis: FIX"""
    #     self.u = dict()
    #     self.v = dict()
    #     self.w = dict()

    #     self.u[0] = sum([self.alpha_solution[m][0]*self.vertical_basis.coefficient_function(m) for m in range(self.M + 1)])
    #     self.v[0] = sum([self.beta_solution[m][0]*self.vertical_basis.coefficient_function(m) for m in range(self.M + 1)])
  
    #     for q in range(1, self.imax + 1):
    #         self.u[-q] = sum([self.alpha_solution[m][-q]*self.vertical_basis.coefficient_function(m) for m in range(self.M + 1)])
    #         self.v[-q] = sum([self.beta_solution[m][-q]*self.vertical_basis.coefficient_function(m) for m in range(self.M + 1)])
    #         self.u[q] = sum([self.alpha_solution[m][q]*self.vertical_basis.coefficient_function(m) for m in range(self.M + 1)])
    #         self.v[q] = sum([self.beta_solution[m][q]*self.vertical_basis.coefficient_function(m) for m in range(self.M + 1)])

    #     omegatilde = dict() # alternative vertical velocity in Burchard & Petersen (1997)
    #     omegatilde[0] = (-1/self.spatial_physical_parameters['H'].cf) * sum([
    #         (self.spatial_physical_parameters['H'].cf * (ngsolve.grad(self.alpha_solution[m][0])[0] + ngsolve.grad(self.beta_solution[m][0])[1])
    #         + self.alpha_solution[m][0] * self.spatial_physical_parameters['H'].gradient_cf[0] + self.beta_solution[m][0] * self.spatial_physical_parameters['H'].gradient_cf[1])
    #         * minusonepower(m) / ((m+0.5)*ngsolve.pi) * (ngsolve.sin((m+0.5)*ngsolve.pi*ngsolve.z) + minusonepower(m))
    #     for m in range(self.M + 1)])

    #     for q in range(1, self.imax + 1):
    #         omegatilde[-q] = (-1/self.spatial_physical_parameters['H'].cf) * sum([
    #             (self.spatial_physical_parameters['H'].cf * (ngsolve.grad(self.alpha_solution[m][-q])[0] + ngsolve.grad(self.beta_solution[m][-q])[1])
    #             + self.alpha_solution[m][-q] * self.spatial_physical_parameters['H'].gradient_cf[0] + self.beta_solution[m][-q] * self.spatial_physical_parameters['H'].gradient_cf[1])
    #             * minusonepower(m) / ((m+0.5)*ngsolve.pi) * (ngsolve.sin((m+0.5)*ngsolve.pi*ngsolve.z) + minusonepower(m))
    #         for m in range(self.M + 1)])
    #         omegatilde[q] = (-1/self.spatial_physical_parameters['H'].cf) * sum([
    #             (self.spatial_physical_parameters['H'].cf * (ngsolve.grad(self.alpha_solution[m][q])[0] + ngsolve.grad(self.beta_solution[m][q])[1])
    #             + self.alpha_solution[m][q] * self.spatial_physical_parameters['H'].gradient_cf[0] + self.beta_solution[m][q] * self.spatial_physical_parameters['H'].gradient_cf[1])
    #             * minusonepower(m) / ((m+0.5)*ngsolve.pi) * (ngsolve.sin((m+0.5)*ngsolve.pi*ngsolve.z) + minusonepower(m))
    #         for m in range(self.M + 1)])

    #     self.w[0] = self.spatial_physical_parameters['H'].cf * omegatilde[0] + self.u[0] * self.spatial_physical_parameters['H'].gradient_cf[0] + \
    #                 self.v[0] * self.spatial_physical_parameters['H'].gradient_cf[1]
        
    #     for q in range(1, self.imax + 1):
    #         self.w[-q] = self.spatial_physical_parameters['H'].cf * omegatilde[-q] + self.u[-q] * self.spatial_physical_parameters['H'].gradient_cf[0] + \
    #                 self.v[-q] * self.spatial_physical_parameters['H'].gradient_cf[1]
    #         self.w[q] = self.spatial_physical_parameters['H'].cf * omegatilde[q] + self.u[q] * self.spatial_physical_parameters['H'].gradient_cf[0] + \
    #                 self.v[q] * self.spatial_physical_parameters['H'].gradient_cf[1]


    # def _construct_depth_averaged_velocities(self):

    #     self.u_DA = dict()
    #     self.v_DA = dict()

    #     self.u_DA[0] = sum([self.vertical_basis.tensor_dict['G4'](m) * self.alpha_solution[m][0] for m in range(self.M)])
    #     self.v_DA[0] = sum([self.vertical_basis.tensor_dict['G4'](m) * self.beta_solution[m][0] for m in range(self.M)])

    #     for q in range(1, self.imax + 1):
    #         self.u_DA[-q] = sum([self.vertical_basis.tensor_dict['G4'](m) * self.alpha_solution[m][-q] for m in range(self.M)])
    #         self.u_DA[q] = sum([self.vertical_basis.tensor_dict['G4'](m) * self.alpha_solution[m][q] for m in range(self.M)])

    #         self.v_DA[-q] = sum([self.vertical_basis.tensor_dict['G4'](m) * self.beta_solution[m][-q] for m in range(self.M)])
    #         self.v_DA[q] = sum([self.vertical_basis.tensor_dict['G4'](m) * self.beta_solution[m][q] for m in range(self.M)])

    # Public methods

    # def add_solution(self, filename):
    #     self.solution_gf = ngsolve.GridFunction(self.femspace)
    #     mesh_functions.set_basevector_from_txt(self.solution_gf.vec, filename)
    #     self._restructure_solution()
    #     # self._construct_velocities()
    #     self._construct_depth_averaged_velocities()


    # def save_solution(self, filename):
    #     mesh_functions.save_gridfunction_to_txt(self.solution_gf, filename)

    def save(self, name, **kwargs):
        """Saves the hydrodynamics object. Only possible if the Fourier/vertical bases are chosen from the predefined 
        bases in TruncationBasis.py. The folder contains:
        
        - options.txt:          contains the model options of the ModelOptions object, including which Fourier/vertical bases were used;
        - params.txt:           contains the SEM expansion basis order, M, imax and the constant physical parameters;
        - mesh.vol:             file that can be read by NGSolve to regenerate your mesh;
        - spatial_parameters:   folder that contain function handles of spatial parameters;
        - solution.txt          file that contains the solution GridFunction;

        """
        os.makedirs(name, exist_ok=True)

        # model options

        options = {'vertical_basis_name': self.vertical_basis_name}
        options.update(self.model_options)

        options['advection_influence_matrix'] = options['advection_influence_matrix'].tolist()

        f_options = open(f"{name}/options.json", 'x')
        json.dump(options, f_options, indent=4)
        f_options.close()

        # options_string = f"bed_bc:{self.model_options['bed_bc']}\nleading_order_surface:{self.model_options['leading_order_surface']}\n"+\
        #                  f"veddy_viscosity_assumption:{self.model_options['veddy_viscosity_assumption']}\ndensity:{self.model_options['density']}\n"+\
        #                  f"advection_epsilon:{self.model_options['advection_epsilon']}\nvertical_basis_name:{self.vertical_basis_name}"
        
        # f_options = open(f"{name}/options.txt", 'x')
        # f_options.write(options_string)
        # f_options.close()

        # constant parameters

        params = {'sem_order': self.order, 'M': self.M, 'imax': self.imax}
        params.update(self.constant_physical_parameters)

        f_params = open(f"{name}/params.json", 'x')
        json.dump(params, f_params, indent=4)
        f_params.close()

        # params_string = f"sem_order:{self.order}\nM:{self.M}\nimax:{self.imax}"
        # for name, value in self.constant_physical_parameters.items():
        #     params_string += f"\n{name}:{value}"
        
        # f_params = open(f"{name}/params.txt", 'x')
        # f_params.write(params_string)
        # f_params.close()

        # mesh

        self.mesh.ngmesh.Save(f'{name}/mesh.vol') # save the netgen mesh

        # spatial parameters

        os.makedirs(f'{name}/spatial_parameters')

        for paramname, value in self.spatial_physical_parameters.items():
            with open(f'{name}/spatial_parameters/{paramname}.pkl', 'wb') as file:
                dill.dump(value.fh, file, protocol=dill.HIGHEST_PROTOCOL)

            # oneDfemspace = ngsolve.H1(self.mesh, order = self.order)
            # gf = ngsolve.GridFunction(oneDfemspace)
            # gf.Set(value.cf)

            # mesh_functions.save_gridfunction(gf, f"{name}/spatial_parameters/{name}")

        # solution

        mesh_functions.save_gridfunction(self.solution_gf, f"{name}/solution", **kwargs)

        


    def hrefine(self, threshold: float, numits: int = 1, based_on = 'bathygrad'):
        """
        Refines the mesh a number of iterations based on the following rule: if the integrated 'based_on'-quantity in a particular element exceeds 
        a threshold times the overall arithmetic average (of all elements) integrated 'based_on'-quantity in the mesh, that element is marked for
        refinement. This procedure is performed a user-specified number of times.

        Arguments:

            - threshold (float):    factor larger than one that is used for the refinement rule;
            - numits (int):         number of times the mesh is refined;
            - based_on (str):       integrable quantity the rule is based on; options are 'bathygrad' which bases the rule on the norm of the bathymetry gradient; 
        
        """
        if based_on == 'bathygrad':
            if self.loaded_from_files:
                bathy_gradnorm = ngsolve.sqrt(ngsolve.grad(self.spatial_physical_parameters['H'])[0] * ngsolve.grad(self.spatial_physical_parameters['H'])[0] + 
                                            ngsolve.grad(self.spatial_physical_parameters['H'])[1] * ngsolve.grad(self.spatial_physical_parameters['H'])[1])
            else:
                bathy_gradnorm = ngsolve.sqrt(self.spatial_physical_parameters['H'].gradient_cf[0] * self.spatial_physical_parameters['H'].gradient_cf[0] + 
                                            self.spatial_physical_parameters['H'].gradient_cf[1] * self.spatial_physical_parameters['H'].gradient_cf[1])
        else:
            raise ValueError("Invalid value for 'based_on'. Please choose from the following options: 'bathygrad'.")
            
        for _ in range(numits):

            num_refined = mesh_functions.refine_mesh_by_elemental_integration(self.mesh, bathy_gradnorm, threshold)

            if not self.loaded_from_files:
                for name, param in self.spatial_physical_parameters.items(): # SpatialParameter-objects need to be redefined on the new mesh
                    bfc = generate_bfc(self.mesh, self.order, 'diffusion')
                    self.spatial_physical_parameters[name] = SpatialParameter(param.fh, bfc)

                bathy_gradnorm = ngsolve.sqrt(self.spatial_physical_parameters['H'].gradient_cf[0] * self.spatial_physical_parameters['H'].gradient_cf[0] + 
                                              self.spatial_physical_parameters['H'].gradient_cf[1] * self.spatial_physical_parameters['H'].gradient_cf[1])
                
            if num_refined == 0:
                break



    def set_constant_physical_parameters(self, Av=None, sigma=None, T=None, g=None, f=None):
        if Av is not None:
            self.constant_physical_parameters['Av'] = Av # You should only set this if you have assumed Av to be constant

        if sigma is not None:
            self.constant_physical_parameters['sigma'] = sigma # M2-Frequency
        elif T is not None:
            self.constant_physical_parameters['sigma'] = 1/T
        
        if g is not None:
            self.constant_physical_parameters['g'] = g # Probably 9.81, but let us keep outer space estuaries into our model :)
        if f is not None:
            self.constant_physical_parameters['f'] = f


    def set_spatial_physical_parameters(self, H=None, density=None, R=None):
        if H is not None:
            self.spatial_physical_parameters['H'] = H
        if density is not None:
            self.spatial_physical_parameters['density'] = density
        if R is not None:
            self.spatial_physical_parameters['R'] = R


    def set_seaward_boundary_condition(self, amplitude_list, phase_list):
        self.seaward_forcing = SeawardForcing(self, amplitude_list, phase_list)


    def set_riverine_boundary_condition(self, discharge_amplitude_list, discharge_phase_list, **kwargs):
        self.riverine_forcing = RiverineForcing(self, discharge_amplitude_list, discharge_phase_list, **kwargs)


    def SolveNewton(self, advection_weighting_parameter, tol, maxitns, printing=True, print_cond=False, autodiff=False, method='pardiso', return_vals=False):
        u_n = copy.copy(self.solution_gf)
        
        # mesh_functions.plot_gridfunction_colormap(u_n, u_n.space.mesh, refinement_level=3)
        for i in range(maxitns):
            if printing:
                print(f"Newton iteration {i}:")

            if not autodiff:
                if return_vals:
                    invtime = self.NewtonIteration(advection_weighting_parameter, print_cond, method=method, return_invtime=True)
                else:
                    self.NewtonIteration(advection_weighting_parameter, print_cond, method=method)
            else:
                self.NewtonIteration_autodiff(u_n.vec, method=method)
            
            residual = u_n.vec.CreateVector()
            apply_start = timeit.default_timer()
            self.total_bilinearform.Apply(self.solution_gf.vec, residual)
            apply_time = timeit.default_timer() - apply_start
            homogenise_essential_Dofs(residual, self.femspace.FreeDofs())
            

            stop_criterion_value = abs(ngsolve.InnerProduct(self.solution_gf.vec - u_n.vec, residual))
            residual_norm_sq = ngsolve.InnerProduct(residual, residual)

            # residual_array = residual.FV().NumPy()

            # stop_criterion_value = np.sqrt(stop_criterion_value) / np.sqrt(self.num_equations)
            print(f"   Evaluating weak forms took {apply_time} seconds")
            print(f"   Stopping criterion value is equal to {stop_criterion_value}")
            print(f"   Residual norm (except Dirichlet boundary) is equal to {np.sqrt(residual_norm_sq)}")
            print(f"   Scaled residual norm (free DOFs) is {np.sqrt(residual_norm_sq) / np.sqrt(self.nfreedofs)}\n")

            residual_gf = ngsolve.GridFunction(self.femspace)
            residual_gf.vec.data = residual

            # for k in range(5):
            #     mesh_functions.plot_gridfunction_colormap(residual_gf.components[k], self.mesh, refinement_level=3, title=f'Iteration {i}, component {k}')

            u_n = copy.copy(self.solution_gf)

            # mesh_functions.plot_gridfunction_colormap(u_n, u_n.space.mesh, refinement_level=3)

            if stop_criterion_value < tol:
                break
        
        if return_vals:
            return np.sqrt(residual_norm_sq), invtime


    def NewtonIteration(self, advection_weighting_parameter, print_cond=False, method='pardiso', return_invtime=False):
        self._restructure_solution()
        forms_start = timeit.default_timer()
        a = ngsolve.BilinearForm(self.femspace)
        # weakforms.add_bilinear_part(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
        #                             self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
        #                             self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis,
        #                             self.riverine_forcing.normal_alpha, forcing=True)
        if self.scaling:
            weakforms.add_weak_form(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                    self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
                                    self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis,
                                    self.riverine_forcing.normal_alpha, only_linear=True)
            if advection_weighting_parameter != 0:
                # weakforms.add_linearised_nonlinear_part(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                #                                         self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.alpha_solution, self.beta_solution, self.gamma_solution,
                #                                         self.M, self.imax, self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis, self.riverine_forcing.normal_alpha,
                #                                         advection_weighting_parameter, self.n)
                weakforms.add_linearised_nonlinear_terms(a, self.model_options, self.alpha_trialfunctions, self.alpha_solution, self.beta_trialfunctions, self.beta_solution,
                                                        self.gamma_trialfunctions, self.gamma_solution, 
                                                        self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
                                                        self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis,
                                                        self.riverine_forcing.normal_alpha)
        else:
            weakforms.add_bilinear_part(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                    self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
                                    self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis,
                                    self.riverine_forcing.normal_alpha, forcing=True)
            if advection_weighting_parameter != 0:
                weakforms.add_linearised_nonlinear_part(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                                        self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.alpha_solution, self.beta_solution, self.gamma_solution,
                                                        self.M, self.imax, self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis, self.riverine_forcing.normal_alpha,
                                                        advection_weighting_parameter, self.n)
                
            # add_linearised_nonlinear_part_to_bilinearform(self, a, self.alpha_solution, self.beta_solution, self.gamma_solution, advection_weighting_parameter)
        forms_time = timeit.default_timer() - forms_start
        if method == 'gmres':
            prec = ngsolve.Preconditioner(a, 'direct')
        assembly_start = timeit.default_timer()
        a.Assemble()
        assembly_time = timeit.default_timer() - assembly_start
        if method == 'gmres':
            prec.Update()

        # Preconditioner

        # Jacobi_pre_mat = a.mat.CreateSmoother(self.femspace.FreeDofs())

        rhs = self.solution_gf.vec.CreateVector()
        self.total_bilinearform.Apply(self.solution_gf.vec, rhs)

        du = ngsolve.GridFunction(self.femspace)
        for i in range(self.femspace.dim):
            du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions
        if print_cond and method != 'gmres': # CANT USE THIS OPTION RIGHT NOW; IT IS PROBABLY INACCURATE ANYWAY
            Idmat = ngsolve.Projector(mask=self.femspace.FreeDofs(), range=True)

            # abs_eigs = np.absolute(np.array(EigenValues_Preconditioner(a.mat, Idmat)))

            # print(f'   Estimated condition number (2-norm) is {np.amax(abs_eigs)/np.amin(abs_eigs)}')

        # Direct
        if method == 'umfpack' or method == 'pardiso':
            inversion_start = timeit.default_timer()
            du.vec.data = a.mat.Inverse(freedofs=self.femspace.FreeDofs(), inverse=method) * rhs
            # du.vec.data = a.mat.Inverse(inverse=method) * rhs
            inversion_time = timeit.default_timer() - inversion_start
        elif method == 'ext_pardiso':
            inversion_start = timeit.default_timer()
            A = mesh_functions.get_csr_matrix(a.mat)
            f = rhs.FV().NumPy()
            sol_arr = pypardiso.spsolve(A, f)
            du.vec.FV().NumPy()[:] = sol_arr

            inversion_time = timeit.default_timer() - inversion_start

        # GMRes(a.mat, freedofs=self.femspace.FreeDofs(), x=du.vec, b=rhs)
        # Iterative
        if method == 'gmres':
            # if print_cond: AGAIN, CANT USE THIS OPTION RIGHT NOW; IT IS PROBABLY INACCURATE ANYWAY
                # abs_eigs = np.absolute(np.array(EigenValues_Preconditioner(a.mat, prec)))
                # print(f'   Estimated preconditioned condition number (2-norm) is {np.amax(abs_eigs)/np.amin(abs_eigs)}')
            inversion_start = timeit.default_timer()
            GMRes(a.mat, pre=prec, x=du.vec, b=rhs)
            inversion_time = timeit.default_timer() - inversion_start

        # To print norm of rhs for non-Dirichlet DOFs
        homogenise_essential_Dofs(rhs, self.femspace.FreeDofs())
        print(f'   Norm of right-hand side is equal to {ngsolve.Norm(rhs)}')
        print(f'   Norm of Newton step is {ngsolve.Norm(du.vec)}')
        print(f'   Setting up weak forms for linearisation took {forms_time} seconds')
        print(f'   Inversion took {inversion_time} seconds')
        print(f'   Assembly took {assembly_time} seconds')

        self.solution_gf.vec.data = self.solution_gf.vec.data - du.vec.data
        if return_invtime:
            return inversion_time


    
    def NewtonIteration_autodiff(self, u_n, method='pardiso'):
        self._restructure_solution()

        rhs = self.solution_gf.vec.CreateVector()
        self.total_bilinearform.Apply(self.solution_gf.vec, rhs)

        autodiff_start = timeit.default_timer()
        self.total_bilinearform.AssembleLinearization(u_n)
        autodiff_time = timeit.default_timer() - autodiff_start

        du = ngsolve.GridFunction(self.femspace)
        for i in range(self.femspace.dim):
            du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions

        inversion_start = timeit.default_timer()
        du.vec.data = self.total_bilinearform.mat.Inverse(freedofs=self.femspace.FreeDofs(), inverse=method) * rhs
        inversion_time = timeit.default_timer() - inversion_start

        # To print norm of rhs for non-Dirichlet DOFs
        homogenise_essential_Dofs(rhs, self.femspace.FreeDofs())
        print(f'   Norm of right-hand side is equal to {ngsolve.Norm(rhs)},')
        print(f'   Norm of Newton step is {ngsolve.Norm(du.vec)},')
        print(f'   Automatic differentiation took {autodiff_time} seconds,')
        print(f'   Inversion took {inversion_time} seconds.')

        self.solution_gf.vec.data = self.solution_gf.vec.data - du.vec.data

        


        

    def solve(self, advection_weighting_parameter_list, skip_nonlinear=False, print_condition_number=False, autodiff=False, maxits=10, tol=1e-5, method='pardiso', return_testvalues=False):      

        if self.loaded_from_files:
            print("Unable to solve: this Hydrodynamics object was loaded from files and can only be used for postprocessing")
            return

        # Set up FEM space
        print(f"\nSetting up Finite Element Space for {'linear' if skip_nonlinear else f'{advection_weighting_parameter_list[0]}-non-linear'} simulation with {self.M} vertical modes and {self.imax+1} harmonic\n"
              +f"components (including subtidal). In total, there are {(2*self.M + 1)*(2*self.imax+1)} equations.\n"
              +f"\nAssumptions used:\n\n- Bed boundary condition: no slip\n- Rigid lid assumption\n- Eddy viscosity: constant\n- Density: depth-independent.\n\n")
        
        
        print(f"Total number of free degrees of freedom: {self.nfreedofs}, so ~{self.nfreedofs / self.num_equations} free DOFs per equation.")

        # Set initial guess
        print(f"Setting initial guess\n")

        sol = ngsolve.GridFunction(self.femspace)
        sol.components[2*(self.M)*(2*self.imax+1)].Set(self.seaward_forcing.boundaryCFdict[0], ngsolve.BND)
        for q in range(1, self.imax + 1):
            sol.components[2*(self.M)*(2*self.imax+1) + q].Set(self.seaward_forcing.boundaryCFdict[-q], ngsolve.BND)
            sol.components[2*(self.M)*(2*self.imax+1) + self.imax + q].Set(self.seaward_forcing.boundaryCFdict[q], ngsolve.BND)

        self.solution_gf = sol

        num_continuation_steps = len(advection_weighting_parameter_list)
        # self.epsilon = advection_weighting_parameter_list[-1] # save the value of epsilon for later use

        for i in range(num_continuation_steps):
            print(f"Epsilon = {advection_weighting_parameter_list[i]}\n")
            print(f"Generating weak forms of the PDE system\n")

            self.model_options['advection_epsilon'] = advection_weighting_parameter_list[i]
            self._setup_forms(advection_weighting_parameter_list[i] == 0)

            # Combine bilinear and linear forms because Newton solver only works with a complete bilinear form
            print(f"Solving using Newton-Raphson method with {maxits} iterations max. and error at most {tol}, using the {method.upper()} solver.\n")

            if skip_nonlinear:
                advection_weighting_parameter = 0
            else:
                advection_weighting_parameter = advection_weighting_parameter_list[i]
            
            if return_testvalues:
                resnorm, invtime = self.SolveNewton(advection_weighting_parameter, tol=tol, maxitns=maxits, printing=True, print_cond = print_condition_number, autodiff=autodiff, method=method, return_vals=True)
            else:
                self.SolveNewton(advection_weighting_parameter, tol=tol, maxitns=maxits, printing=True, print_cond = print_condition_number, autodiff=autodiff, method=method, return_vals=False)

            if skip_nonlinear:
                break

        # reorder components in the gridfunction so that they can be worked with more easily
        print(f"Solution process complete.")
        self._restructure_solution()

        if return_testvalues:
            return resnorm, invtime
        

def load_hydrodynamics(name, **kwargs):
    """Creates a Hydrodynamics object from a folder generated by the save-method of the Hydrodynamics object. This object can *only* be used for postprocessing.
    
    Arguments:
        - name:       name of the folder the data may be found in
        
    """
    # options

    f_options = open(f'{name}/options.json', 'r')
    model_options: dict = json.load(f_options)
    
    vertical_basis_name = model_options.pop('vertical_basis_name')

    if vertical_basis_name == 'eigbasis_constantAv':
        vertical_basis = TruncationBasis.eigbasis_constantAv
    else:
        raise ValueError(f"Could not load hydrodynamics object: vertical basis name {model_options['vertical_basis_name']} invalid.")
    
    model_options['advection_influence_matrix'] = np.array(model_options['advection_influence_matrix'])

    f_options.close()

    if abs(model_options['x_scaling'] - 1) > 1e-12 or abs(model_options['y_scaling'] - 1) > 1e-12:
        scaling = True
    else:
        scaling = False
    
    # params

    f_params = open(f'{name}/params.json', 'r')
    params: dict = json.load(f_params)

    sem_order = params.pop('sem_order')
    M = params.pop('M')
    imax = params.pop('imax')
    f_params.close()
    # the remainder of this dict constitutes the constant physical parameters of the simulation
    if scaling:
        time_basis = TruncationBasis.unit_harmonic_time_basis  # only this particular type of Fourier basis is supported
    else:
        time_basis = TruncationBasis.harmonic_time_basis(params['sigma'])

    # mesh

    mesh = ngsolve.Mesh(f'{name}/mesh.vol')
    bfc = generate_bfc(mesh, order=sem_order, method='diffusion', alpha=1)

    # Make Hydrodynamics object
    hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, time_basis, vertical_basis)
    hydro.loaded_from_files = True
    hydro.scaling = scaling
    
    # add spatial parameters
    hydro.spatial_physical_parameters = dict()

    for param in os.listdir(f'{name}/spatial_parameters'):
        filename = os.fsdecode(param)
        param_name = filename[:-4] # ignore file extension
        with open(f'{name}/spatial_parameters/{param_name}.pkl', 'rb') as file:
            param_fh = dill.load(file)

        hydro.spatial_physical_parameters[param_name] = SpatialParameter(param_fh, bfc)

        # oneDfemspace = ngsolve.H1(mesh, order = sem_order)
        # gf = ngsolve.GridFunction(oneDfemspace)
        # mesh_functions.load_basevector(gf.vec.data, f'{name}/spatial_parameters/{filename}')

        # hydro.spatial_physical_parameters[param_name] = gf

    # add constant parameters

    hydro.constant_physical_parameters = params

    # add solution
    hydro.solution_gf = ngsolve.GridFunction(hydro.femspace)
    mesh_functions.load_basevector(hydro.solution_gf.vec.data, f'{name}/solution.npy', **kwargs)

    hydro._restructure_solution()

    return hydro



class RiverineForcing(object):

    def __init__(self, hydro: Hydrodynamics, discharge_amplitude_list, discharge_phase_list, is_constant=True): # Currently, only constant river discharge works
        
        self.discharge_amplitudes = discharge_amplitude_list
        self.discharge_phases = discharge_phase_list
        self.hydro = hydro
        self.is_constant = is_constant

        self.discharge_dict = dict() # Use a dictionary to enable negative indices
        self.Q_vec = dict() # vector (\int_0^T Q h_p dt), p = -imax, ..., imax

        # fill amplitude and phase lists with zeros for unfilled elements unless is_constant == True and create the vector Q_vec

        if not is_constant:
            for _ in range(hydro.imax + 1 - len(discharge_amplitude_list)):
                self.discharge_amplitudes.append(0)
                self.discharge_phases.append(0)

            self.discharge_dict[0] = self.discharge_amplitudes[0]
            self.Q_vec[0] = hydro.time_basis.inner_product(0, 0) * self.discharge_dict[0]
            for i in range(1, hydro.imax + 1):
                self.discharge_dict[i] = self.discharge_amplitudes[i] * ngsolve.cos(self.discharge_phases[i])
                self.discharge_dict[-i] = self.discharge_amplitudes[i] * ngsolve.sin(self.discharge_phases[i])

                self.Q_vec[i] = self.discharge_dict[i] * hydro.time_basis.inner_product(i, i)
                self.Q_vec[-i] = self.discharge_dict[-i] * hydro.time_basis.inner_product(-i, -i)

        else:
            self.discharge_dict[0] = self.discharge_amplitudes[0]
            self.Q_vec[0] = (0.5 / hydro.constant_physical_parameters['sigma']) * self.discharge_dict[0]
        
        # Computation of normal components

        if is_constant and hydro.model_options['density'] == 'depth-independent':
            d1 = [0.5*(1/hydro.constant_physical_parameters['sigma']) * hydro.constant_physical_parameters['g'] * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M)]
            d2 = [hydro.spatial_physical_parameters['H'].cf / (2 * hydro.constant_physical_parameters['sigma']) * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M)]


            C = [0.25 * (1/hydro.constant_physical_parameters['sigma']) * (k+0.5)*(k+0.5) * np.pi * np.pi * \
                 (hydro.constant_physical_parameters['Av'] / (hydro.spatial_physical_parameters['H'].cf*hydro.spatial_physical_parameters['H'].cf)) \
                    for k in range(hydro.M)]
            
            # sum_d1d2 = sum([d1[k]*d2[k] for k in range(hydro.M)])
            sum_d1d2 = sum([d1[k]*d2[k]/C[k] for k in range(hydro.M)])

            # self.normal_alpha = [{0: (-d1[m]/(sum_d1d2*C[m])) * self.Q_vec[0]} for m in range(hydro.M)]
            self.normal_alpha = [{0: (-d1[m]/(sum_d1d2*C[m])) * self.Q_vec[0]} for m in range(hydro.M)]
            self.normal_alpha_boundaryCF = [{0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][0]}, default=0)} for m in range(hydro.M)]
            for m in range(hydro.M):
                for q in range(1, hydro.imax + 1):
                    self.normal_alpha[m][q] = 0
                    self.normal_alpha[m][-q] = 0

                    self.normal_alpha_boundaryCF[m][q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: 0}, default=0)
                    self.normal_alpha_boundaryCF[m][-q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: 0}, default=0)

        elif (not is_constant) and hydro.model_options['density'] == 'depth-independent':

            C = [0.25 * (1/hydro.constant_physical_parameters['sigma']) * (k+0.5)*(k+0.5) * np.pi*np.pi * \
                 (hydro.constant_physical_parameters['Av'] / (hydro.spatial_physical_parameters['H'].cf*hydro.spatial_physical_parameters['H'].cf)) \
                    for k in range(hydro.M)]
            
            d1 = [0.5*(1/hydro.constant_physical_parameters['sigma']) * hydro.constant_physical_parameters['g'] * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M)]
            d2 = [hydro.spatial_physical_parameters['H'].cf / (2 * hydro.constant_physical_parameters['sigma']) * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M)]
            
            c1 = [[1 + 0.25*np.pi*np.pi*q*q*(4/(4*C[k]*C[k]-np.pi**2 * q**2)) for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]
            c2 = [[-0.5*np.pi*q*(4*C[k])/(4*C[k]*C[k] - np.pi**2 * q**2) for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]
            c3 = [[-0.5*np.pi*q/C[k] for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]
            c4 = [[4*C[k] / (4*C[k]*C[k]-np.pi*np.pi*q*q) for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]

            e1 = [-sum([d1[k]*d2[k]*c1[k][q-1] / C[k] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]
            e2 = [-sum([d1[k]*d2[k]*c2[k][q-1] / C[k] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]
            e3 = [-sum([d1[k]*d2[k]*c3[k][q-1] / c4[k][q-1] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]
            e4 = [-sum([d1[k]*d2[k] / c4[k][q-1] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]

            gamma = dict()
            sum_d1d2 = sum([d1[k]*d2[k]/C[k] for k in range(hydro.M)])
            gamma[0] = -self.Q_vec[0] / sum_d1d2

            for q in range(1, hydro.imax + 1):
                gamma[q] = e1[q-1] / (e4[q-1]*e1[q-1] - e3[q-1]) * (self.Q_vec[q] - (e3[q-1]/e1[q-1])*self.Q_vec[-q])
                gamma[-q] = (self.Q_vec[-q] - e2[q-1]*gamma[q]) / e1[q-1]

            self.normal_alpha = [{0: (-d1[m]/(sum_d1d2*C[m])) * self.Q_vec[0]} for m in range(hydro.M)]
            self.normal_alpha_boundaryCF = [{0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][0]}, default=0)} for m in range(hydro.M)]
            for m in range(hydro.M):
                for q in range(1, hydro.imax + 1):
                    self.normal_alpha[m][q] = d1[m]*c3[m][q-1]*gamma[-q] / c4[m][q-1] + d2[m]*gamma[q] / c4[m][q-1]
                    self.normal_alpha[m][-q] = d1[m]*c1[m][q-1]*gamma[-q] / C[m] + d1[m]*c2[m][q-1]*gamma[q] / C[m]

                    self.normal_alpha_boundaryCF[m][q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][q]}, default=0)
                    self.normal_alpha_boundaryCF[m][-q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][-q]}, default=0)
        


class SeawardForcing(object):

    def __init__(self, hydro: Hydrodynamics, amplitude_list, phase_list):
        """List of amplitudes and phases starting at the subtidal component, moving to M2 frequency and moving to M4, M6, ...
        The phase list also starts at the subtidal component, but the first component is never used."""
        self.hydro = hydro

        # Fill amplitudes and phases with zeros in the places where they are not prescribed
        self.amplitudes = amplitude_list
        self.phases = phase_list
        for _ in range(self.hydro.imax + 1 - len(amplitude_list)):
            self.amplitudes.append(0)
            self.phases.append(0)

        self.cfdict = {0: self.amplitudes[0]}
        self.boundaryCFdict = {0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[0]}, default=0)}
        for i in range(1, self.hydro.imax + 1):
            self.cfdict[i] = self.amplitudes[i] * ngsolve.cos(self.phases[i])
            self.cfdict[-i] = self.amplitudes[i] * ngsolve.sin(self.phases[i])

            self.boundaryCFdict[i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[i]}, default=0)
            self.boundaryCFdict[-i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[-i]}, default=0)


# def add_linear_part_to_bilinearform(hydro, a, forcing=True):
#     if hydro.model_options.bed_bc == 'no_slip' and hydro.model_options.density == 'depth-independent' and hydro.model_options.veddy_viscosity_assumption == 'constant' and hydro.model_options.leading_order_surface:
#         add_linear_part_to_bilinearform_NS_DI_EVC_RL(hydro, a, forcing)


# def add_nonlinear_part_to_bilinearform(hydro, a, advection_weighting_parameter):
#     if hydro.model_options.bed_bc == 'no_slip' and hydro.model_options.density == 'depth-independent' and hydro.model_options.veddy_viscosity_assumption == 'constant' and hydro.model_options.leading_order_surface:
#         add_nonlinear_part_to_bilinearform_NS_DI_EVC_RL(hydro, a, advection_weighting_parameter)


# def add_linearised_nonlinear_part_to_bilinearform(hydro, a, alpha0, beta0, gamma0, advection_weighting_parameter):
#     if hydro.model_options.bed_bc == 'no_slip' and hydro.model_options.density == 'depth-independent' and hydro.model_options.veddy_viscosity_assumption == 'constant' and hydro.model_options.leading_order_surface:
#         add_linearised_nonlinear_part_to_bilinearform_NS_DI_EVC_RL(hydro, a, alpha0, beta0, gamma0, advection_weighting_parameter)


# def add_linear_part_to_bilinearform_NS_DI_EVC_RL(hydro: Hydrodynamics, a: ngsolve.BilinearForm, forcing=True):
#     G3 = hydro.vertical_basis.tensor_dict['G3']
#     G4 = hydro.vertical_basis.tensor_dict['G4']
#     G5 = hydro.vertical_basis.tensor_dict['G5']

#     sig = hydro.constant_physical_parameters['sigma']
#     Av = hydro.constant_physical_parameters['Av']
#     f = hydro.constant_physical_parameters['f']
#     g = hydro.constant_physical_parameters['g']

#     H = hydro.spatial_physical_parameters['H'].cf
#     rho = hydro.spatial_physical_parameters['density'].cf
#     rho_x = hydro.spatial_physical_parameters['density'].gradient_cf[0]
#     rho_y = hydro.spatial_physical_parameters['density'].gradient_cf[1]

#     normalalpha = hydro.riverine_forcing.normal_alpha

#     # Depth-integrated continuity equation with homogeneous boundary conditions
#     # if forcing:
#     #     a += (0.5 / sig * hydro.DIC_testfunctions[0] * H * sum([G4(m) * \
#     #              normalalpha[m][0] for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#     #     for r in range(1, hydro.imax + 1):
#     #         a += (0.5 / sig * hydro.DIC_testfunctions[-r] * H * sum([G4(m) * \
#     #              normalalpha[m][-r] for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#     #         a += (0.5 / sig * hydro.DIC_testfunctions[r] * H * sum([G4(m) * \
#     #              normalalpha[m][r] for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#     # r = 0 term
#     a += (-0.5/sig * H * sum([G4(m) * (hydro.alpha_trialfunctions[m][0] * ngsolve.grad(hydro.DIC_testfunctions[0])[0] + 
#                                         hydro.beta_trialfunctions[m][0] * ngsolve.grad(hydro.DIC_testfunctions[0])[1]) for m in range(hydro.M)])) * ngsolve.dx

#     # r != 0-terms
#     for r in range(1, hydro.imax + 1):
#         a += (ngsolve.pi*r*hydro.gamma_trialfunctions[r]*hydro.DIC_testfunctions[-r] - 0.5/sig*H*sum([G4(m) * (
#             hydro.alpha_trialfunctions[m][-r] * ngsolve.grad(hydro.DIC_testfunctions[-r])[0] + 
#             hydro.beta_trialfunctions[m][-r] * ngsolve.grad(hydro.DIC_testfunctions[-r])[1]
#         ) for m in range(hydro.M)])) * ngsolve.dx
#         a += (ngsolve.pi*-r*hydro.gamma_trialfunctions[-r]*hydro.DIC_testfunctions[r] - 0.5/sig*H*sum([G4(m) * (
#             hydro.alpha_trialfunctions[m][r] * ngsolve.grad(hydro.DIC_testfunctions[r])[0] + 
#             hydro.beta_trialfunctions[m][r] * ngsolve.grad(hydro.DIC_testfunctions[r])[1]
#         ) for m in range(hydro.M)])) * ngsolve.dx


#     # Momentum equations
#     for k in range(hydro.M):
#         # Add everything but the advective terms
#         if forcing:
#             a += (0.5*ngsolve.sqrt(2)/sig*G5(k) * H * H * hydro.alpha_testfunctions[k][0] * rho_x / rho) * ngsolve.dx # U-momentum
#             a += (0.5*ngsolve.sqrt(2)/sig*G5(k) * H * H * hydro.beta_testfunctions[k][0] * rho_y / rho) * ngsolve.dx # V-momentum

#         # r = 0 term
#         # U-momentum
#         a += (-0.5/sig * Av * G3(k, k) * hydro.alpha_trialfunctions[k][0]*hydro.alpha_testfunctions[k][0] / H - 
#                 0.25*f/sig * H * hydro.beta_trialfunctions[k][0] * hydro.alpha_testfunctions[k][0] + 
#                 0.5*g*H/sig * G4(k) * ngsolve.grad(hydro.gamma_trialfunctions[0])[0] * hydro.alpha_testfunctions[k][0]) * ngsolve.dx
#         # V-momentum
#         a += (-0.5/sig * Av * G3(k, k) * hydro.beta_trialfunctions[k][0]*hydro.beta_testfunctions[k][0] / H +
#                 0.25*f/sig * H * hydro.alpha_trialfunctions[k][0] * hydro.beta_testfunctions[k][0] + 
#                 0.5*g*H/sig * G4(k) * ngsolve.grad(hydro.gamma_trialfunctions[0])[1] * hydro.beta_testfunctions[k][0]) * ngsolve.dx
        
#         # r != 0-terms
#         for r in range(1, hydro.imax + 1):
#             # U-momentum
#             a += ((0.5*ngsolve.pi*r*hydro.alpha_trialfunctions[k][r] *hydro.alpha_testfunctions[k][-r]- 
#                     0.5/sig*Av*G3(k,k) * hydro.alpha_trialfunctions[k][-r]*hydro.alpha_testfunctions[k][-r] / H - 
#                     0.25*f/sig * H * hydro.beta_trialfunctions[k][-r] * hydro.alpha_testfunctions[k][-r] + 
#                     0.5*g*H/sig * G4(k) * ngsolve.grad(hydro.gamma_trialfunctions[-r])[0] * hydro.alpha_testfunctions[k][-r]) + 
                    
#                     (0.5*ngsolve.pi*-r*hydro.alpha_trialfunctions[k][-r] * hydro.alpha_testfunctions[k][r] - 
#                     0.5/sig*Av*G3(k,k) * hydro.alpha_trialfunctions[k][r]*hydro.alpha_testfunctions[k][r] / H - 
#                     0.25*f/sig * H * hydro.beta_trialfunctions[k][r] * hydro.alpha_testfunctions[k][r] + 
#                     0.5*g*H/sig * G4(k) * ngsolve.grad(hydro.gamma_trialfunctions[r])[0] * hydro.alpha_testfunctions[k][r])) * ngsolve.dx
#             # V-momentum
#             a += ((0.5*ngsolve.pi*r*hydro.beta_trialfunctions[k][r]*hydro.beta_testfunctions[k][-r] - 
#                     0.5/sig*Av*G3(k,k) * hydro.beta_trialfunctions[k][-r]*hydro.beta_testfunctions[k][-r] / H + 
#                     0.25*f/sig * H * hydro.alpha_trialfunctions[k][-r] * hydro.beta_testfunctions[k][-r] + 
#                     0.5*H*g/sig * G4(k) * ngsolve.grad(hydro.gamma_trialfunctions[-r])[1] * hydro.beta_testfunctions[k][-r]) + 
                    
#                     (0.5*ngsolve.pi*-r*hydro.beta_trialfunctions[k][-r]*hydro.beta_testfunctions[k][r] - 
#                     0.5/sig*Av*G3(k,k) * hydro.beta_trialfunctions[k][r]*hydro.beta_testfunctions[k][r] / H +
#                     0.25*f/sig * H * hydro.alpha_trialfunctions[k][r] * hydro.beta_testfunctions[k][r] + 
#                     0.5*g*H/sig * G4(k) * ngsolve.grad(hydro.gamma_trialfunctions[r])[1] * hydro.beta_testfunctions[k][r])) * ngsolve.dx
            

# def add_nonlinear_part_to_bilinearform_NS_DI_EVC_RL(hydro: Hydrodynamics, a: ngsolve.BilinearForm, advection_weighting_parameter=1):
#     G1 = hydro.vertical_basis.tensor_dict['G1']
#     G2 = hydro.vertical_basis.tensor_dict['G2']

#     H3 = hydro.time_basis.tensor_dict['H3']

#     H = hydro.spatial_physical_parameters['H'].cf
    
#     normalalpha = hydro.riverine_forcing.normal_alpha
#     for k in range(hydro.M):
        
#         # add nonlinear part of a
#         # U-momentum
#         a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, 0) * (
#             (-H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][0] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#         ) - H3(p,q,0) * (G1(m,n,k)+G2(m,n,k)) * (
#             (-H * advection_weighting_parameter * hydro.alpha_testfunctions[k][0] * (ngsolve.grad(hydro.alpha_trialfunctions[m][q])[0] * hydro.alpha_trialfunctions[n][p] + 
#                                                     ngsolve.grad(hydro.alpha_trialfunctions[m][q])[1] * hydro.beta_trialfunctions[n][p])) * ngsolve.dx
#             - (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[0] + 
#                                                     hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][0] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#         ) for m in range(hydro.M)]) for n in range(hydro.M)]) for p in range(-hydro.imax, hydro.imax + 1)]) for q in range(-hydro.imax, hydro.imax + 1)])
#         # V-momentum
#         a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, 0) * (
#             (-H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][0] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#         ) - H3(p,q,0) * (G1(m,n,k)+G2(m,n,k)) * (
#             (-H * advection_weighting_parameter * hydro.beta_testfunctions[k][0] * (ngsolve.grad(hydro.beta_trialfunctions[m][q])[0] * hydro.alpha_trialfunctions[n][p] + 
#                                                     ngsolve.grad(hydro.beta_trialfunctions[m][q])[1] * hydro.beta_trialfunctions[n][p])) * ngsolve.dx
#             - (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[0] + 
#                                                     hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][0] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#         ) for m in range(hydro.M)]) for n in range(hydro.M)]) for p in range(-hydro.imax, hydro.imax + 1)]) for q in range(-hydro.imax, hydro.imax + 1)])

#         for r in range(1, hydro.imax + 1):
#             # add nonlinear part of a
#             # U-momentum component -r
#             a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, -r) * (
#                 (-H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][-r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) - H3(p,q,-r) * (G1(m,n,k)+G2(m,n,k)) * (
#                 (-H * advection_weighting_parameter * hydro.alpha_testfunctions[k][-r] * (ngsolve.grad(hydro.alpha_trialfunctions[m][q])[0] * hydro.alpha_trialfunctions[n][p] + 
#                                                         ngsolve.grad(hydro.alpha_trialfunctions[m][q])[1] * hydro.beta_trialfunctions[n][p])) * ngsolve.dx
#             - (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][-r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) for m in range(hydro.M)]) for n in range(hydro.M)]) for p in range(-hydro.imax, hydro.imax + 1)]) for q in range(-hydro.imax, hydro.imax + 1)])
#             # U-momentum component +r
#             a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, r) * (
#                 (-H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) - H3(p,q,r) * (G1(m,n,k)+G2(m,n,k)) * (
#                 (-H * advection_weighting_parameter * hydro.alpha_testfunctions[k][r] * (ngsolve.grad(hydro.alpha_trialfunctions[m][q])[0] * hydro.alpha_trialfunctions[n][p] + 
#                                                         ngsolve.grad(hydro.alpha_trialfunctions[m][q])[1] * hydro.beta_trialfunctions[n][p])) * ngsolve.dx
#             - (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) for m in range(hydro.M)]) for n in range(hydro.M)]) for p in range(-hydro.imax, hydro.imax + 1)]) for q in range(-hydro.imax, hydro.imax + 1)])
#             # V-momentum component -r
#             a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, -r) * (
#                 (-H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][-r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) - H3(p,q,-r) * (G1(m,n,k)+G2(m,n,k)) * (
#                 (-H * advection_weighting_parameter * hydro.beta_testfunctions[k][-r] * (ngsolve.grad(hydro.beta_trialfunctions[m][q])[0] * hydro.alpha_trialfunctions[n][p] + 
#                                                         ngsolve.grad(hydro.beta_trialfunctions[m][q])[1] * hydro.beta_trialfunctions[n][p])) * ngsolve.dx
#             - (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][-r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) for m in range(hydro.M)]) for n in range(hydro.M)]) for p in range(-hydro.imax, hydro.imax + 1)]) for q in range(-hydro.imax, hydro.imax + 1)])
#             # V-momentum component +r
#             a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, r) * (
#                 (-H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) - H3(p,q,r) * (G1(m,n,k)+G2(m,n,k)) * (
#                 (-H * advection_weighting_parameter * hydro.beta_testfunctions[k][r] * (ngsolve.grad(hydro.beta_trialfunctions[m][q])[0] * hydro.alpha_trialfunctions[n][p] + 
#                                                         ngsolve.grad(hydro.beta_trialfunctions[m][q])[1] * hydro.beta_trialfunctions[n][p])) * ngsolve.dx
#             - (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[0] + 
#                                                         hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[1])) * ngsolve.dx
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#             + (H * advection_weighting_parameter * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][r] * (hydro.n[0]*hydro.alpha_trialfunctions[n][p] + hydro.n[1]*hydro.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) for m in range(hydro.M)]) for n in range(hydro.M)]) for p in range(-hydro.imax, hydro.imax + 1)]) for q in range(-hydro.imax, hydro.imax + 1)])

# def add_linearised_nonlinear_part_to_bilinearform_NS_DI_EVC_RL(hydro: Hydrodynamics, a: ngsolve.BilinearForm, alpha0, beta0, gamma0, advection_weighting_parameter=1):
#     G1 = hydro.vertical_basis.tensor_dict['G1']
#     G2 = hydro.vertical_basis.tensor_dict['G2']

#     H3 = hydro.time_basis.tensor_dict['H3']
#     H3_iszero = hydro.time_basis.tensor_dict['H3_iszero']

#     H = hydro.spatial_physical_parameters['H'].cf
    
#     normalalpha = hydro.riverine_forcing.normal_alpha

#     for k in range(hydro.M):
#         for p in range(-hydro.imax, hydro.imax + 1):
#             for q in range(-hydro.imax, hydro.imax + 1):
#                 if H3_iszero(p, q, 0):
#                     continue
#                 else:
#                     # interior domain integration for u-momentum
#                     a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                         -H * alpha0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[0] + 
#                                             hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[1]) - \
#                         H * hydro.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[0] + \
#                                                                 beta0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[1])
#                     ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.dx
#                     a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
#                         -H * alpha0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[0] + 
#                                             hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[1]) - \
#                         H * hydro.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[0] + \
#                                                                 beta0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][0])[1]) - 
#                         H * hydro.alpha_testfunctions[k][0] * (ngsolve.grad(alpha0[m][q])[0] * hydro.alpha_trialfunctions[n][p] + \
#                                                             ngsolve.grad(alpha0[m][q])[1] * hydro.beta_trialfunctions[n][p]) -
#                         H * hydro.alpha_testfunctions[k][0] * (ngsolve.grad(hydro.alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                             ngsolve.grad(hydro.alpha_trialfunctions[m][q])[1] * beta0[n][p])
#                     ) for n in range(hydro.M)])for m in range(hydro.M)]))*ngsolve.dx
#                     # integration over seaward boundary for u-momentum
#                     a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                         H * alpha0[m][q] * hydro.alpha_testfunctions[k][0] * (
#                             hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                         ) + 
#                         H * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][0] * (
#                             alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                         )
#                     ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                     a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
#                         H * alpha0[m][q] * hydro.alpha_testfunctions[k][0] * (
#                             hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                         ) + 
#                         H * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][0] * (
#                             alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                         )
#                     ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                     # interior domain integration for v-momentum
#                     a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                         -H * beta0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[0] + 
#                                             hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[1]) - \
#                         H * hydro.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[0] + \
#                                                                 beta0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[1])
#                     ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.dx
#                     a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
#                         -H * beta0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[0] + 
#                                             hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[1]) - \
#                         H * hydro.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[0] + \
#                                                                 beta0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][0])[1]) - 
#                         H * hydro.beta_testfunctions[k][0] * (ngsolve.grad(beta0[m][q])[0] * hydro.alpha_trialfunctions[n][p] + \
#                                                             ngsolve.grad(beta0[m][q])[1] * hydro.beta_trialfunctions[n][p]) -
#                         H * hydro.beta_testfunctions[k][0] * (ngsolve.grad(hydro.beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                             ngsolve.grad(hydro.beta_trialfunctions[m][q])[1] * beta0[n][p])
#                     ) for n in range(hydro.M)])for m in range(hydro.M)]))*ngsolve.dx
#                     # integration over seaward boundary for v-momentum
#                     a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                         H * beta0[m][q] * hydro.beta_testfunctions[k][0] * (
#                             hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                         ) + 
#                         H * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][0] * (
#                             alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                         )
#                     ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                     a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
#                         H * beta0[m][q] * hydro.beta_testfunctions[k][0] * (
#                             hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                         ) + 
#                         H * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][0] * (
#                             alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                         )
#                     ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            
#                 # terms r != 0
#         for r in range(1, hydro.imax + 1):
#             for p in range(-hydro.imax, hydro.imax + 1):
#                 for q in range(-hydro.imax, hydro.imax + 1):
#                     if H3_iszero(p, q, r):
#                         continue
#                     else:
#                         # terms -r
#                         # interior domain integration for u-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                             -H * alpha0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[1]) - \
#                             H * hydro.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[1])
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.dx
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
#                             -H * alpha0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[1]) - \
#                             H * hydro.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][-r])[1]) - 
#                             H * hydro.alpha_testfunctions[k][-r] * (ngsolve.grad(alpha0[m][q])[0] * hydro.alpha_trialfunctions[n][p] + \
#                                                                 ngsolve.grad(alpha0[m][q])[1] * hydro.beta_trialfunctions[n][p]) -
#                             H * hydro.alpha_testfunctions[k][-r] * (ngsolve.grad(hydro.alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                 ngsolve.grad(hydro.alpha_trialfunctions[m][q])[1] * beta0[n][p])
#                         ) for n in range(hydro.M)])for m in range(hydro.M)]))*ngsolve.dx
#                         # integration over seaward boundary for u-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                             H * alpha0[m][q] * hydro.alpha_testfunctions[k][-r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][-r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
#                             H * alpha0[m][q] * hydro.alpha_testfunctions[k][-r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][-r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         # interior domain integration for v-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                             -H * beta0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[1]) - \
#                             H * hydro.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[1])
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.dx
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
#                             -H * beta0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[1]) - \
#                             H * hydro.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][-r])[1]) - 
#                             H * hydro.beta_testfunctions[k][-r] * (ngsolve.grad(beta0[m][q])[0] * hydro.alpha_trialfunctions[n][p] + \
#                                                                 ngsolve.grad(beta0[m][q])[1] * hydro.beta_trialfunctions[n][p]) -
#                             H * hydro.beta_testfunctions[k][-r] * (ngsolve.grad(hydro.beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                 ngsolve.grad(hydro.beta_trialfunctions[m][q])[1] * beta0[n][p])
#                         ) for n in range(hydro.M)])for m in range(hydro.M)]))*ngsolve.dx
#                         # integration over seaward boundary for v-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                             H * beta0[m][q] * hydro.beta_testfunctions[k][-r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][-r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
#                             H * beta0[m][q] * hydro.beta_testfunctions[k][-r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][-r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])

#                         # terms +r
#                         # interior domain integration for u-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                             -H * alpha0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[1]) - \
#                             H * hydro.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[1])
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.dx
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
#                             -H * alpha0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[1]) - \
#                             H * hydro.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.alpha_testfunctions[k][r])[1]) - 
#                             H * hydro.alpha_testfunctions[k][r] * (ngsolve.grad(alpha0[m][q])[0] * hydro.alpha_trialfunctions[n][p] + \
#                                                                 ngsolve.grad(alpha0[m][q])[1] * hydro.beta_trialfunctions[n][p]) -
#                             H * hydro.alpha_testfunctions[k][r] * (ngsolve.grad(hydro.alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                 ngsolve.grad(hydro.alpha_trialfunctions[m][q])[1] * beta0[n][p])
#                         ) for n in range(hydro.M)])for m in range(hydro.M)]))*ngsolve.dx
#                         # integration over seaward boundary for u-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                             H * alpha0[m][q] * hydro.alpha_testfunctions[k][r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
#                             H * alpha0[m][q] * hydro.alpha_testfunctions[k][r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.alpha_trialfunctions[m][q] * hydro.alpha_testfunctions[k][r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         # interior domain integration for v-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                             -H * beta0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[1]) - \
#                             H * hydro.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[1])
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.dx
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
#                             -H * beta0[m][q] * (hydro.alpha_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[0] + 
#                                                 hydro.beta_trialfunctions[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[1]) - \
#                             H * hydro.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(hydro.beta_testfunctions[k][r])[1]) - 
#                             H * hydro.beta_testfunctions[k][r] * (ngsolve.grad(beta0[m][q])[0] * hydro.alpha_trialfunctions[n][p] + \
#                                                                 ngsolve.grad(beta0[m][q])[1] * hydro.beta_trialfunctions[n][p]) -
#                             H * hydro.beta_testfunctions[k][r] * (ngsolve.grad(hydro.beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                 ngsolve.grad(hydro.beta_trialfunctions[m][q])[1] * beta0[n][p])
#                         ) for n in range(hydro.M)])for m in range(hydro.M)]))*ngsolve.dx
#                         # integration over seaward boundary for v-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                             H * beta0[m][q] * hydro.beta_testfunctions[k][r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
#                             H * beta0[m][q] * hydro.beta_testfunctions[k][r] * (
#                                 hydro.alpha_trialfunctions[n][p] * hydro.n[0] + hydro.beta_trialfunctions[n][p] * hydro.n[1]
#                             ) + 
#                             H * hydro.beta_trialfunctions[m][q] * hydro.beta_testfunctions[k][r] * (
#                                 alpha0[n][p] * hydro.n[0] + beta0[n][p] * hydro.n[1]
#                             )
#                         ) for n in range(hydro.M)]) for m in range(hydro.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])

# def _setup_forms_NS_DI_EVC_RL_linearisation(self, alpha0, beta0, gamma0, advection_weighting_parameter=1):
#     G1 = self.vertical_basis.tensor_dict['G1']
#     G2 = self.vertical_basis.tensor_dict['G2']
#     G3 = self.vertical_basis.tensor_dict['G3']
#     G4 = self.vertical_basis.tensor_dict['G4']

#     H3 = self.time_basis.tensor_dict['H3']
#     H3_iszero = self.time_basis.tensor_dict['H3_iszero']

#     sig = self.constant_physical_parameters['sigma']
#     Av = self.constant_physical_parameters['Av']
#     f = self.constant_physical_parameters['f']
#     g = self.constant_physical_parameters['g']

#     H = self.spatial_physical_parameters['H'].cf

#     a = ngsolve.BilinearForm(self.femspace)

#     # Depth-integrated continuity equation with homogeneous boundary conditions
#     # r = 0 term
#     a += (0.5/sig * H * sum([G4(m) * (self.alpha_trialfunctions[m][0] * ngsolve.grad(self.DIC_testfunctions[0])[0] + 
#                                         self.beta_trialfunctions[m][0] * ngsolve.grad(self.DIC_testfunctions[0])[1]) for m in range(self.M + 1)])) * ngsolve.dx

#     # r != 0-terms
#     for r in range(1, self.imax + 1):
#         a += (ngsolve.pi*r*self.gamma_trialfunctions[r]*self.DIC_testfunctions[-r] + 0.5/sig*H*sum([G4(m) * (
#             self.alpha_trialfunctions[m][-r] * ngsolve.grad(self.DIC_testfunctions[-r])[0] + 
#             self.beta_trialfunctions[m][-r] * ngsolve.grad(self.DIC_testfunctions[-r])[1]
#         ) for m in range(self.M + 1)])) * ngsolve.dx
#         a += (ngsolve.pi*r*self.gamma_trialfunctions[-r]*self.DIC_testfunctions[r] + 0.5/sig*H*sum([G4(m) * (
#             self.alpha_trialfunctions[m][r] * ngsolve.grad(self.DIC_testfunctions[r])[0] + 
#             self.beta_trialfunctions[m][r] * ngsolve.grad(self.DIC_testfunctions[r])[1]
#         ) for m in range(self.M + 1)])) * ngsolve.dx

#     # Momentum equations
    
    
        
#     for k in range(self.M + 1):
#         # Add everything but the advective terms
#         # r = 0 term
#         # U-momentum
#         a += (-0.5/sig * Av * G3(k, k) * self.alpha_trialfunctions[k][0]*self.alpha_testfunctions[k][0] / H - 
#                 0.25*f/sig * H * self.beta_trialfunctions[k][0] * self.alpha_testfunctions[k][0] + 
#                 0.5*g*H/sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[0])[0] * self.alpha_testfunctions[k][0]) * ngsolve.dx
#         # V-momentum
#         a += (-0.5/sig * Av * G3(k, k) * self.beta_trialfunctions[k][0]*self.beta_testfunctions[k][0] / H +
#                 0.25*f/sig * H * self.alpha_trialfunctions[k][0] * self.beta_testfunctions[k][0] + 
#                 0.5*g*H/sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[0])[1] * self.beta_testfunctions[k][0]) * ngsolve.dx
        
#         # r != 0-terms
#         for r in range(1, self.imax + 1):
#             # U-momentum
#             a += ((0.5*ngsolve.pi*abs(r)*self.alpha_trialfunctions[k][r] *self.alpha_testfunctions[k][-r]- 
#                     0.5/sig*Av*G3(k,k) * self.alpha_trialfunctions[k][-r]*self.alpha_testfunctions[k][-r] / H - 
#                     0.25*f/sig * H * self.beta_trialfunctions[k][-r] * self.alpha_testfunctions[k][-r] + 
#                     0.5*g*H/sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[-r])[0] * self.alpha_testfunctions[k][-r]) + 
                    
#                     (0.5*ngsolve.pi*abs(r)*self.alpha_trialfunctions[k][-r] * self.alpha_testfunctions[k][r] - 
#                     0.5/sig*Av*G3(k,k) * self.alpha_trialfunctions[k][r]*self.alpha_testfunctions[k][r] / H - 
#                     0.25*f/sig * H * self.beta_trialfunctions[k][r] * self.alpha_testfunctions[k][r] + 
#                     0.5*g*H/sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[r])[0] * self.alpha_testfunctions[k][r])) * ngsolve.dx
#             # V-momentum
#             a += ((0.5*ngsolve.pi*abs(r)*self.beta_trialfunctions[k][r]*self.beta_testfunctions[k][-r] - 
#                     0.5/sig*Av*G3(k,k) * self.beta_trialfunctions[k][-r]*self.beta_testfunctions[k][-r] / H + 
#                     0.25*f/sig * H * self.alpha_trialfunctions[k][-r] * self.beta_testfunctions[k][-r] + 
#                     0.5*H*g/sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[-r])[1] * self.beta_testfunctions[k][-r]) + 
                    
#                     (0.5*ngsolve.pi*abs(r)*self.beta_trialfunctions[k][-r]*self.beta_testfunctions[k][r] - 
#                     0.5/sig*Av*G3(k,k) * self.beta_trialfunctions[k][r]*self.beta_testfunctions[k][r] / H +
#                     0.25*f/sig * H * self.alpha_trialfunctions[k][r] * self.beta_testfunctions[k][r] + 
#                     0.5*g*H/sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[r])[1] * self.beta_testfunctions[k][r])) * ngsolve.dx
#     if advection_weighting_parameter != 0:
#         # Add linearised advective terms 
#         # terms r = 0
#         for k in range(self.M + 1):
#             for p in range(-self.imax, self.imax + 1):
#                 for q in range(-self.imax, self.imax + 1):
#                     if H3_iszero(p, q, 0):
#                         continue
#                     else:
#                         # interior domain integration for u-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                             -H * alpha0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[0] + 
#                                                 self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[1]) - \
#                             H * self.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[1])
#                         ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.dx
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
#                             -H * alpha0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[0] + 
#                                                 self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[1]) - \
#                             H * self.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[1]) - 
#                             H * self.alpha_testfunctions[k][0] * (ngsolve.grad(alpha0[m][q])[0] * self.alpha_trialfunctions[n][p] + \
#                                                                 ngsolve.grad(alpha0[m][q])[1] * self.beta_trialfunctions[n][p]) -
#                             H * self.alpha_testfunctions[k][0] * (ngsolve.grad(self.alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                 ngsolve.grad(self.alpha_trialfunctions[m][q])[1] * beta0[n][p])
#                         ) for n in range(self.M)])for m in range(self.M)]))*ngsolve.dx
#                         # integration over seaward boundary for u-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                             H * alpha0[m][q] * self.alpha_testfunctions[k][0] * (
#                                 self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                             ) + 
#                             H * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][0] * (
#                                 alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                             )
#                         ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
#                             H * alpha0[m][q] * self.alpha_testfunctions[k][0] * (
#                                 self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                             ) + 
#                             H * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][0] * (
#                                 alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                             )
#                         ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         # interior domain integration for v-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                             -H * beta0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[0] + 
#                                                 self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[1]) - \
#                             H * self.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[1])
#                         ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.dx
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
#                             -H * beta0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[0] + 
#                                                 self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[1]) - \
#                             H * self.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[0] + \
#                                                                     beta0[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[1]) - 
#                             H * self.beta_testfunctions[k][0] * (ngsolve.grad(beta0[m][q])[0] * self.alpha_trialfunctions[n][p] + \
#                                                                 ngsolve.grad(beta0[m][q])[1] * self.beta_trialfunctions[n][p]) -
#                             H * self.beta_testfunctions[k][0] * (ngsolve.grad(self.beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                 ngsolve.grad(self.beta_trialfunctions[m][q])[1] * beta0[n][p])
#                         ) for n in range(self.M)])for m in range(self.M)]))*ngsolve.dx
#                         # integration over seaward boundary for v-momentum
#                         a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
#                             H * beta0[m][q] * self.beta_testfunctions[k][0] * (
#                                 self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                             ) + 
#                             H * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][0] * (
#                                 alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                             )
#                         ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                         a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
#                             H * beta0[m][q] * self.beta_testfunctions[k][0] * (
#                                 self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                             ) + 
#                             H * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][0] * (
#                                 alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                             )
#                         ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        
#             # terms r != 0
#             for r in range(1, self.imax + 1):
#                 for p in range(-self.imax, self.imax + 1):
#                     for q in range(-self.imax, self.imax + 1):
#                         if H3_iszero(p, q, r):
#                             continue
#                         else:
#                             # terms -r
#                             # interior domain integration for u-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                                 -H * alpha0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[1]) - \
#                                 H * self.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[1])
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.dx
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
#                                 -H * alpha0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[1]) - \
#                                 H * self.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[1]) - 
#                                 H * self.alpha_testfunctions[k][-r] * (ngsolve.grad(alpha0[m][q])[0] * self.alpha_trialfunctions[n][p] + \
#                                                                     ngsolve.grad(alpha0[m][q])[1] * self.beta_trialfunctions[n][p]) -
#                                 H * self.alpha_testfunctions[k][-r] * (ngsolve.grad(self.alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                     ngsolve.grad(self.alpha_trialfunctions[m][q])[1] * beta0[n][p])
#                             ) for n in range(self.M)])for m in range(self.M)]))*ngsolve.dx
#                             # integration over seaward boundary for u-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                                 H * alpha0[m][q] * self.alpha_testfunctions[k][-r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][-r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
#                                 H * alpha0[m][q] * self.alpha_testfunctions[k][-r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][-r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                             # interior domain integration for v-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                                 -H * beta0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[1]) - \
#                                 H * self.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[1])
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.dx
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
#                                 -H * beta0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[1]) - \
#                                 H * self.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[1]) - 
#                                 H * self.beta_testfunctions[k][-r] * (ngsolve.grad(beta0[m][q])[0] * self.alpha_trialfunctions[n][p] + \
#                                                                     ngsolve.grad(beta0[m][q])[1] * self.beta_trialfunctions[n][p]) -
#                                 H * self.beta_testfunctions[k][-r] * (ngsolve.grad(self.beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                     ngsolve.grad(self.beta_trialfunctions[m][q])[1] * beta0[n][p])
#                             ) for n in range(self.M)])for m in range(self.M)]))*ngsolve.dx
#                             # integration over seaward boundary for v-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
#                                 H * beta0[m][q] * self.beta_testfunctions[k][-r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][-r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
#                                 H * beta0[m][q] * self.beta_testfunctions[k][-r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][-r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])

#                             # terms +r
#                             # interior domain integration for u-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                                 -H * alpha0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[1]) - \
#                                 H * self.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[1])
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.dx
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
#                                 -H * alpha0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[1]) - \
#                                 H * self.alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[1]) - 
#                                 H * self.alpha_testfunctions[k][r] * (ngsolve.grad(alpha0[m][q])[0] * self.alpha_trialfunctions[n][p] + \
#                                                                     ngsolve.grad(alpha0[m][q])[1] * self.beta_trialfunctions[n][p]) -
#                                 H * self.alpha_testfunctions[k][r] * (ngsolve.grad(self.alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                     ngsolve.grad(self.alpha_trialfunctions[m][q])[1] * beta0[n][p])
#                             ) for n in range(self.M)])for m in range(self.M)]))*ngsolve.dx
#                             # integration over seaward boundary for u-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                                 H * alpha0[m][q] * self.alpha_testfunctions[k][r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
#                                 H * alpha0[m][q] * self.alpha_testfunctions[k][r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                             # interior domain integration for v-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                                 -H * beta0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[1]) - \
#                                 H * self.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[1])
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.dx
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
#                                 -H * beta0[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[0] + 
#                                                     self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[1]) - \
#                                 H * self.beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[0] + \
#                                                                         beta0[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[1]) - 
#                                 H * self.beta_testfunctions[k][r] * (ngsolve.grad(beta0[m][q])[0] * self.alpha_trialfunctions[n][p] + \
#                                                                     ngsolve.grad(beta0[m][q])[1] * self.beta_trialfunctions[n][p]) -
#                                 H * self.beta_testfunctions[k][r] * (ngsolve.grad(self.beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
#                                                                     ngsolve.grad(self.beta_trialfunctions[m][q])[1] * beta0[n][p])
#                             ) for n in range(self.M)])for m in range(self.M)]))*ngsolve.dx
#                             # integration over seaward boundary for v-momentum
#                             a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
#                                 H * beta0[m][q] * self.beta_testfunctions[k][r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                             a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
#                                 H * beta0[m][q] * self.beta_testfunctions[k][r] * (
#                                     self.alpha_trialfunctions[n][p] * self.n[0] + self.beta_trialfunctions[n][p] * self.n[1]
#                                 ) + 
#                                 H * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][r] * (
#                                     alpha0[n][p] * self.n[0] + beta0[n][p] * self.n[1]
#                                 )
#                             ) for n in range(self.M)]) for m in range(self.M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])

#     return a
# def _setup_forms_NS_DI_EVC_RL(self, advection_weighting_parameter=ngsolve.Parameter(1)):
#     a = ngsolve.BilinearForm(self.femspace)
#     # linform = ngsolve.LinearForm(self.femspace)

#     G1 = self.vertical_basis.tensor_dict['G1']
#     G2 = self.vertical_basis.tensor_dict['G2']
#     G3 = self.vertical_basis.tensor_dict['G3']
#     G4 = self.vertical_basis.tensor_dict['G4']
#     G5 = self.vertical_basis.tensor_dict['G5']

#     H3 = self.time_basis.tensor_dict['H3']

#     sig = self.constant_physical_parameters['sigma']
#     Av = self.constant_physical_parameters['Av']
#     f = self.constant_physical_parameters['f']
#     g = self.constant_physical_parameters['g']

#     H = self.spatial_physical_parameters['H'].cf
#     rho_x = self.spatial_physical_parameters['density'].gradient_cf[0]
#     rho_y = self.spatial_physical_parameters['density'].gradient_cf[1]
#     normalalpha = self.riverine_forcing.normal_alpha
    
#     start = timeit.default_timer()
#     # add forms for DIC-equation
#     # linform += (-0.5 / sig * self.DIC_testfunctions[0] * H * sum([G4(m) * \
#     #             normalalpha[m][0] for m in range(self.M + 1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER]) 
#     a += (-0.5 / sig * self.DIC_testfunctions[0] * H * sum([G4(m) * \
#                 normalalpha[m][0] for m in range(self.M + 1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER]) 
    
#     a += (-0.5 / sig * H * sum([G4(m)*(self.alpha_trialfunctions[m][0] * ngsolve.grad(self.DIC_testfunctions[0])[0] + \
#             self.beta_trialfunctions[m][0] * ngsolve.grad(self.DIC_testfunctions[0])[1]) for m in range(self.M + 1)])) * ngsolve.dx
    
#     for r in range(1, self.imax + 1):
#         a += (-0.5/sig*self.DIC_testfunctions[-r]*H*sum([G4(m)*normalalpha[m][-r] for m in range(self.M)]))*ngsolve.ds(BOUNDARY_DICT[RIVER])
#         a += (-0.5/sig*self.DIC_testfunctions[r]*H*sum([G4(m)*normalalpha[m][r] for m in range(self.M)]))*ngsolve.ds(BOUNDARY_DICT[RIVER])
    
#         a += (ngsolve.pi * r * self.gamma_trialfunctions[r] * self.DIC_testfunctions[-r] - \
#             0.5/sig*H*sum([G4(m)*(self.alpha_trialfunctions[m][-r]*ngsolve.grad(self.DIC_testfunctions[-r])[0] + \
#             self.beta_trialfunctions[m][-r]*ngsolve.grad(self.DIC_testfunctions[-r])[1]) for m in range(self.M + 1)])) * ngsolve.dx
#         a += (ngsolve.pi * r * self.gamma_trialfunctions[-r] * self.DIC_testfunctions[r] - \
#             0.5/sig*H*sum([G4(m)*(self.alpha_trialfunctions[m][r]*ngsolve.grad(self.DIC_testfunctions[r])[0] + \
#             self.beta_trialfunctions[m][r]*ngsolve.grad(self.DIC_testfunctions[r])[1]) for m in range(self.M + 1)])) * ngsolve.dx
#     forms_time_linear = timeit.default_timer() - start
#     # add forms for momentum equations
#     forms_time_nonlinear = 0
#     for k in range(self.M + 1):
        
#         # Components r = 0

#         # linform += (0.5 / sig + G5(k) * H * H * self.alpha_testfunctions[k][0] * rho_x) * ngsolve.dx # U-momentum
#         # linform += (0.5 / sig + G5(k) * H * H * self.beta_testfunctions[k][0] * rho_y) * ngsolve.dx # V-momentum
#         linear_start = timeit.default_timer()
#         # Baroclinic forcing
#         a += (-0.5*ngsolve.sqrt(2)/sig*G5(k) * H * H * self.alpha_testfunctions[k][0] * rho_x) * ngsolve.dx # U-momentum
#         a += (-0.5*ngsolve.sqrt(2)/sig*G5(k) * H * H * self.beta_testfunctions[k][0] * rho_y) * ngsolve.dx # V-momentum

#         # add actually bilinear part of a
#         # U-momentum
#         a += (-0.5 * Av / sig * G3(k, k) / H * self.alpha_trialfunctions[k][0] * self.alpha_testfunctions[k][0]) * ngsolve.dx - \
#                 (0.25 * f / sig * H * self.beta_trialfunctions[k][0] * self.alpha_testfunctions[k][0]) * ngsolve.dx + \
#                 (0.5 * g*H / sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[0])[0] * self.alpha_testfunctions[k][0]) * ngsolve.dx
#         # V-momentum
#         a += (-0.5 * Av / sig * G3(k, k) / H * self.beta_trialfunctions[k][0] * self.beta_testfunctions[k][0]) * ngsolve.dx + \
#                 (0.25 * f / sig * H * self.alpha_trialfunctions[k][0] * self.beta_testfunctions[k][0]) * ngsolve.dx + \
#                 (0.5 * g *H/ sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[0])[1] * self.beta_testfunctions[k][0]) * ngsolve.dx
#         forms_time_linear += timeit.default_timer() - linear_start

#         nonlinear_start = timeit.default_timer()
#         if advection_weighting_parameter != 0:
#             # add nonlinear part of a
#             # U-momentum
#             a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, 0) * (
#                 (-H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][0] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) - H3(p,q,0) * (G1(m,n,k)+G2(m,n,k)) * (
#                 (-H * advection_weighting_parameter * self.alpha_testfunctions[k][0] * (ngsolve.grad(self.alpha_trialfunctions[m][q])[0] * self.alpha_trialfunctions[n][p] + 
#                                                         ngsolve.grad(self.alpha_trialfunctions[m][q])[1] * self.beta_trialfunctions[n][p])) * ngsolve.dx
#                 - (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[0] + 
#                                                         self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][0])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][0] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) for m in range(self.M + 1)]) for n in range(self.M + 1)]) for p in range(-self.imax, self.imax + 1)]) for q in range(-self.imax, self.imax + 1)])
#             # V-momentum
#             a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, 0) * (
#                 (-H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][0] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) - H3(p,q,0) * (G1(m,n,k)+G2(m,n,k)) * (
#                 (-H * advection_weighting_parameter * self.beta_testfunctions[k][0] * (ngsolve.grad(self.beta_trialfunctions[m][q])[0] * self.alpha_trialfunctions[n][p] + 
#                                                         ngsolve.grad(self.beta_trialfunctions[m][q])[1] * self.beta_trialfunctions[n][p])) * ngsolve.dx
#                 - (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[0] + 
#                                                         self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][0])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][0] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#             ) for m in range(self.M + 1)]) for n in range(self.M + 1)]) for p in range(-self.imax, self.imax + 1)]) for q in range(-self.imax, self.imax + 1)])

#         # Components r != 0
#         forms_time_nonlinear += timeit.default_timer() - nonlinear_start

#         for r in range(1, self.imax + 1):
#             linear_start = timeit.default_timer()
#             # add actually bilinear part of a
#             # U-momentum component -r
#             a += (-0.5 * Av / sig * G3(k,k) / H * self.alpha_trialfunctions[k][-r] * self.alpha_testfunctions[k][-r]) * ngsolve.dx - \
#                 (0.25 * f / sig * H * self.beta_trialfunctions[k][-r] * self.alpha_testfunctions[k][-r]) * ngsolve.dx + \
#                 (0.5 * H*g / sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[-r])[0] * self.alpha_testfunctions[k][-r]) * ngsolve.dx + \
#                 (0.5 * ngsolve.pi * r * self.alpha_trialfunctions[k][r] * self.alpha_testfunctions[k][-r]) * ngsolve.dx
#             # U-momentum component +r
#             a += (-0.5 * Av / sig * G3(k,k) / H * self.alpha_trialfunctions[k][r] * self.alpha_testfunctions[k][r]) * ngsolve.dx - \
#                 (0.25 * f / sig * H * self.beta_trialfunctions[k][r] * self.alpha_testfunctions[k][r]) * ngsolve.dx + \
#                 (0.5 * g *H/ sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[r])[0] * self.alpha_testfunctions[k][r]) * ngsolve.dx + \
#                 (0.5 * ngsolve.pi * r * self.alpha_trialfunctions[k][-r] * self.alpha_testfunctions[k][r]) * ngsolve.dx
#             # V-momentum component -r
#             a += (-0.5 * Av / sig * G3(k,k) / H * self.beta_trialfunctions[k][-r] * self.beta_testfunctions[k][-r]) * ngsolve.dx + \
#                 (0.25 * f / sig * H * self.alpha_trialfunctions[k][-r] * self.beta_testfunctions[k][-r]) * ngsolve.dx + \
#                 (0.5 * g *H/ sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[-r])[1] * self.beta_testfunctions[k][-r]) * ngsolve.dx + \
#                 (0.5 * ngsolve.pi * r * self.beta_trialfunctions[k][r] * self.beta_testfunctions[k][-r]) * ngsolve.dx
#             # V-momentum component +r
#             a += (-0.5 * Av / sig * G3(k,k) / H * self.beta_trialfunctions[k][r] * self.beta_testfunctions[k][r]) * ngsolve.dx + \
#                 (0.25 * f / sig * H * self.alpha_trialfunctions[k][r] * self.beta_testfunctions[k][r]) * ngsolve.dx + \
#                 (0.5 * g *H/ sig * G4(k) * ngsolve.grad(self.gamma_trialfunctions[r])[1] * self.beta_testfunctions[k][r]) * ngsolve.dx + \
#                 (0.5 * ngsolve.pi * r * self.beta_trialfunctions[k][-r] * self.beta_testfunctions[k][r]) * ngsolve.dx
#             forms_time_linear += timeit.default_timer() - linear_start
#             nonlinear_start = timeit.default_timer()
#             if advection_weighting_parameter != 0:
#                 # add nonlinear part of a
#                 # U-momentum component -r
#                 a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, -r) * (
#                     (-H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][-r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) - H3(p,q,-r) * (G1(m,n,k)+G2(m,n,k)) * (
#                     (-H * advection_weighting_parameter * self.alpha_testfunctions[k][-r] * (ngsolve.grad(self.alpha_trialfunctions[m][q])[0] * self.alpha_trialfunctions[n][p] + 
#                                                             ngsolve.grad(self.alpha_trialfunctions[m][q])[1] * self.beta_trialfunctions[n][p])) * ngsolve.dx
#                 - (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][-r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) for m in range(self.M + 1)]) for n in range(self.M + 1)]) for p in range(-self.imax, self.imax + 1)]) for q in range(-self.imax, self.imax + 1)])
#                 # U-momentum component +r
#                 a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, r) * (
#                     (-H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][-r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) - H3(p,q,r) * (G1(m,n,k)+G2(m,n,k)) * (
#                     (-H * advection_weighting_parameter * self.alpha_testfunctions[k][r] * (ngsolve.grad(self.alpha_trialfunctions[m][q])[0] * self.alpha_trialfunctions[n][p] + 
#                                                             ngsolve.grad(self.alpha_trialfunctions[m][q])[1] * self.beta_trialfunctions[n][p])) * ngsolve.dx
#                 - (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.alpha_testfunctions[k][r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.alpha_trialfunctions[m][q] * self.alpha_testfunctions[k][r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) for m in range(self.M + 1)]) for n in range(self.M + 1)]) for p in range(-self.imax, self.imax + 1)]) for q in range(-self.imax, self.imax + 1)])
#                 # V-momentum component -r
#                 a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, -r) * (
#                     (-H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][-r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) - H3(p,q,-r) * (G1(m,n,k)+G2(m,n,k)) * (
#                     (-H * advection_weighting_parameter * self.beta_testfunctions[k][-r] * (ngsolve.grad(self.beta_trialfunctions[m][q])[0] * self.alpha_trialfunctions[n][p] + 
#                                                             ngsolve.grad(self.beta_trialfunctions[m][q])[1] * self.beta_trialfunctions[n][p])) * ngsolve.dx
#                 - (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][-r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][-r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) for m in range(self.M + 1)]) for n in range(self.M + 1)]) for p in range(-self.imax, self.imax + 1)]) for q in range(-self.imax, self.imax + 1)])
#                 # V-momentum component +r
#                 a += sum([sum([sum([sum([G1(m, n, k) * H3(p, q, r) * (
#                     (-H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) - H3(p,q,r) * (G1(m,n,k)+G2(m,n,k)) * (
#                     (-H * advection_weighting_parameter * self.beta_testfunctions[k][r] * (ngsolve.grad(self.beta_trialfunctions[m][q])[0] * self.alpha_trialfunctions[n][p] + 
#                                                             ngsolve.grad(self.beta_trialfunctions[m][q])[1] * self.beta_trialfunctions[n][p])) * ngsolve.dx
#                 - (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * (self.alpha_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[0] + 
#                                                             self.beta_trialfunctions[n][p] * ngsolve.grad(self.beta_testfunctions[k][r])[1])) * ngsolve.dx
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
#                 + (H * advection_weighting_parameter * self.beta_trialfunctions[m][q] * self.beta_testfunctions[k][r] * (self.n[0]*self.alpha_trialfunctions[n][p] + self.n[1]*self.beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
#                 ) for m in range(self.M + 1)]) for n in range(self.M + 1)]) for p in range(-self.imax, self.imax + 1)]) for q in range(-self.imax, self.imax + 1)])
#             forms_time_nonlinear += timeit.default_timer() - nonlinear_start
#     self.total_bilinearform = a
#     # self.total_linearform = linform
#     return forms_time_linear, forms_time_nonlinear

