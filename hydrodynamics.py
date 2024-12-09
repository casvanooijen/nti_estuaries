import numpy as np
import os
import json
import cloudpickle
import ngsolve
import matplotlib.pyplot as plt

import truncationbasis
from geometry.create_geometry import parametric_geometry, RIVER, SEA, WALL, WALLUP, WALLDOWN, BOUNDARY_DICT
from geometry.meshing import generate_mesh
from boundary_fitted_coordinates import generate_bfc
from spatial_parameter import SpatialParameter

import define_weak_forms as weakforms
import mesh_functions


def ngsolve_tanh(argument):
    return ngsolve.sinh(argument) / ngsolve.cosh(argument)


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


def select_model_options(bed_bc:str = 'no_slip', surface_in_sigma:bool = True, veddy_viscosity_assumption:str = 'constant', horizontal_diffusion: bool = True, density:str = 'depth-independent',
                 advection_epsilon:float = 1, advection_influence_matrix: np.ndarray = None, x_scaling: float = 1., y_scaling: float = 1, mesh_generation_method='unstructured',
                 element_type:str = 'taylor-hood'):
    
    """
    
    Returns a dictionary of all the available model options. Safer than manually creating this dictionary.
    
    Arguments: ('...'  means that there will possibly be future options added)
        
        - bed_bc:                                   indicates what type of boundary condition is used at the river bed ('no_slip' or 'partial_slip');
        - surface_in_sigma (bool):                  flag to indicate whether non-linear effects stemming from presence of the surface in the sigma-coordinates are included;
        - veddy_viscosity_assumption:               structure of the vertical eddy viscosity parameter ('constant' or 'depth-scaled&constantprofile' or ...);
        - horizontal_diffusion (bool):              flag to indicate whether horizontal eddy viscosity terms should be incorporated (this changes the order of the equations, the boundary conditions (and FESpace), and the structure of the weak forms);
        - density:                                  indicates what type of water density field is used ('depth-independent' or ...);
        - advection_epsilon (float):                scalar by which the advective terms in the momentum equations are multiplied; if set to zero, advective terms are skipped;     
                                                    if set to one, advective terms are fully included;
        - advection_influence_matrix (np.ndarray):  (imax+1) x (imax+1) - boolean matrix where element (i,j) indicates whether constituent i is influenced by constituent j through momentum advection (if possible);
                                                    more precisely, in the equations for constituent i, any product of constituents that includes constituent j will not be present in the advective terms
                                                    if element (i, j) is False, even if that product *should* physically be present;    
        - x_scaling (float):                        factor [m] by which the input geometry should be scaled in the x-direction; this variable adds scaling factors in the equations to compensate for this; default = 1
        - y_scaling (float):                        factor [m] by which the input geometry should be scaled in the y-direction; default = 1;
        - mesh_generation_method (str):             method by which the mesh is generated ('unstructured', 'structured_quads', 'structured_tri', 'manual');
        - element_type (str):                       indicates what type of finite elements are used ('naive', 'taylor-hood', 'MINI'); the MINI element is not available for quadrilateral meshes.
        
        """
    
    if bed_bc == 'partial_slip' and veddy_viscosity_assumption == 'constant':
        raise ValueError("Partial-slip condition and constant vertical eddy viscosity are incompatible")
    
    if element_type == 'MINI' and mesh_generation_method == 'structured_quads':
        raise ValueError("Cannot use the MINI element for quadrilateral mesh. Please use 'taylor-hood' instead.")

    options = {
            'bed_bc': bed_bc,
            'surface_in_sigma': surface_in_sigma,
            'veddy_viscosity_assumption': veddy_viscosity_assumption,
            'horizontal_diffusion': horizontal_diffusion,
            'density': density,
            'advection_epsilon': advection_epsilon,
            'advection_influence_matrix': advection_influence_matrix, # the validity of this matrix is checked when imax is know, i.e. when the hydrodynamics object is initialised
            'x_scaling': x_scaling,
            'y_scaling': y_scaling,
            'mesh_generation_method': mesh_generation_method,
            'element_type': element_type 
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
        - time_basis (truncationbasis):                         harmonic time basis;
        - vertical_basis (truncationbasis):                     vertical basis;
        - vertical_basis_name (str):                            name of the vertical basis if it is a default one from truncationbasis.py
        - constant_physical_parameters (dict):                  dictionary containing values of constant physical parameters;
        - spatial_physical_parameters (dict):                   dictionary containing spatially varying physical parameters, such as
                                                                bathymetry, in the form of SpatialParameter objects;
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
                 boundary_partition_dict=None, boundary_maxh_dict=None, geometrycurves=None, maxh_global=None, num_cells=None, imin=0):
        
        self.mesh = mesh
        self.model_options = model_options
        self.imax = imax
        if isinstance(imin, int) and imin >= 0:
            self.imin = imin
        else:
            raise ValueError(f"Provided minimum constituent {imin} invalid; must be a positive integer.")
        self.M = M
        self.num_equations = (2*M+1)*(2*imax + 1)
        self.order = order
        
        self.time_basis = truncationbasis.unit_harmonic_time_basis
        if self.model_options['bed_bc'] == 'no_slip':
            self.vertical_basis = truncationbasis.eigbasis_constantAv

        # If a partial-slip boundary condition (bed_bc) is chosen, the basis depends on the parameters, and is hence initialised in the method set_constant_physical_parameters
        
        self.constant_physical_parameters = dict()
        self.spatial_physical_parameters = dict()

        if geometrycurves is not None:
            self.geometrycurves = geometrycurves
        else:
            raise ValueError("Please submit geometrycurves to hydrodynamics object.")
        
        if maxh_global is not None and num_cells is None:
            self.maxh = maxh_global
        elif maxh_global is None and num_cells is not None:
            self.num_cells = num_cells
            self.maxh = max(1/num_cells[0], 1/num_cells[1])
        else:
            raise ValueError("Please submit either maxh_global (unstructured mesh) or num_cells (structured mesh); not both")

        if boundary_partition_dict is None:
            self.boundary_partition_dict = {RIVER:[0,1],SEA:[0,1],WALLUP:[0,1],WALLDOWN:[0,1]}
        else:
            self.boundary_partition_dict = boundary_partition_dict

        if boundary_maxh_dict is None:
            self.boundary_maxh_dict = {RIVER:[self.maxh], SEA:[self.maxh], WALLUP: [self.maxh], WALLDOWN: [self.maxh]}
        else:
            self.boundary_maxh_dict = boundary_maxh_dict
        
    
        # check/generate advection_influence_matrix

        if self.model_options['advection_influence_matrix'] is None:
            self.model_options['advection_influence_matrix'] = np.full((self.imax + 1 - self.imin, self.imax + 1 - self.imin), True) # in this case, every constituent affects every other constituent through advection, as would be physical
        elif self.model_options['advection_influence_matrix'].shape != (self.imax+1-self.imin, self.imax+1-self.imin):
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
        if self.model_options['horizontal_diffusion']: # These boundary conditions assume a rectangular estuary with seaward boundary at x=0 and river boundary at x=L
            U = ngsolve.H1(self.mesh, order= 2 if (self.model_options['element_type'] == 'taylor-hood' and self.order==1) else self.order, dirichlet=f"{BOUNDARY_DICT[RIVER]}")  # make sure that zero-th order is not used for the free surface
            V = ngsolve.H1(self.mesh, order= 2 if (self.model_options['element_type'] == 'taylor-hood' and self.order==1) else self.order, dirichlet=f"{BOUNDARY_DICT[WALLUP]}|{BOUNDARY_DICT[WALLDOWN]}")

            # add interior bubble functions if MINI-elements are used
            if self.model_options['element_type'] == 'MINI':
                U.SetOrder(ngsolve.TRIG, 3 if self.order == 1 else self.order + 1)
                V.SetOrder(ngsolve.TRIG, 3 if self.order == 1 else self.order + 1)

                U.Update()
                V.Update()

            # define Z-space with order one less than velocity space in case of Taylor-Hood elements or MINI (k>1) elements
            if ((self.model_options['element_type'] == 'taylor-hood') or (self.model_options['element_type'] == 'MINI')) and self.order > 1:
                Z = ngsolve.H1(self.mesh, order=self.order - 1, dirichlet=BOUNDARY_DICT[SEA])
            else:
                Z = ngsolve.H1(self.mesh, order=self.order, dirichlet=BOUNDARY_DICT[SEA])
            
            if self.imin == 0:
                list_of_spaces = [U for _ in range(self.M*(2*self.imax + 1))]
                for _ in range(self.M*(2*self.imax + 1)):
                    list_of_spaces.append(V)
                for _ in range(2*self.imax + 1):
                    list_of_spaces.append(Z)
            else:
                list_of_spaces = [U for _ in range(self.M * (2*(self.imax - self.imin + 1)))]
                for _ in range(self.M*(2*self.imax - self.imin + 1)):
                    list_of_spaces.append(V)
                for _ in range(2*(self.imax - self.imin + 1)):
                    list_of_spaces.append(Z)

            X = ngsolve.FESpace(list_of_spaces)
            self.femspace = X
        else:
            U = ngsolve.H1(self.mesh, order=self.order) 

            # add interior bubble functions if MINI-elements are used
            if self.model_options['element_type'] == 'MINI':
                U.SetOrder(ngsolve.TRIG, 3 if self.order == 1 else self.order + 1)
                U.Update()

            # define Z-space with order one less than velocity space in case of Taylor-Hood elements or MINI (k>1) elements
            if ((self.model_options['element_type'] == 'taylor-hood') or (self.model_options['element_type'] == 'MINI')) and self.order > 1:
                G = ngsolve.H1(self.mesh, order=self.order - 1, dirichlet=BOUNDARY_DICT[SEA])
            else:
                G = ngsolve.H1(self.mesh, order=self.order, dirichlet=BOUNDARY_DICT[SEA])

            if self.imin == 0:
                list_of_spaces = [U for _ in range(2*self.M*(2*self.imax + 1))]
                for _ in range(2*self.imax+1):
                    list_of_spaces.append(G)
            else:
                list_of_spaces = [U for _ in range(2*self.M*(2*(self.imax - self.imin + 1)))]
                for _ in range(2*(self.imax - self.imin + 1)):
                    list_of_spaces.append(G)

            X = ngsolve.FESpace(list_of_spaces) # tensor product of all spaces
            self.femspace = X

        




    def _get_normalvec(self):
        X = self.femspace
        self.n = ngsolve.specialcf.normal(2)
    

    def _setup_TnT(self):


        """Sorts the ngsolve Trial and Test functions into intuitive dictionaries"""

        trialtuple = self.femspace.TrialFunction()
        testtuple = self.femspace.TestFunction()

        if self.imin == 0:
            num_time_components = 2 * self.imax + 1
        else:
            num_time_components = 2 * (self.imax - self.imin + 1)

        alpha_trialfunctions = [dict() for _ in range(self.M)]
        umom_testfunctions = [dict() for _ in range(self.M)] # test functions for momentum equation u

        beta_trialfunctions = [dict() for _ in range(self.M)]
        vmom_testfunctions = [dict() for _ in range(self.M)] # test functions for momentum equation v

        gamma_trialfunctions = dict()
        DIC_testfunctions = dict() # test functions for Depth-Integrated Continuity equation

        for m in range(self.M):
            if self.imin == 0: # add residual component only if imin == 0
                alpha_trialfunctions[m][0] = trialtuple[m * num_time_components]
                umom_testfunctions[m][0] = testtuple[m * num_time_components]

                beta_trialfunctions[m][0] = trialtuple[(self.M + m) * num_time_components]
                vmom_testfunctions[m][0] = testtuple[(self.M + m) * num_time_components]
            for q in range(1, self.imax + 1):
                alpha_trialfunctions[m][-q] = trialtuple[m * num_time_components + q]
                alpha_trialfunctions[m][q] = trialtuple[m * num_time_components + self.imax + q]

                umom_testfunctions[m][-q] = testtuple[m * num_time_components + q]
                umom_testfunctions[m][q] = testtuple[m * num_time_components + self.imax + q]

                beta_trialfunctions[m][-q] = trialtuple[(self.M + m) * num_time_components + q]
                beta_trialfunctions[m][q] = trialtuple[(self.M + m) * num_time_components + self.imax + q]

                vmom_testfunctions[m][-q] = testtuple[(self.M + m) * num_time_components + q]
                vmom_testfunctions[m][q] = testtuple[(self.M + m) * num_time_components + self.imax + q]
        
        gamma_trialfunctions[0] = trialtuple[2*(self.M)*num_time_components]
        DIC_testfunctions[0] = testtuple[2*(self.M)*num_time_components]

        for q in range(1, self.imax + 1):
            gamma_trialfunctions[-q] = trialtuple[2*(self.M)*num_time_components + q]
            gamma_trialfunctions[q] = trialtuple[2*(self.M)*num_time_components + self.imax + q]

            DIC_testfunctions[-q] = testtuple[2*(self.M)*num_time_components + q]
            DIC_testfunctions[q] = testtuple[2*(self.M)*num_time_components + self.imax + q]

        self.alpha_trialfunctions = alpha_trialfunctions
        self.umom_testfunctions = umom_testfunctions
        self.beta_trialfunctions = beta_trialfunctions
        self.vmom_testfunctions = vmom_testfunctions
        self.gamma_trialfunctions = gamma_trialfunctions
        self.DIC_testfunctions = DIC_testfunctions


    def setup_forms(self, skip_nonlinear=False):

        self._get_normalvec()

        a_total = ngsolve.BilinearForm(self.femspace)

        if self.scaling:
            weakforms.add_weak_form(a_total, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
                                self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
                                self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis,
                                self.riverine_forcing.normal_alpha, only_linear=skip_nonlinear)
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


    def restructure_solution(self):

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


    def save(self, name, **kwargs):
        """Saves the hydrodynamics object. Only possible if the Fourier/vertical bases are chosen from the predefined 
        bases in truncationbasis.py. The folder contains:
        
        - options.txt:          contains the model options of the ModelOptions object, including which Fourier/vertical bases were used;
        - params.txt:           contains the SEM expansion basis order, M, imax and the constant physical parameters;
        - mesh.vol:             file that can be read by NGSolve to regenerate your mesh;
        - spatial_parameters:   folder that contain function handles of spatial parameters;
        - solution.txt          file that contains the solution GridFunction;

        """
        os.makedirs(name, exist_ok=True)

        # model options

        options = {}
        options.update(self.model_options)

        options['advection_influence_matrix'] = options['advection_influence_matrix'].tolist()

        with open(f"{name}/options.json", 'x') as f_options:
            json.dump(options, f_options, indent=4)

        # constant parameters
        if options['mesh_generation_method'] == 'unstructured':
            params = {'sem_order': self.order, 'M': self.M, 'imax': self.imax, 'maxh': self.maxh}
        else:
            params = {'sem_order': self.order, 'M': self.M, 'imax': self.imax, 'num_cells': self.num_cells}
        params.update(self.constant_physical_parameters)

        with open(f"{name}/params.json", 'x') as f_params:
            json.dump(params, f_params, indent=4)

        # geometrycurves and partition/maxh dictionaries

        os.makedirs(f"{name}/geometry")

        with open(f"{name}/geometry/boundary_partition_dict.json", 'x') as f_partdict:
            json.dump(self.boundary_partition_dict, f_partdict)

        with open(f"{name}/geometry/boundary_maxh_dict.json", 'x') as f_maxhdict:
            json.dump(self.boundary_maxh_dict, f_maxhdict)

        with open(f'{name}/geometry/geometrycurves.pkl', 'wb') as f_geomcurves:
            cloudpickle.dump(self.geometrycurves, f_geomcurves, protocol=4) # use protocol=4 which is compatible with cross-platform use to enable remote parallel computing

        # spatial parameters

        os.makedirs(f'{name}/spatial_parameters')

        for paramname, value in self.spatial_physical_parameters.items():
            with open(f'{name}/spatial_parameters/{paramname}.pkl', 'wb') as file:
                cloudpickle.dump(value.fh, file, protocol=4)

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
            bathy_gradnorm = ngsolve.sqrt(self.spatial_physical_parameters['H'].gradient_cf[0] * self.spatial_physical_parameters['H'].gradient_cf[0] + 
                                          self.spatial_physical_parameters['H'].gradient_cf[1] * self.spatial_physical_parameters['H'].gradient_cf[1])
        else:
            raise ValueError("Invalid value for 'based_on'. Please choose from the following options: 'bathygrad'.")
            
        for _ in range(numits):

            num_refined = mesh_functions.refine_mesh_by_elemental_integration(self.mesh, bathy_gradnorm, threshold)

           
            for name, param in self.spatial_physical_parameters.items(): # SpatialParameter-objects need to be redefined on the new mesh
                bfc = generate_bfc(self.mesh, self.order, 'diffusion')
                self.spatial_physical_parameters[name] = SpatialParameter(param.fh, bfc)

            bathy_gradnorm = ngsolve.sqrt(self.spatial_physical_parameters['H'].gradient_cf[0] * self.spatial_physical_parameters['H'].gradient_cf[0] + 
                                              self.spatial_physical_parameters['H'].gradient_cf[1] * self.spatial_physical_parameters['H'].gradient_cf[1])
                
            if num_refined == 0:
                break

        self.nfreedofs = count_free_dofs(self.femspace)



    def set_constant_physical_parameters(self, Av=None, Ah=None, sf = None, sigma=None, T=None, g=None, f=None):
        if Av is not None:
            self.constant_physical_parameters['Av'] = Av # vertical eddy viscosity or vertical eddy viscosity scale if it's assumed to scale linearly with local depth

        if Ah is not None:
            self.constant_physical_parameters['Ah'] = Ah # horizontal eddy viscosity; is unused if horizontal_diffusion = False in self.model_options

        if sf is not None:
            self.constant_physical_parameters['sf'] = sf

        if self.model_options['bed_bc'] == 'partial_slip': # Can only be set after calling this method because the basis depends on the parameters
            self.vertical_basis = truncationbasis.eigbasis_partialslip(self.M, sf, Av)

        if sigma is not None:
            self.constant_physical_parameters['sigma'] = sigma # M2-Frequency
        elif T is not None:
            self.constant_physical_parameters['sigma'] = 1/T
        
        if g is not None:
            self.constant_physical_parameters['g'] = g # Probably 9.81, but let us keep outer space estuaries into our model :)
        if f is not None:
            self.constant_physical_parameters['f'] = f


    def set_spatial_physical_parameters(self, H=None, density=None, R=None, nonlinear_ramp=None):
        if H is not None:
            self.spatial_physical_parameters['H'] = H
        if density is not None:
            self.spatial_physical_parameters['density'] = density
        if R is not None:
            self.spatial_physical_parameters['R'] = R
        if nonlinear_ramp is not None:
            self.spatial_physical_parameters['non-linear_ramp'] = nonlinear_ramp


    def set_seaward_boundary_condition(self, amplitude_list, phase_list, enhanced_winant_conforming=False, only_linear=False):
        """Sets seaward boundary condition with given amplitudes and phases. If enhanced_winant_conforming is True, then an O(B/L)-correction will be calculated for the M2-tide, to better
        fit the boundary condition with the internal dynamics. This reduces any boundary layer effects resulting from the boundary condition not matching the internal physics.
        
        
        Arguments:

        - amplitude_list:               list of amplitudes for the seaward boundary condition
        - phase_list:                   list of phases for the seaward boundary condition
        - enhanced_winant_conforming:   flag indicating whether the O(B/L) correction should be computed
        - only_linear:                  flag indicating whether only linear basis functions should be used to set the boundary condition
        
        """
        if not enhanced_winant_conforming:
            self.seaward_forcing = SeawardForcing(self, amplitude_list, phase_list, only_linear=only_linear)
        else:
            # Assumes a rectangular estuary with seaward boundary at x = 0, and a bathymetry independent of x!
            Av = self.constant_physical_parameters['Av']
            omega = 2*np.pi * self.constant_physical_parameters['sigma']
            f = self.constant_physical_parameters['f']
            g = self.constant_physical_parameters['g']

            L = self.model_options['x_scaling']
            B = self.model_options['y_scaling']
            epsilon = B / L

            H = self.spatial_physical_parameters['H'].cf
            R = self.spatial_physical_parameters["R"].cf

            P = np.sqrt(0.5) * np.array([[1, 1], [-1j, 1j]])
            Pstar = np.sqrt(0.5) * np.array([[1, 1j], [1, -1j]])

            alpha = np.array([[np.sqrt((1j/Av) * (omega + f)), 0], [0, np.sqrt((1j/Av) * (omega - f))]])

            if self.model_options['bed_bc'] == 'no_slip':
                C = np.array([[g / (Av * alpha[0,0]**2) * (ngsolve_tanh(alpha[0,0]*(H+R)) / alpha[0,0] - (H+R)), ngsolve.CF(0)], [ngsolve.CF(0), g / (Av * alpha[1,1]**2) * (ngsolve_tanh(alpha[1,1]*(H+R)) / alpha[1,1] - (H+R))]])
            elif self.model_options['bed_bc'] == 'partial_slip':
                sf = self.constant_physical_parameters['sf']
                beta = np.array([[1/(Av * alpha[0,0] * ngsolve.sinh(alpha[0,0] * (H+R)) + sf*ngsolve.cosh(alpha[0,0] * (H+R))), ngsolve.CF(0)], [ngsolve.CF(0), 1/(Av * alpha[1,1] * ngsolve.sinh(alpha[1,1] * (H+R)) + sf*ngsolve.cosh(alpha[1,1] * (H+R)))]])
                C = np.array([[g / (Av * alpha[0,0]**2) * ((sf*beta[0,0]/alpha[0,0])*ngsolve.sinh(alpha[0,0] * (H+R)) - (H+R)), ngsolve.CF(0)], [ngsolve.CF(0), g / (Av * alpha[1,1]**2) * ((sf*beta[1,1]/alpha[1,1])*ngsolve.sinh(alpha[1,1] * (H+R)) - (H+R))]])

            D = P @ C @ Pstar

            curlyD = D / (1j*omega)

            # evaluate curly_D at the seaward boundary
            y = np.linspace(-0.5, 0.5, 101, endpoint=True)
            dy = y[1] - y[0]

            eval_curlyD_11 = mesh_functions.evaluate_CF_range(curlyD[0,0], self.mesh, np.zeros_like(y), y)
            eval_curlyD_12 = mesh_functions.evaluate_CF_range(curlyD[0,1], self.mesh, np.zeros_like(y), y)
            eval_curlyD_21 = mesh_functions.evaluate_CF_range(curlyD[1,0], self.mesh, np.zeros_like(y), y)
            eval_curlyD_22 = mesh_functions.evaluate_CF_range(curlyD[1,1], self.mesh, np.zeros_like(y), y)

            Deff_integrand = eval_curlyD_11 - (eval_curlyD_12 * eval_curlyD_21) / eval_curlyD_22
            Deff = dy * Deff_integrand.sum() # integrate based on leftpoint rule

            inner_integral = dy * np.cumsum(eval_curlyD_21 / eval_curlyD_22)

            A = amplitude_list[1] * np.exp(1j*phase_list[1])
            Z = A + epsilon * A / np.sqrt(Deff) * np.tan(L / np.sqrt(Deff)) * (B * dy * inner_integral.sum() - B * inner_integral)

            amplitude = np.sqrt(Z.real**2 + Z.imag**2)
            amplitude_spline = ngsolve.BSpline(2, [y[0]] + list(y) + [0.51], list(amplitude)) # add a point to the right as well to prevent the CF from evaluating to zero at the end
            amplitude_list[1] = ngsolve.CF(amplitude_spline(ngsolve.y))
            phase = np.arctan2(Z.imag, Z.real)
            phase_spline = ngsolve.BSpline(2, [y[0]] + list(y) + [0.51], list(phase))
            phase_list[1] = ngsolve.CF(phase_spline(ngsolve.y))

            self.seaward_forcing = SeawardForcing(self, amplitude_list, phase_list, only_linear=only_linear)







    def set_riverine_boundary_condition(self, discharge, **kwargs):
        """Sets riverine boundary condition assuming that the depth-averaged along-channel velocity scales linearly with local depth. Only total river discharge (dimensional)
        needs to be provided. If in **kwargs, manual is set to True, a user-provided lateral distribution of the discharge is used instead.
        
        Arguments:
        
        - discharge (float):            total amount of river discharge [m^3 / s] 
        
        """
        self.riverine_forcing = RiverineForcing(self, discharge, **kwargs)


    # Classification in terms of elliptic, hyperbolic, or parabolic (or neither)

    def construct_classification_matrices(self, x, y):
        """Rewrites the model equations as a first order system of PDEs and constructs the matrices in front of the x-derivative and in front of the y-derivative at a specific point.
        The ordering of the unknowns is as follows:

        gamma_0, gamma_(-1), gamma(1), gamma(-2), gamma(2), ...,
        alpha_(0,0), alpha_(1,0), ..., alpha(0,-1), alpha(1,-1), ..., alpha(0,1), alpha(1,1), ..., alpha(0,-2), alpha(1,-2), ..., alpha(0,2), alpha(1,2), ...
        beta_(0,0), beta_(1,0), ..., beta(0,-1), beta(1,-1), ..., beta(0,1), beta(1,1), ..., beta(0,-2), beta(1,-2), ..., beta(0,2), beta(1,2), ...
        d/dx (alpha_(0,0), alpha_(1,0), ..., alpha(0,-1), alpha(1,-1), ..., alpha(0,1), alpha(1,1), ..., alpha(0,-2), alpha(1,-2), ..., alpha(0,2), alpha(1,2), ...)
        d/dx (beta_(0,0), beta_(1,0), ..., beta(0,-1), beta(1,-1), ..., beta(0,1), beta(1,1), ..., beta(0,-2), beta(1,-2), ..., beta(0,2), beta(1,2), ...)
        d/dy (alpha_(0,0), alpha_(1,0), ..., alpha(0,-1), alpha(1,-1), ..., alpha(0,1), alpha(1,1), ..., alpha(0,-2), alpha(1,-2), ..., alpha(0,2), alpha(1,2), ...)
        d/dy (beta_(0,0), beta_(1,0), ..., beta(0,-1), beta(1,-1), ..., beta(0,1), beta(1,1), ..., beta(0,-2), beta(1,-2), ..., beta(0,2), beta(1,2), ...).

        Arguments:

        - x (float):            x-value at which matrices should be computed
        - y (float):            y-value at which matrices should be computed
        """

        num_unknowns = 2*self.imax+1 + 6*self.M*(2*self.imax+1)
        Ax = np.zeros((num_unknowns, num_unknowns)) # x-derivative system matrix
        Ay = np.zeros((num_unknowns, num_unknowns)) # y-derivative system matrix
        
        L = self.model_options['x_scaling']
        B = self.model_options['y_scaling']

        H = mesh_functions.evaluate_CF_point(self.spatial_physical_parameters['H'].cf, self.mesh, x, y)
        R =  mesh_functions.evaluate_CF_point(self.spatial_physical_parameters['R'].cf, self.mesh, x, y)
        g = self.constant_physical_parameters['g']

        G4 = self.vertical_basis.tensor_dict['G4']

        # The index of gamma_0 is 0, the index of gamma_(-|i|) is 2*abs(i)-1, the index of gamma_(|i|) is 2*i
        # The index of alpha_(m,0) is 2*imax+1 + m, the index of alpha_(m,-|i|) is 2*imax+1 + M*(2i-1) + m, the index of alpha_(m,|i|) is 2*imax+1 + M*(2i) + m
        # The index of beta_(m,0) is 2*imax+1 + M*(2imax+1) + m, the index of beta_(m,-|i|) is 2*imax+1 + M*(2imax+1) + M*(2i-1) + m, the index of beta_(m,|i|) is 2*imax+1 + M*(2imax+1) + M*(2i) + m
        # The index of d/dx alpha_(m,0) is 2*imax+1 + 2*M*(2imax+1) + m, the index of d/dx alpha_(m,-|i|) is 2*imax+1 + 2*M*(2imax+1) + M*(2i-1) + m, the index of d/dx alpha_(m,|i|) is 2*imax+1 + 2*M*(2imax+1) + M*(2i) + m
        # The index of d/dx beta_(m,0) is 2*imax+1 + 3*M*(2imax+1) + m, the index of d/dx beta_(m,-|i|) is 2*imax+1 + 3*M*(2imax+1) + M*(2i-1) + m, the index of d/dx beta_(m,|i|) is 2*imax+1 + 3*M*(2imax+1) + M*(2i) + m
        # The index of d/dy alpha_(m,0) is 2*imax+1 + 4*M*(2imax+1) + m, the index of d/dy alpha_(m,-|i|) is 2*imax+1 + 4*M*(2imax+1) + M*(2i-1) + m, the index of d/dy alpha_(m,|i|) is 2*imax+1 + 4*M*(2imax+1) + M*(2i) + m
        # The index of d/dy beta_(m,0) is 2*imax+1 + 5*M*(2imax+1) + m, the index of d/dy beta_(m,-|i|) is 2*imax+1 + 5*M*(2imax+1) + M*(2i-1) + m, the index of d/dy beta_(m,|i|) is 2*imax+1 + 5*M*(2imax+1) + M*(2i) + m

        # depth-integrated continuity equation (derivatives w.r.t. x and y are considered to be derivatives here; otherwise the matrices are always singular)
        for m in range(self.M):
            Ax[0, 2*self.imax + 1 + m] = 0.5 * G4(m) / L * (H+R)
            Ay[0, 2*self.imax + 1 + self.M * (2*self.imax + 1) + m] = 0.5 * G4(m) / B * (H+R)
            for i in range(1, self.imax+1):
                Ax[2*i-1, 2*self.imax + 1 + self.M*(2*i-1) + m] = 0.5 * G4(m) / L * (H+R)
                Ay[2*i-1, 2*self.imax + 1 + self.M * (2*self.imax + 1) + self.M*(2*i-1) + m] = 0.5 * G4(m) / B * (H+R)

                Ax[2*i, 2*self.imax + 1 + self.M*(2*i) + m] = 0.5 * G4(m) / L * (H+R)
                Ay[2*i, 2*self.imax + 1 + self.M * (2*self.imax + 1) + self.M*(2*i) + m] = 0.5 * G4(m) / B * (H+R)


        # u-momentum equation

        for m in range(self.M):
            # residual component
            Ax[2*self.imax + 1 + m, 2*self.imax+1 + 2*self.M*(2*self.imax+1) + m] = -0.25 / (L**2) * (H+R)
            Ay[2*self.imax + 1 + m, 2*self.imax+1 + 4*self.M*(2*self.imax+1) + m] = -0.25 / (B**2) * (H+R)
            Ax[2*self.imax + 1 + m, 0] = 0.5 * g * G4(m) / L
            for i in range(1, self.imax+1):
                # -i-component
                Ax[2*self.imax + 1 + self.M*(2*i-1) + m, 2*self.imax+1 + 2*self.M*(2*self.imax+1) + self.M*(2*i-1) + m] = -0.25 / (L**2) * (H+R)
                Ay[2*self.imax + 1 + self.M*(2*i-1) + m, 2*self.imax+1 + 4*self.M*(2*self.imax+1) + self.M*(2*i-1) + m] = -0.25 / (B**2) * (H+R)
                Ax[2*self.imax + 1 + self.M*(2*i-1) + m, 2*i - 1] = 0.5 * g * G4(m) / L
                # +i-component
                Ax[2*self.imax + 1 + self.M*(2*i) + m, 2*self.imax+1 + 2*self.M*(2*self.imax+1) + self.M*(2*i) + m] = -0.25 / (L**2) * (H+R)
                Ay[2*self.imax + 1 + self.M*(2*i) + m, 2*self.imax+1 + 4*self.M*(2*self.imax+1) + self.M*(2*i) + m] = -0.25 / (B**2) * (H+R)
                Ax[2*self.imax + 1 + self.M*(2*i) + m, 2*i] = 0.5 * g * G4(m) / L

        # v-momentum equation

        for m in range(self.M):
            # residual component
            Ax[2*self.imax + 1 + self.M*(2*self.imax+1) + m, 2*self.imax+1 + 3*self.M*(2*self.imax+1) + m] = -0.25 / (L**2) * (H+R)
            Ay[2*self.imax + 1 + self.M*(2*self.imax+1) + m, 2*self.imax+1 + 5*self.M*(2*self.imax+1) + m] = -0.25 / (B**2) * (H+R)
            Ay[2*self.imax + 1 + self.M*(2*self.imax+1) + m, 0] = 0.5 * g * G4(m) / B
            for i in range(1, self.imax+1):
                # -i-component
                Ax[2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i-1) + m, 2*self.imax+1 + 3*self.M*(2*self.imax+1) + self.M*(2*i-1) + m] = -0.25 / (L**2) * (H+R)
                Ay[2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i-1) + m, 2*self.imax+1 + 5*self.M*(2*self.imax+1) + self.M*(2*i-1) + m] = -0.25 / (B**2) * (H+R)
                Ay[2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i-1) + m, 2*i - 1] = 0.5 * g * G4(m) / B
                # +i-component
                Ax[2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i) + m, 2*self.imax+1 + 3*self.M*(2*self.imax+1) + self.M*(2*i) + m] = -0.25 / (L**2) * (H+R)
                Ay[2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i) + m, 2*self.imax+1 + 5*self.M*(2*self.imax+1) + self.M*(2*i) + m] = -0.25 / (B**2) * (H+R)
                Ay[2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i) + m, 2*i] = 0.5 * g * G4(m) / B

        # compatibility of alpha and d/dx alpha

        for m in range(self.M):
            # residual component
            Ax[2*self.imax + 1 + 2*self.M*(2*self.imax+1) + m, 2*self.imax + 1 + m] = 1
            for i in range(1, self.imax+1):
                # -i-component
                Ax[2*self.imax + 1 + 2*self.M*(2*self.imax+1) + self.M*(2*i-1) + m, 2*self.imax + 1 + self.M*(2*i-1) + m] = 1
                # +i-component
                Ax[2*self.imax + 1 + 2*self.M*(2*self.imax+1) + self.M*(2*i) + m, 2*self.imax + 1 + self.M*(2*i) + m] = 1

        # compatibility of beta and d/dx beta

        for m in range(self.M):
            # residual component
            Ax[2*self.imax + 1 + 3*self.M*(2*self.imax+1) + m, 2*self.imax + 1 + self.M*(2*self.imax+1) + m] = 1
            for i in range(1, self.imax+1):
                # -i-component
                Ax[2*self.imax + 1 + 3*self.M*(2*self.imax+1) + self.M*(2*i-1) + m, 2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i-1) + m] = 1
                # +i-component
                Ax[2*self.imax + 1 + 3*self.M*(2*self.imax+1) + self.M*(2*i) + m, 2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i) + m] = 1

        # compatibility of alpha and d/dy alpha

        for m in range(self.M):
            # residual component
            Ay[2*self.imax + 1 + 4*self.M*(2*self.imax+1) + m, 2*self.imax + 1 + m] = 1
            for i in range(1, self.imax+1):
                # -i-component
                Ay[2*self.imax + 1 + 4*self.M*(2*self.imax+1) + self.M*(2*i-1) + m, 2*self.imax + 1 + self.M*(2*i-1) + m] = 1
                # +i-component
                Ay[2*self.imax + 1 + 4*self.M*(2*self.imax+1) + self.M*(2*i) + m, 2*self.imax + 1 + self.M*(2*i) + m] = 1

        # compatibility of beta and d/dy beta

        for m in range(self.M):
            # residual component
            Ay[2*self.imax + 1 + 5*self.M*(2*self.imax+1) + m, 2*self.imax + 1 + self.M*(2*self.imax+1) + m] = 1
            for i in range(1, self.imax+1):
                # -i-component
                Ay[2*self.imax + 1 + 5*self.M*(2*self.imax+1) + self.M*(2*i-1) + m, 2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i-1) + m] = 1
                # +i-component
                Ay[2*self.imax + 1 + 5*self.M*(2*self.imax+1) + self.M*(2*i) + m, 2*self.imax + 1 + self.M*(2*self.imax+1) + self.M*(2*i) + m] = 1

        return Ax, Ay
    

    def classify(self, x, y, zero_tolerance=1e-8):
        """Returns 'hyperbolic', 'parabolic', 'elliptic', 'untraditional' depending on the eigenvalues and eigenvectors of Ay^(-1)Ax
        
        Arguments:

        - x (float):                    x-value,
        - y (float):                    y-value,
        - zero_tolerance (float):       tolerance where a number is considered to be equal to zero
        
        """

        Ax, Ay = self.construct_classification_matrices(x, y)
        try:
            Ay_inv = np.linalg.inv(Ay)
            eigvals, eigvecs = np.linalg.eig(Ay_inv @ Ax)
        except np.linalg.LinAlgError:
            Ax_inv = np.linalg.inv(Ax)
            eigvals, eigvecs = np.linalg.eig(Ax_inv @ Ay)

        if eigvals.dtype == complex:
            if np.all(np.imag(eigvals) > zero_tolerance * np.real(eigvals)): # if the imaginary part is at least zero_tolerance times the real part, it is considered non-zero
                return 'elliptic'
            else: # if for at least one eigenvalue, the imaginary part is zero (considered to be zero_tolerance times the real part to incorporate rounding errors)
                return 'untraditional'
        elif eigvals.dtype == float:
            sorted_eigvals = np.sort(eigvals)
            differences = sorted_eigvals[1:] - sorted_eigvals[:-1]
            if np.all(differences > zero_tolerance * sorted_eigvals[1:]): # if the differences are at least zero_tolerance times the eigenvalues themselves, they are considered to be all distinct and the system is hyperbolic
                return 'hyperbolic'
            else:
                if np.linalg.matrix_rank(eigvecs) == eigvecs.shape[0]: # if the geometric multiplicity of the matrix of eigenvectors matches the algebraic multiplicities, the system is still hyperbolic
                    return 'hyperbolic'
                else:
                    return 'parabolic'
        else:
            raise ValueError(f"Unrecognised eigenvalue datatype {eigvals.dtype}.")
        
    


        





        

def load_hydrodynamics(name, **kwargs):
    """Creates a Hydrodynamics object from a folder generated by the save-method of the Hydrodynamics object. This object can *only* be used for postprocessing.
    
    Arguments:
        - name:       name of the folder the data may be found in
        
    """
    # options

    f_options = open(f'{name}/options.json', 'r')
    model_options: dict = json.load(f_options)
    
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
    if model_options['mesh_generation_method'] == 'unstructured':
        maxh = params.pop('maxh')
    else:
        num_cells = params.pop('num_cells')
    f_params.close()


    with open(f'{name}/geometry/geometrycurves.pkl', 'rb') as f_geomcurves:
        geometrycurves = cloudpickle.load(f_geomcurves)

    with open(f'{name}/geometry/boundary_partition_dict.json', 'rb') as f_partdict:
        boundary_partition_dict = json.load(f_partdict)

    with open(f'{name}/geometry/boundary_maxh_dict.json', 'rb') as f_maxhdict:
        boundary_maxhdict = json.load(f_maxhdict)

    geometry = parametric_geometry(geometrycurves, boundary_partition_dict, boundary_maxhdict)
    # mesh = ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh))
    if model_options['mesh_generation_method'] == 'unstructured':
        mesh = ngsolve.Mesh(generate_mesh(geometry, model_options['mesh_generation_method'], maxh_unstructured=maxh))
        hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, maxh_global=maxh, geometrycurves=geometrycurves)
    else:
        mesh = ngsolve.Mesh(generate_mesh(geometry, model_options['mesh_generation_method'], num_cells=num_cells))
        hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, num_cells=num_cells, geometrycurves=geometrycurves)

    # Make Hydrodynamics object
    # hydro = Hydrodynamics(mesh, model_options, imax, M, sem_order, time_basis)
    # hydro.scaling = scaling


    bfc = generate_bfc(mesh, order=sem_order, method='diffusion', alpha=1)

    
    # add spatial parameters
    hydro.spatial_physical_parameters = dict()

    for param in os.listdir(f'{name}/spatial_parameters'):
        filename = os.fsdecode(param)
        param_name = filename[:-4] # ignore file extension
        with open(f'{name}/spatial_parameters/{param_name}.pkl', 'rb') as file:
            param_fh = cloudpickle.load(file)

        hydro.spatial_physical_parameters[param_name] = SpatialParameter(param_fh, bfc)

        # oneDfemspace = ngsolve.H1(mesh, order = sem_order)
        # gf = ngsolve.GridFunction(oneDfemspace)
        # mesh_functions.load_basevector(gf.vec.data, f'{name}/spatial_parameters/{filename}')

        # hydro.spatial_physical_parameters[param_name] = gf

    # add constant parameters

    hydro.constant_physical_parameters = params

    # add solution

    # with open(f'{name}/solution.pkl', 'rb') as file:
    #     hydro.solution_gf = dill.load(file)

    hydro.solution_gf = ngsolve.GridFunction(hydro.femspace)
    mesh_functions.load_basevector(hydro.solution_gf.vec.data, f'{name}/solution.npy', **kwargs)

    hydro.restructure_solution()

    # set partial-slip basis in case necessary
    if model_options['bed_bc'] == 'partial_slip':
        hydro.vertical_basis = truncationbasis.eigbasis_partialslip(hydro.M, hydro.constant_physical_parameters['sf'], hydro.constant_physical_parameters['Av'])

    return hydro



class RiverineForcing(object):

    
    def __init__(self, hydro: Hydrodynamics, discharge, manual=False):

        # initialise parameters
        
        self.hydro = hydro

        Av = self.hydro.constant_physical_parameters['Av']

        if self.hydro.model_options['bed_bc'] == 'partial_slip':
            sf = self.hydro.constant_physical_parameters['sf']

        H = self.hydro.spatial_physical_parameters['H'].cf
        R = self.hydro.spatial_physical_parameters['R'].cf

        if manual:
            self.discharge_cf = discharge
        else:
            self.discharge = discharge

            # integrate (H+R)^2 over width
            y = np.linspace(-0.5, 0.5, 10001)
            dy = y[1] - y[0]

            eval_integrand = mesh_functions.evaluate_CF_range((H+R)*(H+R), hydro.mesh, np.ones_like(y), y) # currently only works for unit square domains!
            integral = dy * eval_integrand.sum() # numerical integration with leftpoint rule

            self.discharge_cf = self.discharge * (H+R) * (H+R) / (hydro.model_options['y_scaling'] * integral)



        # project vertical structure onto vertical basis

        if self.hydro.model_options['bed_bc'] == 'no_slip':

            def vertical_structure(z):
                return 1 - z**2
        
        elif self.hydro.model_options['bed_bc'] == 'partial_slip':

            def vertical_structure(z):
                return 2*Av/sf + 1 - z**2
            
        projection = truncationbasis.Projection(vertical_structure, hydro.vertical_basis, hydro.M)
        projection.construct_analytical_massmatrix()
        projection.project_galerkin(10, 30, sparse=False) # matrix is so small that a sparse structure is completely unnecessary

        self.normal_alpha = []

        if self.hydro.model_options['bed_bc'] == 'no_slip':
            for m in range(hydro.M):
                self.normal_alpha.append(-projection.coefficients[m] * 1.5 * self.discharge_cf / (H + R))

        elif self.hydro.model_options['bed_bc'] == 'partial_slip':
            for m in range(hydro.M):
                self.normal_alpha.append(-projection.coefficients[m] * (3 * sf * self.discharge_cf) / ((H + R) * (2*sf + 6*Av)))







    # def __init__(self, hydro: Hydrodynamics, discharge_amplitude_list, discharge_phase_list, is_constant=True): # Currently, only constant river discharge works
        
    #     self.discharge_amplitudes = discharge_amplitude_list
    #     self.discharge_phases = discharge_phase_list
    #     self.hydro = hydro
    #     self.is_constant = is_constant

    #     self.discharge_dict = dict() # Use a dictionary to enable negative indices
    #     self.Q_vec = dict() # vector (\int_0^T Q h_p dt), p = -imax, ..., imax

    #     G3 = hydro.vertical_basis.tensor_dict['G3']
    #     G4 = hydro.vertical_basis.tensor_dict['G4']


    #     # fill amplitude and phase lists with zeros for unfilled elements unless is_constant == True and create the vector Q_vec

    #     if not is_constant:
    #         for _ in range(hydro.imax + 1 - len(discharge_amplitude_list)):
    #             self.discharge_amplitudes.append(0)
    #             self.discharge_phases.append(0)

    #         self.discharge_dict[0] = self.discharge_amplitudes[0]
    #         self.Q_vec[0] = hydro.time_basis.inner_product(0, 0) * self.discharge_dict[0]
    #         for i in range(1, hydro.imax + 1):
    #             self.discharge_dict[i] = self.discharge_amplitudes[i] * ngsolve.cos(self.discharge_phases[i])
    #             self.discharge_dict[-i] = self.discharge_amplitudes[i] * ngsolve.sin(self.discharge_phases[i])

    #             self.Q_vec[i] = self.discharge_dict[i] * hydro.time_basis.inner_product(i, i)
    #             self.Q_vec[-i] = self.discharge_dict[-i] * hydro.time_basis.inner_product(-i, -i)

    #     else:
    #         self.discharge_dict[0] = self.discharge_amplitudes[0]
    #         self.Q_vec[0] = (0.5 / hydro.constant_physical_parameters['sigma']) * self.discharge_dict[0]
        
    #     # Computation of normal components

    #     if is_constant and hydro.model_options['density'] == 'depth-independent':
    #         d1 = [0.5*(1/hydro.constant_physical_parameters['sigma']) * hydro.constant_physical_parameters['g'] * G4(k) for k in range(hydro.M)]
    #         d2 = [hydro.spatial_physical_parameters['H'].cf / (2 * hydro.constant_physical_parameters['sigma']) * G4(k) for k in range(hydro.M)]

    #         C = [-0.25 * (1/hydro.constant_physical_parameters['sigma']) * G3(k,k) *  \
    #              (hydro.constant_physical_parameters['Av'] / (hydro.spatial_physical_parameters['H'].cf*hydro.spatial_physical_parameters['H'].cf)) \
    #                 for k in range(hydro.M)]
            
    #         # sum_d1d2 = sum([d1[k]*d2[k] for k in range(hydro.M)])
    #         sum_d1d2 = sum([d1[k]*d2[k]/C[k] for k in range(hydro.M)])

    #         # self.normal_alpha = [{0: (-d1[m]/(sum_d1d2*C[m])) * self.Q_vec[0]} for m in range(hydro.M)]
    #         self.normal_alpha = [{0: (-d1[m]/(sum_d1d2*C[m])) * self.Q_vec[0]} for m in range(hydro.M)]
    #         self.normal_alpha_boundaryCF = [{0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][0]}, default=0)} for m in range(hydro.M)]
    #         for m in range(hydro.M):
    #             for q in range(1, hydro.imax + 1):
    #                 self.normal_alpha[m][q] = 0
    #                 self.normal_alpha[m][-q] = 0

    #                 self.normal_alpha_boundaryCF[m][q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: 0}, default=0)
    #                 self.normal_alpha_boundaryCF[m][-q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: 0}, default=0)

    #     elif (not is_constant) and hydro.model_options['density'] == 'depth-independent':

    #         C = [0.25 * (1/hydro.constant_physical_parameters['sigma']) * (k+0.5)*(k+0.5) * np.pi*np.pi * \
    #              (hydro.constant_physical_parameters['Av'] / (hydro.spatial_physical_parameters['H'].cf*hydro.spatial_physical_parameters['H'].cf)) \
    #                 for k in range(hydro.M)]
            
    #         d1 = [0.5*(1/hydro.constant_physical_parameters['sigma']) * hydro.constant_physical_parameters['g'] * \
    #               (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M)]
    #         d2 = [hydro.spatial_physical_parameters['H'].cf / (2 * hydro.constant_physical_parameters['sigma']) * \
    #               (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M)]
            
    #         c1 = [[1 + 0.25*np.pi*np.pi*q*q*(4/(4*C[k]*C[k]-np.pi**2 * q**2)) for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]
    #         c2 = [[-0.5*np.pi*q*(4*C[k])/(4*C[k]*C[k] - np.pi**2 * q**2) for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]
    #         c3 = [[-0.5*np.pi*q/C[k] for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]
    #         c4 = [[4*C[k] / (4*C[k]*C[k]-np.pi*np.pi*q*q) for q in range(1, hydro.imax + 1)] for k in range(hydro.M)]

    #         e1 = [-sum([d1[k]*d2[k]*c1[k][q-1] / C[k] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]
    #         e2 = [-sum([d1[k]*d2[k]*c2[k][q-1] / C[k] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]
    #         e3 = [-sum([d1[k]*d2[k]*c3[k][q-1] / c4[k][q-1] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]
    #         e4 = [-sum([d1[k]*d2[k] / c4[k][q-1] for k in range(hydro.M)]) for q in range(1, hydro.imax + 1)]

    #         gamma = dict()
    #         sum_d1d2 = sum([d1[k]*d2[k]/C[k] for k in range(hydro.M)])
    #         gamma[0] = -self.Q_vec[0] / sum_d1d2

    #         for q in range(1, hydro.imax + 1):
    #             gamma[q] = e1[q-1] / (e4[q-1]*e1[q-1] - e3[q-1]) * (self.Q_vec[q] - (e3[q-1]/e1[q-1])*self.Q_vec[-q])
    #             gamma[-q] = (self.Q_vec[-q] - e2[q-1]*gamma[q]) / e1[q-1]

    #         self.normal_alpha = [{0: (-d1[m]/(sum_d1d2*C[m])) * self.Q_vec[0]} for m in range(hydro.M)]
    #         self.normal_alpha_boundaryCF = [{0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][0]}, default=0)} for m in range(hydro.M)]
    #         for m in range(hydro.M):
    #             for q in range(1, hydro.imax + 1):
    #                 self.normal_alpha[m][q] = d1[m]*c3[m][q-1]*gamma[-q] / c4[m][q-1] + d2[m]*gamma[q] / c4[m][q-1]
    #                 self.normal_alpha[m][-q] = d1[m]*c1[m][q-1]*gamma[-q] / C[m] + d1[m]*c2[m][q-1]*gamma[q] / C[m]

    #                 self.normal_alpha_boundaryCF[m][q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][q]}, default=0)
    #                 self.normal_alpha_boundaryCF[m][-q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][-q]}, default=0)
        


class SeawardForcing(object):

    def __init__(self, hydro: Hydrodynamics, amplitude_list, phase_list, only_linear=False):
        """List of amplitudes and phases starting at the subtidal component, moving to M2 frequency and moving to M4, M6, ...
        The phase list also starts at the subtidal component, but the first component is never used."""
        self.hydro = hydro
        self.only_linear = only_linear

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


# Old solve code; now moved to solve.py
    # def SolveNewton(self, advection_weighting_parameter, tol, maxitns, printing=True, print_cond=False, autodiff=False, method='pardiso', return_vals=False):
    #     u_n = copy.copy(self.solution_gf)
        
    #     # mesh_functions.plot_gridfunction_colormap(u_n, u_n.space.mesh, refinement_level=3)
    #     for i in range(maxitns):
    #         if printing:
    #             print(f"Newton iteration {i}:")

    #         if not autodiff:
    #             if return_vals:
    #                 invtime = self.NewtonIteration(advection_weighting_parameter, print_cond, method=method, return_invtime=True)
    #             else:
    #                 self.NewtonIteration(advection_weighting_parameter, print_cond, method=method)
    #         else:
    #             self.NewtonIteration_autodiff(u_n.vec, method=method)
            
    #         residual = u_n.vec.CreateVector()
    #         apply_start = timeit.default_timer()
    #         self.total_bilinearform.Apply(self.solution_gf.vec, residual)
    #         apply_time = timeit.default_timer() - apply_start
    #         homogenise_essential_Dofs(residual, self.femspace.FreeDofs())
            

    #         stop_criterion_value = abs(ngsolve.InnerProduct(self.solution_gf.vec - u_n.vec, residual))
    #         residual_norm_sq = ngsolve.InnerProduct(residual, residual)

    #         # residual_array = residual.FV().NumPy()

    #         # stop_criterion_value = np.sqrt(stop_criterion_value) / np.sqrt(self.num_equations)
    #         print(f"   Evaluating weak forms took {apply_time} seconds")
    #         print(f"   Stopping criterion value is equal to {stop_criterion_value}")
    #         print(f"   Residual norm (except Dirichlet boundary) is equal to {np.sqrt(residual_norm_sq)}")
    #         print(f"   Scaled residual norm (free DOFs) is {np.sqrt(residual_norm_sq) / np.sqrt(self.nfreedofs)}\n")

    #         residual_gf = ngsolve.GridFunction(self.femspace)
    #         residual_gf.vec.data = residual

    #         # for k in range(5):
    #         #     mesh_functions.plot_gridfunction_colormap(residual_gf.components[k], self.mesh, refinement_level=3, title=f'Iteration {i}, component {k}')

    #         u_n = copy.copy(self.solution_gf)

    #         # mesh_functions.plot_gridfunction_colormap(u_n, u_n.space.mesh, refinement_level=3)

    #         if stop_criterion_value < tol:
    #             break
        
    #     if return_vals:
    #         return np.sqrt(residual_norm_sq), invtime


    # def NewtonIteration(self, advection_weighting_parameter, print_cond=False, method='pardiso', return_invtime=False):
    #     self.restructure_solution()
    #     forms_start = timeit.default_timer()
    #     a = ngsolve.BilinearForm(self.femspace)
    #     if self.scaling:
    #         weakforms.add_weak_form(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
    #                                 self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
    #                                 self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis,
    #                                 self.riverine_forcing.normal_alpha, only_linear=True)
    #         if advection_weighting_parameter != 0:
    #             weakforms.add_linearised_nonlinear_terms(a, self.model_options, self.alpha_trialfunctions, self.alpha_solution, self.beta_trialfunctions, self.beta_solution,
    #                                                     self.gamma_trialfunctions, self.gamma_solution, 
    #                                                     self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
    #                                                     self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis,
    #                                                     self.riverine_forcing.normal_alpha)
    #     else:
    #         weakforms.add_bilinear_part(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
    #                                 self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.M, self.imax,
    #                                 self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis,
    #                                 self.riverine_forcing.normal_alpha, forcing=True)
    #         if advection_weighting_parameter != 0:
    #             weakforms.add_linearised_nonlinear_part(a, self.model_options, self.alpha_trialfunctions, self.beta_trialfunctions, self.gamma_trialfunctions,
    #                                                     self.umom_testfunctions, self.vmom_testfunctions, self.DIC_testfunctions, self.alpha_solution, self.beta_solution, self.gamma_solution,
    #                                                     self.M, self.imax, self.constant_physical_parameters, self.spatial_physical_parameters, self.vertical_basis, self.time_basis, self.riverine_forcing.normal_alpha,
    #                                                     advection_weighting_parameter, self.n)
                
    #         # add_linearised_nonlinear_part_to_bilinearform(self, a, self.alpha_solution, self.beta_solution, self.gamma_solution, advection_weighting_parameter)
    #     forms_time = timeit.default_timer() - forms_start
    #     if method == 'gmres':
    #         prec = ngsolve.Preconditioner(a, 'direct')
    #     assembly_start = timeit.default_timer()
    #     a.Assemble()
    #     assembly_time = timeit.default_timer() - assembly_start
    #     if method == 'gmres':
    #         prec.Update()

    #     # Preconditioner

    #     # Jacobi_pre_mat = a.mat.CreateSmoother(self.femspace.FreeDofs())

    #     rhs = self.solution_gf.vec.CreateVector()
    #     self.total_bilinearform.Apply(self.solution_gf.vec, rhs)

    #     du = ngsolve.GridFunction(self.femspace)
    #     for i in range(self.femspace.dim):
    #         du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions
    #     if print_cond and method != 'gmres': # CANT USE THIS OPTION RIGHT NOW; IT IS PROBABLY INACCURATE ANYWAY
    #         Idmat = ngsolve.Projector(mask=self.femspace.FreeDofs(), range=True)

    #         # abs_eigs = np.absolute(np.array(EigenValues_Preconditioner(a.mat, Idmat)))

    #         # print(f'   Estimated condition number (2-norm) is {np.amax(abs_eigs)/np.amin(abs_eigs)}')

    #     # Direct
    #     if method == 'umfpack' or method == 'pardiso':
    #         inversion_start = timeit.default_timer()
    #         du.vec.data = a.mat.Inverse(freedofs=self.femspace.FreeDofs(), inverse=method) * rhs
    #         # du.vec.data = a.mat.Inverse(inverse=method) * rhs
    #         inversion_time = timeit.default_timer() - inversion_start
    #     # elif method == 'ext_pardiso':
    #     #     inversion_start = timeit.default_timer()
    #     #     A = mesh_functions.get_csr_matrix(a.mat)
    #     #     f = rhs.FV().NumPy()
    #     #     sol_arr = pypardiso.spsolve(A, f)
    #     #     du.vec.FV().NumPy()[:] = sol_arr

    #         inversion_time = timeit.default_timer() - inversion_start

    #     # GMRes(a.mat, freedofs=self.femspace.FreeDofs(), x=du.vec, b=rhs)
    #     # Iterative
    #     if method == 'gmres':
    #         # if print_cond: AGAIN, CANT USE THIS OPTION RIGHT NOW; IT IS PROBABLY INACCURATE ANYWAY
    #             # abs_eigs = np.absolute(np.array(EigenValues_Preconditioner(a.mat, prec)))
    #             # print(f'   Estimated preconditioned condition number (2-norm) is {np.amax(abs_eigs)/np.amin(abs_eigs)}')
    #         inversion_start = timeit.default_timer()
    #         GMRes(a.mat, pre=prec, x=du.vec, b=rhs)
    #         inversion_time = timeit.default_timer() - inversion_start

    #     # To print norm of rhs for non-Dirichlet DOFs
    #     homogenise_essential_Dofs(rhs, self.femspace.FreeDofs())
    #     print(f'   Norm of right-hand side is equal to {ngsolve.Norm(rhs)}')
    #     print(f'   Norm of Newton step is {ngsolve.Norm(du.vec)}')
    #     print(f'   Setting up weak forms for linearisation took {forms_time} seconds')
    #     print(f'   Inversion took {inversion_time} seconds')
    #     print(f'   Assembly took {assembly_time} seconds')

    #     self.solution_gf.vec.data = self.solution_gf.vec.data - du.vec.data
    #     if return_invtime:
    #         return inversion_time


    
    # def NewtonIteration_autodiff(self, u_n, method='pardiso'):
    #     self.restructure_solution()

    #     rhs = self.solution_gf.vec.CreateVector()
    #     self.total_bilinearform.Apply(self.solution_gf.vec, rhs)

    #     autodiff_start = timeit.default_timer()
    #     self.total_bilinearform.AssembleLinearization(u_n)
    #     autodiff_time = timeit.default_timer() - autodiff_start

    #     du = ngsolve.GridFunction(self.femspace)
    #     for i in range(self.femspace.dim):
    #         du.components[i].Set(0, ngsolve.BND) # homogeneous boundary conditions

    #     inversion_start = timeit.default_timer()
    #     du.vec.data = self.total_bilinearform.mat.Inverse(freedofs=self.femspace.FreeDofs(), inverse=method) * rhs
    #     inversion_time = timeit.default_timer() - inversion_start

    #     # To print norm of rhs for non-Dirichlet DOFs
    #     homogenise_essential_Dofs(rhs, self.femspace.FreeDofs())
    #     print(f'   Norm of right-hand side is equal to {ngsolve.Norm(rhs)},')
    #     print(f'   Norm of Newton step is {ngsolve.Norm(du.vec)},')
    #     print(f'   Automatic differentiation took {autodiff_time} seconds,')
    #     print(f'   Inversion took {inversion_time} seconds.')

    #     self.solution_gf.vec.data = self.solution_gf.vec.data - du.vec.data

        


        

    # def solve(self, advection_weighting_parameter_list, skip_nonlinear=False, print_condition_number=False, autodiff=False, maxits=10, tol=1e-5, method='pardiso', return_testvalues=False):      

    #     if self.loaded_from_files:
    #         print("Unable to solve: this Hydrodynamics object was loaded from files and can only be used for postprocessing")
    #         return
        
    #     # Set up FEM space
    #     print(f"\nSetting up Finite Element Space for {'linear' if skip_nonlinear else f'{advection_weighting_parameter_list[0]}-non-linear'} simulation with {self.M} vertical modes and {self.imax+1} harmonic\n"
    #           +f"components (including subtidal). In total, there are {(2*self.M + 1)*(2*self.imax+1)} equations.\n"
    #           +f"\nAssumptions used:\n\n- Bed boundary condition: no slip\n- Rigid lid assumption\n- Eddy viscosity: constant\n- Density: depth-independent.\n\n")
        
        
    #     print(f"Total number of free degrees of freedom: {self.nfreedofs}, so ~{self.nfreedofs / self.num_equations} free DOFs per equation.")

    #     # Set initial guess
    #     print(f"Setting initial guess\n")

    #     sol = ngsolve.GridFunction(self.femspace)
    #     sol.components[2*(self.M)*(2*self.imax+1)].Set(self.seaward_forcing.boundaryCFdict[0], ngsolve.BND)
    #     for q in range(1, self.imax + 1):
    #         sol.components[2*(self.M)*(2*self.imax+1) + q].Set(self.seaward_forcing.boundaryCFdict[-q], ngsolve.BND)
    #         sol.components[2*(self.M)*(2*self.imax+1) + self.imax + q].Set(self.seaward_forcing.boundaryCFdict[q], ngsolve.BND)

    #     self.solution_gf = sol

    #     num_continuation_steps = len(advection_weighting_parameter_list)
    #     # self.epsilon = advection_weighting_parameter_list[-1] # save the value of epsilon for later use

    #     for i in range(num_continuation_steps):
    #         print(f"Epsilon = {advection_weighting_parameter_list[i]}\n")
    #         print(f"Generating weak forms of the PDE system\n")

    #         self.model_options['advection_epsilon'] = advection_weighting_parameter_list[i]
    #         self.setup_forms(advection_weighting_parameter_list[i] == 0)

    #         # Combine bilinear and linear forms because Newton solver only works with a complete bilinear form
    #         print(f"Solving using Newton-Raphson method with {maxits} iterations max. and error at most {tol}, using the {method.upper()} solver.\n")

    #         if skip_nonlinear:
    #             advection_weighting_parameter = 0
    #         else:
    #             advection_weighting_parameter = advection_weighting_parameter_list[i]
            
    #         if return_testvalues:
    #             resnorm, invtime = self.SolveNewton(advection_weighting_parameter, tol=tol, maxitns=maxits, printing=True, print_cond = print_condition_number, autodiff=autodiff, method=method, return_vals=True)
    #         else:
    #             self.SolveNewton(advection_weighting_parameter, tol=tol, maxitns=maxits, printing=True, print_cond = print_condition_number, autodiff=autodiff, method=method, return_vals=False)

    #         if skip_nonlinear:
    #             break

    #     # reorder components in the gridfunction so that they can be worked with more easily
    #     print(f"Solution process complete.")
    #     self.restructure_solution()

    #     if return_testvalues:
    #         return resnorm, invtime