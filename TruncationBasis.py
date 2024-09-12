###############################################################
## THIS FILE CONTAINS CLASSES TruncationBasis AND Projection ##
###############################################################


import numpy as np
import matplotlib.pyplot as plt
import IntegrationTools
import ngsolve
import scipy.special
import scipy.sparse
from scipy.sparse.linalg import spsolve



class TruncationBasis(object):
    """TruncationBasis object with evaluation_function attribute. This is an ngsolve-CoefficientFunction that takes as arguments 
    a coordinate z, and a non-negative integer n to denote the index of the basis function. Assumed to have domain [-1,0]. By means
    of the weight function, also has information about the underlying version of L^2([-1,0]).

    Attributes:

    - evaluation_function:              function of z and n that evaluates basis function n at z;
    - derivative_evaluation_function:   derivative w.r.t. z;
    - evaluation_cf:                    ngsolve CoefficientFunction version of evaluation_function with n as parameter;
    - inner_product:                    function of i and j that returns the value of the (weighted) inner product of basis functions i and j;
    - weight_function:                  function of z that specifies the weight function of this inner product and the underlying version of L^2. 
    
    Methods:
    
    - plot:                     plots a number of basis functions.
    - add_analytical_integrals  adds a number of functions that return

    # DO NOT YET WORK

    - get_normalising_factors:  normalises the basis functions w.r.t. the weighted L^2-norm up to index n;
    - normalise:                updates the basis functions using the normalising factors from [get_normalising_factors];
    
    """

    def __init__(self, evaluation_function, coefficient_function, inner_product=None, weight_function=None, derivative_evaluation_function=None, second_derivative_evaluation_function=None):
        """Arguments:
        
        - evaluation_function:      function with input z and (integer) n;
        - coefficient_function:     (Python) function with input (integer) n that outputs an ngsolve Coefficient function-version of the
                                    nth basis function;
        - inner_product:            function that takes input i and j and outputs the L^2([-1,0])-inner product of 
                                    basis functions i and j; could be a weighted inner product
        - weight_function:          weight function of the inner product above; if no weight is provided, a constant 1 one is used.
        
        """
        self.evaluation_function = evaluation_function
        self.derivative_evaluation_function = derivative_evaluation_function
        if inner_product:
            self.inner_product = inner_product

        self.coefficient_function = coefficient_function

        # You can provide a weight function without providing an inner product
        if weight_function:
            self.weight_function = weight_function
        else:
        # Use identity weight function if no weight function is provided
            def wf(x):
                return 1
            
            self.weight_function = wf

        if derivative_evaluation_function is not None:
            self.derivative_evaluation_function = derivative_evaluation_function
        
        if second_derivative_evaluation_function is not None:
            self.second_derivative_evaluation_function = second_derivative_evaluation_function


    def plot(self, num_functions: int, num_points: int = 1000):
        """Plots a number of the basis functions.
        
        Arguments:
        
        - num_functions:        number of basis functions (starting at 0) that should be displayed;
        - num_points:           number of points the basis functions should be evaluated at;
        """
        _, ax = plt.subplots()
        z = np.linspace(-1, 0, num_points)
        for n in range(num_functions):
            ax.plot(z, self.evaluation_function(z, n))
        plt.show()


    def add_analytical_tensors(self, tensor_dict):
        self.tensor_dict = tensor_dict


    # def get_normalising_factors(self, index: int, num_quadrature_points: int, num_mesh_elements: int):
    #     """Normalises the basis functions w.r.t. the weighted L^2-norm up to index [index] (starting at 0). Uses 
    #     Gauss-Lobatto-Legendre quadrature to calculate the norm.
        
    #     Arguments:

    #     - index:                    index up to and including which the basis functions are normalised;
    #     - num_quadrature_points:    number of quadrature points per element;
    #     - num_mesh_elements:        number of elements used in the mesh for the numerical integration;
    #     """

    #     self.normalising_factors = np.zeros(index + 1)
    #     GLL_points = IntegrationTools.get_GLL_points(num_quadrature_points)
    #     mesh_GLL_points = IntegrationTools.get_mesh_GLL_points(num_mesh_elements, GLL_points, -1, 0)

    #     for i in range(index + 1):
    #         values = self.evaluation_function(mesh_GLL_points, i)
    #         norm_squared = IntegrationTools.GLL_integrate_mesh(values*values*self.weight_function(mesh_GLL_points), GLL_points, -1, 0)
    #         self.normalising_factors[i] = 1 / np.sqrt(norm_squared)


    # def normalise(self, index):
    #     """Updates the evaluation function of the basis using the normalising factors up to index [index];
        
    #     Arguments:
        
    #     - index:            index up to and including which the basis functions are normalised;
    #     """

    #     def new_evaluation_function(z, n):
    #         if n <= index:
    #             return self.normalising_factors[n] * self.evaluation_function(z, n)
    #         else:
    #             return self.evaluation_function(z, n)
        
    #     self.evaluation_function = new_evaluation_function


##################################
## PREDEFINED EXAMPLES OF BASES ##
##################################
        
    # 1: vertical eigenfunction basis corresponding to constant eddy viscosity and no-slip bed boundary condition

def f(z, n):
    return  np.cos((n+0.5)*np.pi*z)


def fprime(z, n):
    return -(n+0.5)*np.pi*np.sin((n+0.5)*np.pi*z)


def fdoubleprime(z, n):
    return -(n+0.5)**2 * np.pi**2 * f(z,n)


def cf(n):
    return ngsolve.cos((n+0.5)*ngsolve.pi*ngsolve.z)


def inner_prod(m, k):
    if m == k:
        return 0.5
    else:
        return 0
    
eigbasis_constantAv = TruncationBasis(f, cf, inner_prod, derivative_evaluation_function=fprime, second_derivative_evaluation_function=fdoubleprime)

def minusonepower(num):
    if num % 2 == 0:
        return 1
    else:
        return -1

def analytical_G1(m, n, k):
    return 0.25*(minusonepower(m+n+k+1)/((m+n+k+1.5)*np.pi) + minusonepower(m+n-k)/((m+n-k+0.5)*np.pi)
                    + minusonepower(m-n+k)/((m-n+k+0.5)*np.pi) + minusonepower(m-n-k+1)/((m-n-k-0.5)*np.pi))

def G1_iszero(m, n, k):
    return False

def analytical_G2(m, n, k):
    if m != k:
        term1 = minusonepower(m-n+k+1)/((m-n+k+0.5)*np.pi) + minusonepower(m+n+k+1)/((m+n+k+1.5)*np.pi) + \
                minusonepower(m-n-k)/((m-n-k-0.5)*np.pi) + minusonepower(m+n-k)/((m+n-k+0.5)*np.pi)
        term2 = (minusonepower(m+n+k)-1)/((m+k+1)*np.pi) + (minusonepower(m+n-k+1)-1)/((m-k)*np.pi)
    else:
        term1 = minusonepower(m-n+k+1)/((m-n+k+0.5)*np.pi) + minusonepower(m+n+k+1)/((m+n+k+1.5)*np.pi) + \
                minusonepower(m-n-k)/((m-n-k-0.5)*np.pi) + minusonepower(m+n-k)/((m+n-k+0.5)*np.pi)
        term2 = (minusonepower(m+n+k)-1)/((m+k+1)*np.pi)
    
    return ((2*m+1)/(8*n+4))*term1 + ((2*m+1)/(4*n+2))*term2

def G2_iszero(m, n, k):
    return False

def analytical_G3(m, k):
    if m == k:
        return -0.5*(m+0.5)*(m+0.5)*np.pi*np.pi
    else:
        return 0
    
def G3_iszero(m, k):
    return m != k
    

def analytical_G4(k):
    return minusonepower(k)/((k+0.5)*np.pi)


def G4_iszero(k):
    return False


def analytical_G5(k):
    return 1/((k+0.5)*(k+0.5)*np.pi*np.pi) - minusonepower(k)/((k+0.5)*np.pi) 

def G5_iszero(k):
    return False

eigbasis_constantAv.add_analytical_tensors({'G1': analytical_G1,
                                            'G1_iszero': G1_iszero,
                                            'G2': analytical_G2,
                                            'G2_iszero': G2_iszero,
                                            'G3': analytical_G3,
                                            'G3_iszero': G3_iszero,
                                            'G4': analytical_G4,
                                            'G4_iszero': G4_iszero,
                                            'G5': analytical_G5,
                                            'G5_iszero': G5_iszero})  

# 2: temporal basis based on sines (negative index) and cosines (positive index)

def harmonic_time_basis(sigma):

    def h(t, q):
        if q < 0:
            return np.sin(2*np.pi*sigma*q*t)
        elif q == 0:
            return 0.5 * np.sqrt(2) * np.ones_like(t)
        elif q > 0:
            return np.cos(2*np.pi*sigma*q*t)
        
    def h_cf(q):
        if q < 0:
            return ngsolve.sin(2*ngsolve.pi*sigma*q*ngsolve.x)
        elif q == 0:
            return 0.5 * np.sqrt(2)
        elif q > 0:
            return ngsolve.cos(2*ngsolve.pi*sigma*q*ngsolve.x)

    def inner_prod_h(i, j):
            if i == j:
                return 0.5 / sigma
            else:
                return 0
            
    time_basis = TruncationBasis(h, h_cf, inner_prod_h)

    def analytical_H2(p, q):
        if p == -q:
            return np.pi*abs(p)
        else:
            return 0
        
    def H2_iszero(p,q):
        return p != -q
        
    def analytical_H3(p, q, r):
        if p>=0 and q>=0 and r>=0:
            if p == q+r or q == p+r or r == p+q:
                return 0.25 / sigma
            else:
                return 0
        elif (p>=0 and q<0 and r<0) or (q>=0 and p<0 and r<0) or (r>=0 and p<0 and q<0):
            sorted_list = sorted([p, q, r])
            if sorted_list[2] == abs(sorted_list[0]) + abs(sorted_list[1]):
                return -0.25 / sigma
            elif (sorted_list[0] == -sorted_list[2]+sorted_list[1]) or (sorted_list[1] == -sorted_list[2]+sorted_list[0]):
                return 0.25 / sigma
            else:
                return 0
        else:
            return 0
        
    def H3_iszero(p, q, r):
        if p>=0 and q>=0 and r>=0:
            return not ((p == q+r) or (q == p+r) or (r == p+q))
        elif (p>= 0 and q<0 and r<0) or (q>=0 and p<0 and r<0) or (r>=0 and p<0 and q<0):
            sorted_list = sorted([p, q, r])
            return not ((sorted_list[2] == abs(sorted_list[0]) + abs(sorted_list[1])) or (sorted_list[0] == -sorted_list[2]+sorted_list[1]) or (sorted_list[1] == -sorted_list[2]+sorted_list[0]))
        
        
    time_basis.add_analytical_tensors({'H2': analytical_H2,
                                       'H2_iszero': H2_iszero,
                                       'H3': analytical_H3,
                                       'H3_iszero': H3_iszero})
    
    return time_basis

unit_harmonic_time_basis = harmonic_time_basis(1)


class Projection(object):
    """Object that contains information about a function that was projected onto a finite-dimensional subspace of L^2([-1,0]).

    Attributes:

    - original_function:                function of z that returns value of the original function at z;
    - basis:                            TruncationBasis object; original function is projected on these basis functions;
    - dimension:                        number of basis functions used for the projection;
    - massmatrix:                       mass matrix associated to this projection; two-dimensional numpy array; generated by
                                        methods 'construct_numerical_massmatrix' or 'construct_analytical_massmatrix';
    - sparse_massmatrix:                matrix in CSC format for sparse inversion; generated by 'project_galerkin' in 'sparse'=True;
    - coefficients:                     array of coefficients for each basis function; generated by 'project_galerkin';
    - projected_function:               function of z that returns value of the projection at z; generated by 'project_galerkin';
    
    Methods:
    
    - project_galerkin:                 computes Galerkin projection w.r.t. the basis' weighted inner product;
    - construct_numerical_massmatrix:   constructs mass matrix for this projection using Gauss-Lobatto-Legendre quadrature;
    - construct_analytical_massmatrix:  constructs mass matrix for this projection using the basis' inner product function;
    
    """

    def __init__(self, original_function, basis: TruncationBasis, dimension: int):
        
        self.original_function = original_function
        self.basis = basis
        self.dimension = dimension


    def project_galerkin(self, num_quadrature_points: int, num_mesh_elements: int, sparse=True, minimum_coefficient = 0):
        """Performs a Galerkin projection of the original function onto [dimension] basis functions. Computation of the right-hand
        side vector is done using Gauss-Lobatto-Legendre integration with [num_quadrature_points] points.
        
        Arguments:
        
        - num_quadrature_points:            number of quadrature points for GLL-integration of right-hand side;
        - num_mesh_elements:                number of mesh elements used for the GLL-integration of right-hand side;
        - sparse:                           True or False; if True, mass matrix inversion will be done via scipy.sparse.linalg.spsolve;
        - minimum_coefficient:              minimal value of the coefficients to eliminate rounding errors;
        
        """

        # Construct right-hand side vector
        GLL_points = IntegrationTools.get_GLL_points(num_quadrature_points)
        mesh_GLL_points = IntegrationTools.get_mesh_GLL_points(num_mesh_elements, GLL_points, -1, 0)
        f = np.zeros(self.dimension)
        for j in range(self.dimension):
            f[j] = IntegrationTools.GLL_integrate_mesh(self.original_function(mesh_GLL_points)*self.basis.evaluation_function(mesh_GLL_points, j) * \
                                                  self.basis.weight_function(mesh_GLL_points), GLL_points, xl=-1, xr=0)
        if sparse:
            self.sparse_massmatrix = scipy.sparse.csc_matrix(self.massmatrix)
            self.coefficients = spsolve(self.sparse_massmatrix, f)
        else:
            self.coefficients = np.linalg.solve(self.massmatrix, f) # uses LU-decomposition

        self.coefficients = np.where(np.absolute(self.coefficients) >= minimum_coefficient, self.coefficients, 
                                     np.zeros_like(self.coefficients))

        # Define evaluation function for the Galerkin projection
        def evaluate_projection(z):
            return sum([self.coefficients[i] * self.basis.evaluation_function(z, i) for i in range(self.dimension)])
        
        self.projected_function = evaluate_projection


    def construct_numerical_massmatrix(self, num_quadrature_points: int, num_mesh_elements: int, use_weight_function: bool = False):
        """Constructs a mass matrix for the basis using numerical integration using num_quadrature_points points.
        If one desires an analytical mass matrix, one may assign it directly to the massmatrix attribute without constructing it via
        this method.
        
        Arguments:
        
        - num_quadrature_points:    number of quadrature points in case of numerical construction;
        - num_mesh_elements:        number of elements in the mesh of the numerical integration;
        - use_weight_function:      True or False; if True, then integration will be done using the weight function of the basis."""
        
        self.massmatrix = np.zeros((self.dimension, self.dimension))
        GLL_points = IntegrationTools.get_GLL_points(num_quadrature_points)
        mesh_GLL_points = IntegrationTools.get_mesh_GLL_points(num_mesh_elements, GLL_points, -1, 0)
        for i in range(self.dimension):
            for j in range(self.dimension):
                if use_weight_function:
                    values = self.basis.evaluation_function(mesh_GLL_points, i) * self.basis.evaluation_function(mesh_GLL_points, j) \
                             * self.basis.weight_function(mesh_GLL_points)
                else:
                    values = self.basis.evaluation_function(mesh_GLL_points, i) * self.basis.evaluation_function(mesh_GLL_points, j)
                self.massmatrix[i, j] = IntegrationTools.GLL_integrate_mesh(values, GLL_points, xl=-1, xr=0)
        return self.massmatrix # option to save it to another variable


    def construct_analytical_massmatrix(self):
        """Constructs a mass matrix using the weighted inner product of the associated TruncationBasis object."""
        
        self.massmatrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                self.massmatrix[i, j] = self.basis.inner_product(i, j)

        return self.massmatrix # option to save it to another variable

    

    # def project_collocation(self):
    #     """Collocation projection using GLL points"""

    #     GLL_points = IntegrationTools.get_GLL_points(self.dimension)
    #     mapped_GLL_points = IntegrationTools.map_GLL_points(GLL_points, -1, 0)

    #     mat = np.zeros((self.dimension, self.dimension))
    #     f = np.zeros(self.dimension)

    #     for i in range(self.dimension):
    #         f[i] = self.original_function(mapped_GLL_points[i])
    #         for j in range(self.dimension):
    #             mat[i, j] = self.basis.evaluation_function(mapped_GLL_points[j], i)

    #     self.coefficients = np.linalg.solve(mat, f)

    #     # Define evaluation function for the collocation projection
    #     def evaluate_projection(z):
    #         return sum([self.coefficients[i] * self.basis.evaluation_function(z, i) for i in range(self.dimension)])
        
    #     self.evaluation_function = evaluate_projection

    #     self.method = 'Collocation'
    
# Example
    
if __name__ == '__main__':

    # def orig_func(z):
    #     return -4.5 * (1 - z*z)
    
    # z_range = np.linspace(-1, 0, 1000)
    # funcvals = orig_func(z_range)

    # basis = eigbasis_constantAv
    # projections = []
    # error_norms = []

    # GLL_points = IntegrationTools.get_GLL_points(15)
    # mesh_GLL_points = IntegrationTools.get_mesh_GLL_points(30, GLL_points, -1, 0)

    # for i in range(10):
    #     projections.append(Projection(orig_func, basis, i+1))
    #     projections[-1].construct_analytical_massmatrix()
    #     projections[-1].project_galerkin(15, 30)

    #     error_vals = np.power(orig_func(mesh_GLL_points) - projections[-1].projected_function(mesh_GLL_points), 2)

    #     error_norms.append(IntegrationTools.GLL_integrate_mesh(error_vals, GLL_points, -1, 0))


    # fig, ax = plt.subplots()
    # ax.grid(':')
    # ax.plot(np.array(range(10)), np.log(error_norms))
    # plt.show()
    plt.rcParams["mathtext.fontset"] = 'dejavuserif'
    plt.rcParams["font.family"] = 'serif'

    fig, ax = plt.subplots()
    ax.grid(linestyle=':')
    depth = np.linspace(-1, 0)
    ax.plot(eigbasis_constantAv.evaluation_function(depth, 0), depth, label=r'$f_1(\varsigma)$', linewidth=1)
    ax.plot(eigbasis_constantAv.evaluation_function(depth, 1), depth, label=r'$f_2(\varsigma)$', linewidth=1)
    ax.plot(eigbasis_constantAv.evaluation_function(depth, 2), depth, label=r'$f_3(\varsigma)$', linewidth=1)
    ax.plot(eigbasis_constantAv.evaluation_function(depth, 3), depth, label=r'$f_4(\varsigma)$', linewidth=1)
    ax.set_title(r'First four vertical basis functions $f_m(\varsigma)$')
    ax.set_ylabel(r'$\varsigma$')
    ax.axvline(x=0, color='k', lw=1.5)
    ax.legend()
    plt.show()

    arr = np.array([(1, 2, 3), (4, 5, 6)])
    np.savetxt('linear_sensitivity_data/test.txt', arr)


    


    



    
        
    

    
    

    






    