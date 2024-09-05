import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ngsolve
import scipy.sparse as sp
import copy

import boundary_fitted_coordinates
from spatial_parameter import SpatialParameter


def mesh_to_coordinate_array(mesh):
    """Generates a coordinate array from a netgen mesh.
    
    Args:

    - mesh:      netgen mesh object
    """

    coords = [[]]
    for p in mesh.Points():
        x, y, z = p.p
        coords[-1] += [x, y, z]
        coords.append([])

    coords = coords[:-1] # Delete last empty list        
    return np.array(coords)


def mesh2d_to_triangles(mesh):
    """Gives an array containing the indices of the triangle vertices in mesh.
    
    Args:
    
    - mesh:     netgen mesh object.
    """

    triangles = [[]]
    for el in mesh.Elements2D():
        # Netgen does not store integers in el.vertices, but netgen.libngpy._meshing.PointId objects; first we convert
        vertices = [v.nr - 1 for v in el.vertices] # PointId objects start counting at 1
        triangles[-1] += vertices
        triangles.append([])
    
    triangles = triangles[:-1] # Delete last empty list
    return np.array(triangles)


def refine_mesh_by_elemental_integration(mesh: ngsolve.Mesh, cf: ngsolve.CoefficientFunction, K: float, p=1):

    if K <= 1:
        print("Please enter K>1")
        return

    integralvals = ngsolve.Integrate(cf, mesh, ngsolve.VOL, element_wise=True)

    counter = 0

    # compute average integral val
    avg = (1/mesh.ne) * sum([integralvals[el.nr]**p for el in mesh.Elements()])


    # print(avg)

    for el in mesh.Elements():
        if integralvals[el.nr]**(p) > K * avg:
            counter += 1
        # print(integralvals[el.nr]**(1/p) / avg)
        mesh.SetRefinementFlag(el, integralvals[el.nr]**(p)  > K * avg)
    
    mesh.Refine()
    return counter


def refine_mesh_bathygrad(mesh: ngsolve.Mesh, H: SpatialParameter, rho: SpatialParameter, threshold, SEM_order, p=1, numits=1):
    bathy_gradnorm = ngsolve.sqrt(H.gradient_cf[0]*H.gradient_cf[0] + H.gradient_cf[1]*H.gradient_cf[1])
    for _ in range(numits):
        num_refined = refine_mesh_by_elemental_integration(mesh, bathy_gradnorm, threshold)

        bfc = boundary_fitted_coordinates.generate_bfc(mesh, SEM_order, "diffusion", alpha=1)
        H = SpatialParameter(H.fh, bfc)
        rho = SpatialParameter(rho.fh, bfc)

        bathy_gradnorm = ngsolve.sqrt(H.gradient_cf[0]*H.gradient_cf[0] + H.gradient_cf[1]*H.gradient_cf[1])

        if num_refined == 0:
            break


def get_triangulation(mesh):
    coords = mesh_to_coordinate_array(mesh)
    triangles = mesh2d_to_triangles(mesh)
    triangulation = tri.Triangulation(coords[:,0], coords[:, 1], triangles)
    return triangulation


def evaluate_gridfunction_point(gfu, mesh, x, y):
    return gfu(mesh(x, y))

def evaluate_gridfunction_range(gfu, mesh, x, y):
    return gfu(mesh(x, y)).flatten()

def plot_gridfunction_colormap(gfu, mesh, refinement_level=1, show_mesh=False, title='Gridfunction', **kwargs):
    """"""
    triangulation = get_triangulation(mesh.ngmesh)
    refiner = tri.UniformTriRefiner(triangulation)
    refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)
    
    eval_gfu = evaluate_gridfunction_range(gfu, mesh, refined_triangulation.x, refined_triangulation.y)
    fig, ax = plt.subplots()
    if show_mesh:
        ax.triplot(triangulation, linewidth=0.5, color='k', zorder=2)
    colormesh = ax.tripcolor(refined_triangulation, eval_gfu, **kwargs)

    ax.set_title(title)
    cbar = fig.colorbar(colormesh)
    # plt.show()


def plot_mesh2d(mesh, title=None, color='k', linewidth=0.5):
    coords = mesh_to_coordinate_array(mesh.ngmesh)
    triangles = mesh2d_to_triangles(mesh.ngmesh)
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    _, ax = plt.subplots()
    ax.triplot(triangulation, color=color, linewidth=linewidth)
    if title:
        ax.set_title(title)
    plt.show()


def evaluate_coefficient_function_point(func, mesh, x, y):
    return func(mesh(x, y))


def evaluate_coefficient_function_range(func, mesh, xrange, yrange):
    eval_func = np.zeros((xrange.shape[0], yrange.shape[0]))
    for i in range(yrange.shape[0]):
        eval_func[i, :] = func(mesh(xrange, np.ones_like(xrange)*yrange[i]))[:, 0]
    return eval_func


def get_boundaryelement_vertices(mesh: ngsolve.Mesh, bnd):
    """Returns a list of (x,y)-coordinates of all the gridpoints in a certain part of the mesh boundary"""

    IDlist = []
    for el in mesh.Boundaries(bnd).Elements():
        IDlist.append(el.vertices[0])
        IDlist.append(el.vertices[1])
    IDlist = list(set(IDlist)) # Remove duplicates

    plist = []
    for p in mesh.vertices:
        if p in IDlist:
            plist.append(p.point)

    return plist


def save_gridfunction(gfu: ngsolve.GridFunction, filename, format='npy'):
    gfu_vec_array = gfu.vec.FV().NumPy()
    if format == 'npy':
        np.save(filename, gfu_vec_array)
    elif format == 'txt':
        np.savetxt(filename, gfu_vec_array)
    else:
        raise ValueError(f"Invalid format {format}: please choose npy or txt")


# def save_gridfunction_to_txt(gfu: ngsolve.GridFunction, filename, **kwargs):
#     gfu_vec_array = gfu.vec.FV().NumPy()
#     np.savetxt(filename, gfu_vec_array, **kwargs)


# def save_gridfunction_to_npy(gfu: ngsolve.GridFunction, filename):
#     gfu_vec_array = gfu.vec.FV().NumPy()
#     np.save(filename, gfu_vec_array)


def load_basevector(vec: ngsolve.BaseVector, filename, format = 'npy'):
    if format == 'npy':
        array = np.load(filename)
    elif format == 'txt':
        array = np.genfromtxt(filename)
    else:
        raise ValueError(f"Invalid format {format}: please choose npy or txt")
    
    vec.FV().NumPy()[:] = array

# def set_basevector_from_txt(vec: ngsolve.BaseVector, filename, **kwargs):
#     array = np.genfromtxt(filename, **kwargs)
#     vec.FV().NumPy()[:] = array


# def set_basevector_from_npy(vec: ngsolve.BaseVector, filename):
#     array = np.load(filename)
#     vec.FV().NumPy()[:] = array

# def set_basevector_from_NumPyArr(vec: ngsolve.BaseVector, arr):
#     vec.FV().NumPy()[:] = arr


def get_dirichletdof_indices(freedofs: ngsolve.BitArray):
    """Returns a list of indices corresponding to constrained (Dirichlet) degrees of freedom, based on a 
    bitarray where 0 denotes a constrained DOF."""
    
    indices = []
    counter = 0
    for isFree in freedofs:
        if not isFree:
            indices.append(counter)
        counter += 1
    return indices
    

def get_csr_matrix(mat: ngsolve.BaseMatrix):
    rows, cols, vals = mat.COO()
    return sp.csr_matrix((vals, (rows, cols)))


def set_complex_gridfunction(mesh, order, real, imag, dirichlet=None):
    V = ngsolve.H1(mesh, order, dirichlet=dirichlet, complex=True)
    gf = ngsolve.GridFunction(V)

    gf.Set(real + 1j*imag)



















