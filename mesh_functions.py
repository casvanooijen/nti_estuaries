import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import ngsolve
import scipy.sparse as sp
import copy



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


def get_triangulation(mesh):
    """Converts a netgen mesh to a matplotlib.tri-Triangularion object.
    
    Arguments:
    
        - mesh:     netgen mesh object.
    """
    coords = mesh_to_coordinate_array(mesh)
    triangles = mesh2d_to_triangles(mesh)
    triangulation = tri.Triangulation(coords[:,0], coords[:, 1], triangles)
    return triangulation


def refine_mesh_by_elemental_integration(mesh: ngsolve.Mesh, cf: ngsolve.CoefficientFunction, K: float, p=1):
    """
    Refines an ngsolve-Mesh by integrating a user-provided ngsolve.CoefficientFunction over each element.
    If the p-norm of the coefficient function in a particular element exceeds the average p-norm among all elements by 
    a factor of K, that element is marked for refinement. Returns how many elements were refined.

    Arguments:

    - mesh:     the mesh that will be refined;
    - cf:       the coefficient function that is used for the mesh refinement rule;
    - K:        threshold by which the p-norm in a particular element must exceed the average;
    - p:        indicates which L^p-norm is used for the rule.    
    
    """
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


def evaluate_CF_point(cf, mesh, x, y):
    """
    Evaluates an ngsolve CoefficientFunction, of which the ngsolve GridFunction is a child class, at a point (x,y).
    Returns function value.

    Arguments:

        - cf:       Coefficient function to be evaluated;
        - mesh:     ngsolve mesh that contains (x,y);
        - x:        x-coordinate of evaluation point;
        - y:        y-coordinate of evaluation point;
    
    """
    return cf(mesh(x, y))

def evaluate_CF_range(cf, mesh, x, y):
    """
    Evaluates an ngsolve CoefficientFunction, of which the ngsolve GridFunction is a child class, at a range of points
    contained in two arrays x and y. Returns array of function values.

    Arguments:
        
        - cf:       Coefficient function to be evaluated;
        - mesh:     mesh that contains all points in (x,y);
        - x:        array of x-values;
        - y:        array of y-values;
    
    """
    return cf(mesh(x, y)).flatten()

def plot_CF_colormap(cf, mesh, refinement_level=1, show_mesh=False, title='Gridfunction', save=None, **kwargs):
    """
    Plots a simple colormap of a Coefficient function on a refined display mesh.

    Arguments:

        - cf:                   Coefficient function to be plotted;
        - mesh:                 computational mesh that is refined for the display mesh; 
        - refinement_level:     how many times the mesh is refined to get a display mesh on which the function is evaluated;
        - show_mesh:            if True, overlays the colormap with the mesh;
        - title:                title of the plot;
        - save:                 saves the figure to f'{save}.png',
        - **kwargs:             keyword arguments for matplotlib.pyplot.tripcolor
    
    """
    triangulation = get_triangulation(mesh.ngmesh)
    refiner = tri.UniformTriRefiner(triangulation)
    refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)
    
    eval_cf = evaluate_CF_range(cf, mesh, refined_triangulation.x, refined_triangulation.y)
    fig, ax = plt.subplots()
    if show_mesh:
        ax.triplot(triangulation, linewidth=0.5, color='k', zorder=2)
    colormesh = ax.tripcolor(refined_triangulation, eval_cf, **kwargs)

    ax.set_title(title)
    cbar = fig.colorbar(colormesh)
    if save is not None:
        fig.savefig(f'{save}.png')
    # plt.show()


def plot_mesh2d(mesh, title=None, color='k', linewidth=0.5):
    """
    Plots a wireframe of an ngsolve Mesh.

    Arguments:

        - mesh:         mesh to be plotted;
        - title:        title of the plot;
        - color:        color of the wireframe;
        - linewidth:    linewidth of the wireframe;
    
    """
    coords = mesh_to_coordinate_array(mesh.ngmesh)
    triangles = mesh2d_to_triangles(mesh.ngmesh)
    triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

    _, ax = plt.subplots()
    ax.triplot(triangulation, color=color, linewidth=linewidth)
    if title:
        ax.set_title(title)
    plt.show()


# def evaluate_coefficient_function_point(func, mesh, x, y):
#     return func(mesh(x, y))


# def evaluate_coefficient_function_range(func, mesh, xrange, yrange):
#     eval_func = np.zeros((xrange.shape[0], yrange.shape[0]))
#     for i in range(yrange.shape[0]):
#         eval_func[i, :] = func(mesh(xrange, np.ones_like(xrange)*yrange[i]))[:, 0]
#     return eval_func


def get_boundaryelement_vertices(mesh: ngsolve.Mesh, bnd):
    """Returns a list of (x,y)-coordinates of all the gridpoints in a certain part of the mesh boundary.
    
    Arguments:
    
        - mesh:     the considered mesh;
        - bnd:      the name of the considered boundary;
        
        
    """

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


def save_gridfunction(gf: ngsolve.GridFunction, filename, format='npy'):
    """
    Converts an ngsolve GridFunction to a numpy array and subsequently saves it.

    Arguments:

        - gf:           ngsolve GridFunction to be saved;
        - filename:     name of the file in which the GridFunction is saved;
        - format:       file format; choices are 'npy' for a .npy-file and 'txt' for a .txt-file. 
    
    
    """
    gf_vec_array = gf.vec.FV().NumPy()
    if format == 'npy':
        np.save(filename, gf_vec_array)
    elif format == 'txt':
        np.savetxt(filename, gf_vec_array)
    else:
        raise ValueError(f"Invalid format {format}: please choose npy or txt")


# def save_gridfunction_to_txt(gfu: ngsolve.GridFunction, filename, **kwargs):
#     gfu_vec_array = gfu.vec.FV().NumPy()
#     np.savetxt(filename, gfu_vec_array, **kwargs)


# def save_gridfunction_to_npy(gfu: ngsolve.GridFunction, filename):
#     gfu_vec_array = gfu.vec.FV().NumPy()
#     np.save(filename, gfu_vec_array)


def load_basevector(vec: ngsolve.BaseVector, filename, format = 'npy'):
    """
    Sets the value of an ngsolve BaseVector using a file.

    Arguments:
        - vec:              ngsolve BaseVector to be set;
        - filename:         name of the file;
        - format:           format of the file; choices are 'npy' for .npy-files and 'txt' for .txt-files;
    
    
    """
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
    bitarray where 0 denotes a constrained DOF.
    
    Arguments:
    
        - freedofs:     bitarray determining the free degrees of freedom, obtainable by using the ngsolve method femspace.FreeDofs()"""
    
    indices = []
    counter = 0
    for isFree in freedofs:
        if not isFree:
            indices.append(counter)
        counter += 1
    return indices
    

def get_csr_matrix(mat: ngsolve.BaseMatrix):
    """Converts an ngsolve BaseMatrix to a scipy sparse matrix in CSR-format. Returns the CSR-matrix
    
    Arguments:

        - mat:      to-be-converted BaseMatrix      
        
    """
    rows, cols, vals = mat.COO()
    return sp.csr_matrix((vals, (rows, cols)))


# def set_complex_gridfunction(mesh, order, real, imag, dirichlet=None):
#     """
#         Constructs 
#     """
#     V = ngsolve.H1(mesh, order, dirichlet=dirichlet, complex=True)
#     gf = ngsolve.GridFunction(V)

#     gf.Set(real + 1j*imag)



















