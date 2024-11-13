import ngsolve
from netgen.meshing import Element0D, Element1D, Element2D, MeshPoint, FaceDescriptor, Mesh
from netgen.csg import Pnt
from create_geometry import BOUNDARY_DICT, SEA, WALLUP, WALLDOWN, RIVER


def generate_mesh(geometry, method='unstructured', maxh_unstructured=None, num_els_x=None, num_els_y=None):
    """Generates an ngsolve Mesh from a geometry. Structured mesh generation currently only works for unit square geometries.
    
    Arguments:
    
    - geometry:             geometry to create mesh for;
    - method:               method to generate mesh; 'unstructured' (default) makes a Delaunay triangulation, 'structured_quads' makes a structured grid from rectangles,
                            'structured_tri' makes a structured grid from triangles;
    - maxh_unstructured:    mesh size; only used if method == 'unstructured';
    - num_els_x:            number of elements in x-direction; only used if method != 'unstructured';
    - num_els_y:            number of elements in y-direction; only used if method != 'unstructured';    
    """

    if method == 'unstructured':
        if isinstance(maxh_unstructured, float) or isinstance(maxh_unstructured, int):
            return ngsolve.Mesh(geometry.GenerateMesh(maxh=maxh_unstructured))
        else:
            raise ValueError("Could not generate unstructured mesh: mesh size not provided or invalid (maxh_unstructured)")
    elif method == 'structured_quads':
        # Meshing process based on https://docu.ngsolve.org/latest/netgen_tutorials/manual_mesh_generation.html

        if num_els_x is None or num_els_y is None:
            raise ValueError("Could not generate structured mesh: number of elements in at least one direction not provided.")
        
        # initialisation
        mesh = ngsolve.Mesh()
        mesh.SetGeometry(geometry)
        mesh.dim = 2

        # add mesh points
        meshpoints = []
        for i in range(num_els_x + 1):
            for j in range(num_els_y + 1):
                meshpoints.append(mesh.Add(MeshPoint(Pnt(i / num_els_x, j / num_els_y - 0.5, 0)))) # the unit square geometry is centered at y = 0, x = 0.5

        # add interior elements
        mesh.SetMaterial(1, 'all')
        for j in range(num_els_y):
            for i in range(num_els_x):
                mesh.Add(Element2D(1, [meshpoints[i + j * (num_els_y+1)], meshpoints[i + (j+1)*(num_els_y+1)], meshpoints[i+1 + (j+1)*(num_els_y+1)], meshpoints[i+1 + j*(num_els_y+1)]]))

        # add boundary elements
        mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=BOUNDARY_DICT[SEA]))
        for j in range(num_els_y):
            mesh.Add(Element1D([meshpoints[j], meshpoints[j+1]], index=BOUNDARY_DICT[SEA]))

        mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=BOUNDARY_DICT[WALLUP]))
        for i in range(num_els_x):
            mesh.Add(Element1D([meshpoints[num_els_y + (num_els_y + 1) * i], meshpoints[num_els_y + (num_els_y + 1) * (i+1)]], index=BOUNDARY_DICT[WALLUP]))

        mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=BOUNDARY_DICT[RIVER]))
        for j in range(num_els_y):
            mesh.Add(Element1D([meshpoints[num_els_x * (num_els_y + 1) + j], meshpoints[num_els_x * (num_els_y + 1) + j + 1]], index=BOUNDARY_DICT[RIVER]))

        mesh.Add(FaceDescriptor(surfnr=1, domin=1, bc=BOUNDARY_DICT[WALLDOWN]))
        for i in range(num_els_x):
            mesh.Add(Element1D([meshpoints[(num_els_y + 1) * i], meshpoints[((num_els_y + 1) * (i+1))]], index=BOUNDARY_DICT[WALLDOWN]))

        return mesh
    elif method == 'structured_tri':
        raise ValueError("Sorry, triangular structured grid not implemented yet")
    else:
        raise ValueError(f"{method} is an invalid mesh generation method; please choose from 'unstructured', 'structured_quads', or 'structured_tri'.")

