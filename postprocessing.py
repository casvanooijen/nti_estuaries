import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.animation as anim
import ngsolve

from hydrodynamics import Hydrodynamics, count_free_dofs
from copy import copy
from minusonepower import minusonepower

from mesh_functions import *


## MESH FUNCTIONS ##

def amp(gfu):
    # 29-7-2020: Old sqrt(gfu.real ** 2 + gfu.imag ** 2)
    return ngsolve.sqrt(gfu*ngsolve.Conj(gfu)).real

# def count_free_dofs(fes):
#     i = 0
#     for isFree in fes.FreeDofs():
#         i = i + isFree
#     return i


# def mesh_to_coordinate_array(mesh):
#     """Generates a coordinate array from a netgen mesh.
    
#     Args:

#     - mesh:      netgen mesh object
#     """

#     coords = [[]]
#     for p in mesh.Points():
#         x, y, z = p.p
#         coords[-1] += [x, y, z]
#         coords.append([])

#     coords = coords[:-1] # Delete last empty list        
#     return np.array(coords)


# def mesh2d_to_triangles(mesh):
#     """Gives an array containing the indices of the triangle vertices in mesh.
    
#     Args:
    
#     - mesh:     netgen mesh object.
#     """

#     triangles = [[]]
#     for el in mesh.Elements2D():
#         # Netgen does not store integers in el.vertices, but netgen.libngpy._meshing.PointId objects; first we convert
#         vertices = [v.nr - 1 for v in el.vertices] # PointId objects start counting at 1
#         triangles[-1] += vertices
#         triangles.append([])
    
#     triangles = triangles[:-1] # Delete last empty list
#     return np.array(triangles)


# def get_triangulation(mesh):
#     coords = mesh_to_coordinate_array(mesh)
#     triangles = mesh2d_to_triangles(mesh)
#     triangulation = tri.Triangulation(coords[:,0], coords[:, 1], triangles)
#     return triangulation

# ## EVALUATE GENERAL NGSOLVE COEFFICIENT FUNCTIONS ##

# def evaluate_CF_point(cf, mesh, x, y):
#     return cf(mesh(x, y))

# def evaluate_CF_range(cf, mesh, x, y):
#     return cf(mesh(x, y)).flatten()


## EVALUATE ON DIFFERENT TYPES OF DOMAINS ##

def evaluate_vertical_structure_at_point(hydro, quantity_function, p, num_vertical_points):
    sigma_range = np.linspace(-1, 0, num_vertical_points)
    Q = np.zeros_like(sigma_range)

    for i in range(num_vertical_points):
        Q[i] = evaluate_CF_point(quantity_function(sigma_range[i]), hydro.mesh, p[0], p[1])

    return Q


def evaluate_vertical_structure_at_cross_section(hydro, quantity_function, p1, p2, num_horizontal_points, num_vertical_points):
    sigma_range = np.linspace(-1, 0, num_vertical_points)
    x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
    y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

    Q = np.zeros((num_horizontal_points, num_vertical_points))
    
    for i in range(num_vertical_points):
        Q[i, :] = evaluate_CF_range(quantity_function(sigma_range[i]), hydro.mesh, x_range, y_range)

    return Q





class PostProcessing(object):

    def __init__(self, hydro: Hydrodynamics):

        self.hydro = hydro
        self.onedimfemspace = ngsolve.H1(hydro.mesh, order=hydro.order)

        self.u = lambda q, sigma : sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M)])
        self.v = lambda q, sigma : sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M)])
        self.gamma = lambda q : hydro.gamma_solution[q]
        self.gamma_abs = lambda q: ngsolve.sqrt(self.gamma(q)*self.gamma(q)) if q == 0 else ngsolve.sqrt(self.gamma(q)*self.gamma(q)+self.gamma(-q)*self.gamma(-q))

        self.u_DA = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.alpha_solution[m][q] for m in range(hydro.M)])
        self.v_DA = lambda q: sum([hydro.vertical_basis.tensor_dict['G4'](m) * hydro.beta_solution[m][q] for m in range(hydro.M)])

        self.u_timed = lambda t, sigma: sum([sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax+1)])
        self.v_timed = lambda t, sigma: sum([sum([hydro.beta_solution[m][q] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax + 1)])

        if hydro.loaded_from_files: # in this case, spatial parameters are stored as GridFunctions instead of SpatialParameter objects
            H = hydro.spatial_physical_parameters['H']
            Hx = ngsolve.grad(H)[0]
            Hy = ngsolve.grad(H)[1]
        else:
            H = hydro.spatial_physical_parameters['H'].cf
            Hx = hydro.spatial_physical_parameters['H'].gradient_cf[0]
            Hy = hydro.spatial_physical_parameters['H'].gradient_cf[1]

        F = dict()
        F[0] = [H * (ngsolve.grad(hydro.alpha_solution[m][0])[0] + ngsolve.grad(hydro.beta_solution[m][0])[1]) + \
                  hydro.alpha_solution[m][0] * Hx + hydro.beta_solution[m][0] * Hy for m in range(hydro.M)]
        for q in range(1, hydro.imax + 1):
            F[-q] = [H * (ngsolve.grad(hydro.alpha_solution[m][-q])[0] + ngsolve.grad(hydro.beta_solution[m][-q])[1]) + \
                        hydro.alpha_solution[m][-q] * Hx + hydro.beta_solution[m][-q] * Hy for m in range(hydro.M)]
            F[q] = [H * (ngsolve.grad(hydro.alpha_solution[m][q])[0] + ngsolve.grad(hydro.beta_solution[m][q])[1]) + \
                        hydro.alpha_solution[m][q] * Hx + hydro.beta_solution[m][q] * Hy for m in range(hydro.M)]
            
        self.w = lambda q, sigma : -sum([1/((m+0.5)*np.pi) * F[q][m] * (np.sin((m+0.5)*np.pi*sigma) + minusonepower(m) * np.ones_like(sigma)) for m in range(hydro.M)]) + \
                                   sigma * self.u(q, sigma) * Hx + sigma * self.v(q, sigma) * Hy
        self.w_timed = lambda t, sigma : sum([-sum([1/((m+0.5)*np.pi) * F[q][m] * (np.sin((m+0.5)*np.pi*sigma) + minusonepower(m) * np.ones_like(sigma)) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax)]) + \
                                         sigma * self.u_timed(t, sigma) * Hx + sigma * self.v_timed(t, sigma) * Hy


        self.u_abs = lambda q, sigma : ngsolve.sqrt(self.u(q,sigma)*self.u(q,sigma)) if q == 0 else ngsolve.sqrt(self.u(q,sigma)*self.u(q,sigma)+self.u(-q,sigma)*self.u(-q,sigma)) 
        self.v_abs = lambda q, sigma : ngsolve.sqrt(self.v(q,sigma)*self.v(q,sigma)) if q == 0 else ngsolve.sqrt(self.v(q,sigma)*self.v(q,sigma)+self.v(-q,sigma)*self.v(-q,sigma)) 
        self.w_abs = lambda q, sigma : ngsolve.sqrt(self.w(q,sigma)*self.w(q,sigma)) if q == 0 else ngsolve.sqrt(self.w(q,sigma)*self.w(q,sigma)+self.w(-q,sigma)*self.w(-q,sigma))
        self.gamma_abs = lambda q : ngsolve.sqrt(self.gamma(q)*self.gamma(q)) if q == 0 else ngsolve.sqrt(self.gamma(q)*self.gamma(q)+self.gamma(-q)*self.gamma(-q)) 
        
        self.u_DA_abs = lambda q: ngsolve.sqrt(self.u_DA(q)*self.u_DA(q) if q == 0 else ngsolve.sqrt(self.u_DA(q)*self.u_DA(q) + self.u_DA(-q)*self.u_DA(-q)))
        self.v_DA_abs = lambda q: ngsolve.sqrt(self.v_DA(q)*self.v_DA(q) if q == 0 else ngsolve.sqrt(self.v_DA(q)*self.v_DA(q) + self.v_DA(-q)*self.v_DA(-q)))

        # Get derivative gridfunctions

        self.ux = lambda q, sigma : sum([ngsolve.grad(hydro.alpha_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M)])
        self.vx = lambda q, sigma : sum([ngsolve.grad(hydro.beta_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M)])
        self.gammax = lambda q: ngsolve.grad(hydro.gamma_solution[q])[0]

        self.uy = lambda q, sigma : sum([ngsolve.grad(hydro.alpha_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M)])
        self.vy = lambda q, sigma : sum([ngsolve.grad(hydro.beta_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) for m in range(hydro.M)])
        self.gammay = lambda q: ngsolve.grad(hydro.gamma_solution[q])[0]

        self.usig = lambda q, sigma : sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) for m in range(hydro.M)]) #This is the derivative w.r.t. sigma, and not z. To transform this to the derivative w.r.t. z, divide by H
        self.vsig = lambda q, sigma : sum([hydro.beta_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) for m in range(hydro.M)]) #same

        self.usigsig = lambda q, sigma: sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.second_derivative_evaluation_function(sigma, m) for m in range(hydro.M)]) #This is the derivative w.r.t. sigma, and not z. To transform this to the derivative w.r.t. z, divide by H^2
        self.vsigsig = lambda q, sigma: sum([hydro.beta_solution[m][q] * hydro.vertical_basis.second_derivative_evaluation_function(sigma, m) for m in range(hydro.M)]) #This is the derivative w.r.t. sigma, and not z. To transform this to the derivative w.r.t. z, divide by H^2

        self.ux_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.alpha_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax+1)])
        self.vx_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.beta_solution[m][q])[0] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax + 1)])

        self.uy_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.alpha_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax+1)])
        self.vy_timed = lambda t, sigma: sum([sum([ngsolve.grad(hydro.beta_solution[m][q])[1] * hydro.vertical_basis.evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax + 1)])

        self.usig_timed = lambda t, sigma: sum([sum([hydro.alpha_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax+1)])
        self.vsig_timed = lambda t, sigma: sum([sum([hydro.beta_solution[m][q] * hydro.vertical_basis.derivative_evaluation_function(sigma, m) * hydro.time_basis.evaluation_function(t, q) for m in range(hydro.M)]) for q in range(-hydro.imax, hydro.imax + 1)])

    ## PLOTTING ##
        
    def plot_mesh2d(self, title=None, **kwargs):
        coords = mesh_to_coordinate_array(self.hydro.mesh.ngmesh)
        triangles = mesh2d_to_triangles(self.hydro.mesh.ngmesh)
        triangulation = tri.Triangulation(coords[:, 0], coords[:, 1], triangles)

        fig_mesh, ax_mesh = plt.subplots()
        ax_mesh.triplot(triangulation, **kwargs)
        if title:
            ax_mesh.set_title(title)
    
        
    def plot_colormap(self, quantity, refinement_level=1, show_mesh=False, title='Colormap', clabel='Color', center_range = False, contourlines=True, num_levels=10, subamplitude_lines=2, save=None, figsize=(12, 6), **kwargs):
        """"""
        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)
        
        eval_gfu = evaluate_CF_range(quantity, self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        if center_range:
            maxamp = max(np.amax(eval_gfu), -np.amin(eval_gfu))
        fig_colormap, ax_colormap = plt.subplots(figsize=figsize)
        if show_mesh:
            ax_colormap.triplot(triangulation, linewidth=0.5, color='k', zorder=2)
        if center_range:
            colormesh = ax_colormap.tripcolor(refined_triangulation, eval_gfu, vmin=-maxamp, vmax=maxamp, **kwargs)
        else:
            colormesh = ax_colormap.tripcolor(refined_triangulation, eval_gfu, **kwargs)

        if contourlines:
            levels = np.linspace(np.min(eval_gfu), np.max(eval_gfu), num_levels*(subamplitude_lines+1))
            contour = ax_colormap.tricontour(refined_triangulation, eval_gfu, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
            ax_colormap.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')

        ax_colormap.set_title(title)
        cbar = fig_colormap.colorbar(colormesh)
        cbar.ax.set_ylabel(clabel)

        ax_colormap.set_xlabel('x [m]')
        ax_colormap.set_ylabel('y [m]')

        if save is not None:
            fig_colormap.savefig(save)

        plt.tight_layout()


    def plot_contourmap(self, quantity, num_levels, refinement_level=1, subamplitude_lines=2, show_mesh=False, title='Contours', save=None, **kwargs):
        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

        eval_gfu = evaluate_CF_range(quantity, self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)

        levels = np.linspace(np.min(eval_gfu), np.max(eval_gfu), num_levels*(subamplitude_lines+1))

        fig_contour, ax_contour = plt.subplots()
        if show_mesh:
            ax_contour.triplot(triangulation, linewidth=.5, color='k', zorder=2)
        contourf = ax_contour.tricontourf(refined_triangulation, eval_gfu, levels, **kwargs)
        contour = ax_contour.tricontour(refined_triangulation, eval_gfu, levels, colors=['k'] + ["0.4"] * subamplitude_lines)

        ax_contour.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
        ax_contour.set_title(title)
        cbar = fig_contour.colorbar(contourf)

        if save is not None:
            fig_contour.savefig(save)


    def animate_colormap(self, quantity_cf, quantity_string, unit_string, file_basename, num_frames, refinement_level=1, show_mesh=False, constituent='all'):
        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

        phase = np.linspace(0, 1, num_frames, endpoint=True)

        sigma = self.hydro.constant_physical_parameters['sigma']

        if constituent == 0:
            U = np.zeros((num_frames, np.shape(refined_triangulation.x)[0]))
            for i in range(num_frames):
                U[i,:] = evaluate_CF_range(quantity_cf[0], self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        elif constituent > 0:
            U = np.zeros((num_frames, np.shape(refined_triangulation.x)[0]))
            for i in range(num_frames):
                U[i,:] = evaluate_CF_range(quantity_cf[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      quantity_cf[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent),
                                      self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                
        maxflow = max(np.amax(U), np.amax(-U))

        for i in range(num_frames):
            fig, ax = plt.subplots()

            colormesh = ax.tripcolor(refined_triangulation, U[i,:], vmin=-maxflow, vmax=maxflow, cmap='bwr')

            cbar = plt.colorbar(colormesh)
            cbar.ax.set_ylabel(quantity_string + f'[{unit_string}]')

            phasestring = str(np.round(phase[i],5))
            phasestring += '0' * (7 - len(phasestring))

            constituent_string = '' if constituent == 'all' else f'M{2*constituent} '

            ax.set_title(f'{constituent_string}{quantity_string} at t = {phasestring}' + r'$\sigma^{-1}$ s')

            fig.savefig(f'{file_basename}_phase{i}.png')


    def animate_transports(self, num_frames, num_arrows: tuple, L, B, refinement_level=3, constituent=1, basename='transport_frame', num_maxeval_points=100):
        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

        xtemp, ytemp = np.linspace(0, L, num_arrows[0]+2), np.linspace(-B/2, B/2, num_arrows[1]+2)
        x, y = xtemp[1:-1], ytemp[1:-1]
        X, Y = np.meshgrid(x,y) 

        phase = np.linspace(0, 1, num_frames, endpoint=True)

        sigma = self.hydro.constant_physical_parameters['sigma']
        H = self.hydro.spatial_physical_parameters['H'].cf

        Z = np.zeros((num_frames, np.shape(refined_triangulation.x)[0]))
        U = np.zeros((num_frames, np.shape(refined_triangulation.x)[0]))
        V = np.zeros((num_frames, np.shape(refined_triangulation.x)[0]))
        Umid = np.zeros((num_frames, num_maxeval_points))
        Vmid = np.zeros((num_frames, num_maxeval_points))

        Uquiv = np.zeros((num_frames, X.flatten().shape[0]))
        Vquiv = np.zeros((num_frames, X.flatten().shape[0]))

        if constituent == 0:
            for i in range(num_frames):
                Z[i,:] = evaluate_CF_range(self.gamma(constituent), self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                U[i,:] = evaluate_CF_range(H * self.hydro.u_DA[constituent], self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                V[i,:] = evaluate_CF_range(H * self.hydro.v_DA[constituent], self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                Umid[i,:] = evaluate_CF_range(H * self.hydro.u_DA[constituent], self.hydro.mesh, L/2 * np.ones(num_maxeval_points), np.linspace(-B/2, B/2, num_maxeval_points))
                Vmid[i,:] = evaluate_CF_range(H * self.hydro.v_DA[constituent], self.hydro.mesh, L/2 * np.ones(num_maxeval_points), np.linspace(-B/2, B/2, num_maxeval_points))
                Uquiv[i,:] = evaluate_CF_range(H * self.hydro.u_DA[constituent], self.hydro.mesh, X.flatten(), Y.flatten())
                Vquiv[i,:] = evaluate_CF_range(H * self.hydro.v_DA[constituent], self.hydro.mesh, X.flatten(), Y.flatten())
        elif constituent > 0:
            for i in range(num_frames):
                Z[i,:] = evaluate_CF_range(self.gamma(constituent) * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.gamma(-constituent) * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent),
                                      self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                U[i,:] = evaluate_CF_range(H * (self.hydro.u_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.u_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)),
                                      self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                V[i,:] = evaluate_CF_range(H * (self.hydro.v_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.v_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)),
                                      self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                Umid[i,:] = evaluate_CF_range(H * (self.hydro.u_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.u_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)),
                                      self.hydro.mesh, L/2 * np.ones(num_maxeval_points), np.linspace(-B/2,B/2,num_maxeval_points))
                Vmid[i,:] = evaluate_CF_range(H * (self.hydro.v_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.v_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)),
                                      self.hydro.mesh, L/2 * np.ones(num_maxeval_points), np.linspace(-B/2,B/2,num_maxeval_points))
                Uquiv[i,:] = evaluate_CF_range(H * (self.hydro.u_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.u_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)),
                                      self.hydro.mesh, X.flatten(), Y.flatten())
                Vquiv[i,:] = evaluate_CF_range(H * (self.hydro.v_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.v_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)),
                                      self.hydro.mesh, X.flatten(), Y.flatten())
                
        maxflowU = np.amax(np.absolute(Umid))
        maxflowV = np.amax(np.absolute(Vmid))
        maxZ = np.amax(np.absolute(Z))

        # visual_arrownorms = np.sqrt((Uquiv/L)**2 + (Vquiv/B)**2)
        visual_arrownorms = np.sqrt((Uquiv)**2 + (Vquiv)**2)

        physical_arrownorms = np.sqrt(U**2+V**2)
        maxnorms = np.amax(physical_arrownorms)

        

        for i in range(num_frames):
            fig, ax = plt.subplots(2, 2, figsize=(16, 12))

            surf = ax[0, 0].tripcolor(refined_triangulation, Z[i, :], vmin=-maxZ, vmax=maxZ, cmap='bwr')
            cbarZ = plt.colorbar(surf, ax=ax[0,0])
            cbarZ.ax.set_ylabel('Surface elevation [m]')
            ax[0, 0].set_title('Water surface')

            u = ax[0, 1].tripcolor(refined_triangulation, U[i, :], vmin=-maxflowU, vmax=maxflowU, cmap='bwr')
            cbarU = plt.colorbar(u, ax=ax[0,1])
            cbarU.ax.set_ylabel('Along-channel transport '+r'[m$^2$/s]')
            ax[0, 1].set_title('Depth-integrated along-channel velocity')

            v = ax[1, 0].tripcolor(refined_triangulation, V[i, :], vmin=-maxflowV, vmax=maxflowV, cmap='bwr')
            cbarV = plt.colorbar(v, ax=ax[1,0])
            cbarV.ax.set_ylabel('Cross-channel transport '+r'[m$^2$/s]')
            ax[1, 0].set_title('Depth-integrated cross-channel velocity')

            norms = ax[1,1].tripcolor(refined_triangulation, physical_arrownorms[i,:], vmin=0, vmax=maxnorms, cmap='viridis')
            cbarN = plt.colorbar(norms, ax=ax[1,1])
            cbarN.ax.set_ylabel('Norm of depth-integrated velocity vector '+r'[m$^2$/s]')
            ax[1, 1].set_title('Transport vectors')

            # quiv = ax[1,1].quiver(X, Y, Uquiv[i,:]/(L*visual_arrownorms[i,:]), Vquiv[i,:]/(B*visual_arrownorms[i,:]), color='white', pivot='mid')
            quiv = ax[1,1].quiver(X, Y, Uquiv[i,:]/(visual_arrownorms[i,:]), Vquiv[i,:]/(visual_arrownorms[i,:]), color='white', pivot='mid')

            phasestring = str(np.round(phase[i],5))
            phasestring += '0' * (7 - len(phasestring))
            
            plt.suptitle(r'$t=$' + phasestring + r'$\sigma^{-1}$ s')

            fig.savefig(f'{basename}_{i}.png')


    def animate_transports_midcross(self, num_frames, num_samples, L, B, basename='cross_section_transport', constituent=1):
        y = np.linspace(-B/2, B/2, num_samples)
        x = L/2 * np.ones(num_samples)

        U = np.zeros((num_frames, num_samples))
        V = np.zeros((num_frames, num_samples))

        phase = np.linspace(0, 1, num_frames, endpoint=True)

        sigma = self.hydro.constant_physical_parameters['sigma']
        H = self.hydro.spatial_physical_parameters['H'].cf

        if constituent == 1:
            for i in range(num_frames):
                U[i,:] = evaluate_CF_range(H * (self.hydro.u_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.u_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)), self.hydro.mesh, x, y)
                V[i,:] = evaluate_CF_range(H * (self.hydro.v_DA[constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, constituent) + \
                                      self.hydro.v_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase[i] / sigma, -constituent)), self.hydro.mesh, x, y)
                
        maxu = np.amax(np.absolute(U))
        maxv = np.amax(np.absolute(V))
        
        for i in range(num_frames):
            fig, ax = plt.subplots(1, 2, figsize=(12, 8))
            ax[0].plot(y, U[i,:], color='red')
            ax[0].set_ylabel('Transport '+r'm$^2$s^{-1}')
            ax[0].set_title('Along-channel transport')
            ax[0].grid(True, linestyle=':')
            ax[0].set_ylim(-maxu, maxu)

            ax[1].plot(y, V[i,:], color='blue')
            ax[1].set_ylabel('Transport '+r'm$^2$s$^{-1}$')
            ax[1].set_title('Cross-channel transport')
            ax[1].grid(True, linestyle=':')
            ax[1].set_ylim(-maxv, maxv)

            phasestring = str(np.round(phase[i],5))
            phasestring += '0' * (7 - len(phasestring))

            plt.suptitle('Depth-integrated velocities at x=L/2\n'+f't={phasestring}'+r'$\sigma^{-1}$s')

            fig.savefig(f'{basename}_{i}.png')
            

    def plot_transport_vectors(self, L, B, phase, num_arrows, title='Depth-integrated velocity vectors', refinement_level=3, constituent=1, figsize=(12,6), maxflow=None, savename=None):
        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

        xtemp, ytemp = np.linspace(0, L, num_arrows[0]+2), np.linspace(-B/2, B/2, num_arrows[1]+2)
        x, y = xtemp[1:-1], ytemp[1:-1]
        X, Y = np.meshgrid(x,y) 

        sigma = self.hydro.constant_physical_parameters['sigma']
        H = self.hydro.spatial_physical_parameters['H'].cf

        Uquiv = np.zeros(X.flatten().shape[0])
        Vquiv = np.zeros(X.flatten().shape[0])
        U = np.zeros(np.shape(refined_triangulation.x)[0])
        V = np.zeros(np.shape(refined_triangulation.x)[0])

        Uquiv[:] = evaluate_CF_range(H * (self.hydro.u_DA[constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, constituent) + \
                                self.hydro.u_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, -constituent)),
                                self.hydro.mesh, X.flatten(), Y.flatten())
        Vquiv[:] = evaluate_CF_range(H * (self.hydro.v_DA[constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, constituent) + \
                                self.hydro.v_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, -constituent)),
                                self.hydro.mesh, X.flatten(), Y.flatten())
        
        U[:] = evaluate_CF_range(H * (self.hydro.u_DA[constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, constituent) + \
                                self.hydro.u_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, -constituent)),
                                self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        V[:] = evaluate_CF_range(H * (self.hydro.v_DA[constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, constituent) + \
                                self.hydro.v_DA[-constituent] * self.hydro.time_basis.evaluation_function(phase / sigma, -constituent)),
                                self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        
        visual_arrownorms = np.sqrt((Uquiv)**2 + (Vquiv)**2)

        physical_arrownorms = np.sqrt(U**2+V**2)
        if maxflow is None:
            maxflow = np.amax(physical_arrownorms)

        fig, ax = plt.subplots(figsize=figsize)
        norms = ax.tripcolor(refined_triangulation, physical_arrownorms, vmin=0, vmax=maxflow, cmap='viridis')
        cbarN = plt.colorbar(norms, ax=ax)
        cbarN.ax.set_ylabel('Norm of depth-integrated velocity vector '+r'[m$^2$/s]')
        # ax[1, 1].set_title('Transport vectors')

        # quiv = ax[1,1].quiver(X, Y, Uquiv[i,:]/(L*visual_arrownorms[i,:]), Vquiv[i,:]/(B*visual_arrownorms[i,:]), color='white', pivot='mid')
        quiv = ax.quiver(X, Y, Uquiv/(visual_arrownorms), Vquiv/(visual_arrownorms), color='white', pivot='mid')
        ax.set_title(title)

        if savename is not None:
            fig.savefig(savename)
        



    def plot_depth_averaged_residual_circulation(self, num_arrows: tuple, L, B, refinement_level=3):
        """Only works with rectangular domains of length L and width B; temporary solution"""

        xtemp, ytemp = np.linspace(0, L, num_arrows[0]+2), np.linspace(-B/2, B/2, num_arrows[1]+2)
        x, y = xtemp[1:-1], ytemp[1:-1]
        X, Y = np.meshgrid(x,y)        

        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

        H = self.hydro.spatial_physical_parameters['H']
        # bathy_gradnorm = ngsolve.sqrt(H.gradient_cf[0]**2 + H.gradient_cf[1]**2)
        bathy = evaluate_CF_range(H.cf, self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)

        U = evaluate_CF_range(self.hydro.u_DA[0], self.hydro.mesh, X.flatten(), Y.flatten())
        V = evaluate_CF_range(self.hydro.v_DA[0], self.hydro.mesh, X.flatten(), Y.flatten())

        # U, V = np.meshgrid(u, v)

        visual_norms = np.sqrt((U/L)**2 + (V/B)**2)
        physical_norms = np.sqrt(U**2+V**2)

        fig, ax = plt.subplots(figsize=(12,6))

        bathymesh = ax.tripcolor(refined_triangulation, bathy, cmap='binary')
        cbar = plt.colorbar(bathymesh)
        cbar.ax.set_ylabel('Water depth [m]')

        # quiv = ax.quiver(X, Y, U/(L*visual_norms), V/(B*visual_norms), color='r', alpha=physical_norms/np.amax(physical_norms), pivot='mid')
        quiv = ax.quiver(X, Y, U/(L*visual_norms), V/(B*visual_norms), physical_norms, alpha=1, pivot='mid', cmap='viridis')


        cbar_quiv = plt.colorbar(quiv)
        cbar_quiv.ax.set_ylabel('Depth-averaged velocity [m/s]')

        ax.set_title('Depth-averaged residual circulation')
        
        plt.tight_layout()


    def plot_phaselag(self,L, B, refinement_level = 3, constituent=1, figsize=(10,12)):

        x = np.linspace(0, 9*L/10, 1000)
        y = np.linspace(-B/2, B/2, 1000)

        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

        surface_real = evaluate_CF_range(self.gamma(constituent), self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        surface_imag = evaluate_CF_range(-self.gamma(-constituent), self.hydro.mesh, refined_triangulation.x, refined_triangulation.y) # h_(-q)= -sin(2*pi*sigma*|q|*t)

        surface_phase = np.arctan2(surface_imag, surface_real)

        surface_real_short = evaluate_CF_range(self.gamma(constituent), self.hydro.mesh, x, y)
        surface_imag_short = evaluate_CF_range(-self.gamma(-constituent), self.hydro.mesh, x, y) # h_(-q)= -sin(2*pi*sigma*|q|*t)

        surface_phase_short = np.arctan2(surface_imag_short, surface_real_short)

        flow_real = evaluate_CF_range(self.hydro.u_DA[constituent], self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        flow_imag = evaluate_CF_range(-self.hydro.u_DA[-constituent], self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        
        flow_phase = np.arctan2(flow_imag, flow_real)

        flow_real_short = evaluate_CF_range(self.hydro.u_DA[constituent], self.hydro.mesh, x, y)
        flow_imag_short = evaluate_CF_range(-self.hydro.u_DA[-constituent], self.hydro.mesh, x, y)

        flow_phase_short = np.arctan2(flow_imag_short, flow_real_short)

        phaselag = surface_phase - flow_phase

        phaselag_short = surface_phase_short - flow_phase_short  

        max_flowphase = np.amax(flow_phase_short)
        min_flowphase = np.amin(flow_phase_short)
        max_phasediff = np.amax(phaselag_short)
        min_phasediff = np.amin(phaselag_short)


        fig, ax = plt.subplots(3,1, figsize=figsize)

        # subamplitude_lines = 2
        # num_levels = 8

        

        surface_phasemap = ax[0].tripcolor(refined_triangulation, surface_phase, cmap='plasma')
        flow_phasemap = ax[1].tripcolor(refined_triangulation, flow_phase, cmap='plasma', vmin=min_flowphase, vmax=max_flowphase)
        phaselag_map = ax[2].tripcolor(refined_triangulation, phaselag, cmap='viridis', vmin=min_phasediff, vmax=max_phasediff)
        cbar_flow = plt.colorbar(flow_phasemap, ax=ax[1])
        cbar_surf = plt.colorbar(surface_phasemap, ax=ax[0])
        cbar_lag = plt.colorbar(phaselag_map, ax=ax[2])

        # levels_surface = np.linspace(np.min(surface_phase), np.max(surface_phase), num_levels*(subamplitude_lines+1))
        # contour = ax[0].tricontour(refined_triangulation, surface_phase, levels_surface, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
        # ax[0].clabel(contour, levels_surface[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')

        # levels_flow = np.linspace(min_flowphase, max_flowphase, num_levels*(subamplitude_lines+1))
        # contour = ax[1].tricontour(refined_triangulation, surface_phase, levels_flow, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
        # ax[1].clabel(contour, levels_surface[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')

        # levels_diff = np.linspace(min_phasediff, max_phasediff, num_levels*(subamplitude_lines+1))
        # contour = ax[2].tricontour(refined_triangulation, surface_phase, levels_diff, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
        # ax[2].clabel(contour, levels_surface[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')

        ax[0].set_title('Phase of water surface')
        ax[1].set_title('Phase of along-channel velocity')
        ax[2].set_title('Phase difference')

        for i in range(3):
            ax[i].set_xlabel('x [m]')
            ax[i].set_ylabel('y [m]')

        plt.tight_layout()

    
    
    def plot_residual_strains(self, refinement_level=3):
        triangulation = get_triangulation(self.hydro.mesh.ngmesh)
        refiner = tri.UniformTriRefiner(triangulation)
        refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

        plot_fes = ngsolve.H1(self.hydro.mesh, order = self.hydro.order)

        U_DA = ngsolve.GridFunction(plot_fes)
        U_DA.Set(self.hydro.u_DA[0])

        Uy = ngsolve.grad(U_DA)[1]

        V_DA = ngsolve.GridFunction(plot_fes)
        V_DA.Set(self.hydro.v_DA[0])

        Vx = ngsolve.grad(V_DA)[0]

        Uy_eval = evaluate_CF_range(Uy, self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
        Vx_eval = evaluate_CF_range(Vx, self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)

        max_Uy = max(np.amax(Uy_eval), np.amax(-Uy_eval))
        max_Vx = max(np.amax(Vx_eval), np.amax(-Vx_eval))


        max_index = np.argmax(Uy_eval)

        fig, ax = plt.subplots(2, 1, figsize=(12,6))

        Uy_colormesh = ax[0].tripcolor(refined_triangulation, np.absolute(Uy_eval), cmap='viridis', vmin=0, vmax=max_Uy)
        cbaru = plt.colorbar(Uy_colormesh, ax=ax[0])
        ax[0].set_title(r'Absolute residual strain $\frac{\partial u}{\partial y}$')
        ax[0].scatter(refined_triangulation.x[max_index], refined_triangulation.y[max_index], color='k', marker='o')

        Vx_colormesh = ax[1].tripcolor(refined_triangulation, np.absolute(Vx_eval), cmap='viridis', vmin=0, vmax=max_Vx)
        cbarv = plt.colorbar(Vx_colormesh, ax=ax[1])
        ax[1].set_title(r'Absolute residual strain $\frac{\partial v}{\partial x}$')

        plt.tight_layout()

        return abs(Uy_eval[max_index]), abs(Vx_eval[max_index])
    

    ## COMPUTE SCALAR STATISTICS ##

    def compute_scalar_statistics(self, B, L, refinement_level=4, phaselag=True, vel_stats=True, surface_amp=True, residual_stats=True, num_vertical_samples=100):
        """
        - minimum phase lag between surface and depth-averaged longitudinal velocity at estuary middle cross section
        - maximum, median, average and minimum u and v (M2 amplitude) (not depth-averaged)
        - amplification of free surface forcing (M2)
        - residual exchange rate and location of positive maximum (M0)

        Only works for rectangular domains right now!

        """
        

        statistics = {}

        # SURFACE AMPLIFICATION/DAMPING
        if surface_amp:
            absolute_surface = evaluate_CF_range(self.gamma_abs(1), self.hydro.mesh, L * np.ones(100), np.linspace(-B/2,B/2,100))
            statistics['surface amplification'] = np.mean(absolute_surface) # amplification means either increasing amplitude of surface wave or dampening it

        # PHASE LAG
        if phaselag:
            flow_real_entrance = evaluate_CF_range(self.hydro.u_DA[1], self.hydro.mesh, L/2*np.ones(100), np.linspace(-B/2, B/2, 100))
            flow_imag_entrance = evaluate_CF_range(-self.hydro.u_DA[-1], self.hydro.mesh, L/2*np.ones(100), np.linspace(-B/2,B/2, 100))

            flow_real_channel = evaluate_CF_point(self.hydro.u_DA[1], self.hydro.mesh, L/2, 0)
            flow_imag_channel = evaluate_CF_point(-self.hydro.u_DA[-1], self.hydro.mesh, L/2, 0)

            flow_real_bank = evaluate_CF_point(self.hydro.u_DA[1], self.hydro.mesh, L/2, B/2-100) # slightly off the bank
            flow_imag_bank = evaluate_CF_point(-self.hydro.u_DA[-1], self.hydro.mesh, L/2, B/2-100)

            flow_phase_entrance = np.arctan2(flow_imag_entrance, flow_real_entrance)
            flow_phase_channel = np.arctan2(flow_imag_channel, flow_real_channel)
            flow_phase_bank = np.arctan2(flow_imag_bank, flow_real_bank)

            surface_real_entrance = evaluate_CF_range(self.gamma(1), self.hydro.mesh, L/2*np.ones(100), np.linspace(-B/2, B/2, 100))
            surface_imag_entrance = evaluate_CF_range(-self.gamma(-1), self.hydro.mesh, L/2*np.ones(100), np.linspace(-B/2, B/2, 100)) # h_(-q)= -sin(2*pi*sigma*|q|*t)

            surface_real_channel = evaluate_CF_point(self.gamma(1), self.hydro.mesh, L/2, 0)
            surface_imag_channel = evaluate_CF_point(-self.gamma(-1), self.hydro.mesh, L/2, 0)

            surface_real_bank = evaluate_CF_point(self.gamma(1), self.hydro.mesh, L/2, B/2-100) # slightly off the bank
            surface_imag_bank = evaluate_CF_point(-self.gamma(-1), self.hydro.mesh, L/2, B/2-100)

            surface_phase_entrance = np.arctan2(surface_imag_entrance, surface_real_entrance)
            surface_phase_channel = np.arctan2(surface_imag_channel, surface_real_channel)
            surface_phase_bank = np.arctan2(surface_imag_bank, surface_real_bank)

            statistics['minimum phase lag'] = np.amin(surface_phase_entrance - flow_phase_entrance)
            statistics['channel phase lag'] = surface_phase_channel - flow_phase_channel
            statistics['bank phase lag'] = surface_phase_bank - flow_phase_bank

        # VELOCITY STATISTICS
        if vel_stats:
            triangulation = get_triangulation(self.hydro.mesh.ngmesh)
            refiner = tri.UniformTriRefiner(triangulation)
            refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

            sigma_range = np.linspace(-1, 0, num_vertical_samples, endpoint=True)

            U = np.zeros((num_vertical_samples, refined_triangulation.x.shape[0]))
            V = np.zeros((num_vertical_samples, refined_triangulation.x.shape[0]))
            Vmid = np.zeros((num_vertical_samples, num_vertical_samples))
            Umid = np.zeros((num_vertical_samples, num_vertical_samples))

            for i in range(num_vertical_samples):
                U[i,:] = evaluate_CF_range(self.u_abs(1, sigma_range[i]), self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                V[i,:] = evaluate_CF_range(self.v_abs(1, sigma_range[i]), self.hydro.mesh, refined_triangulation.x, refined_triangulation.y)
                Vmid[i, :] = evaluate_CF_range(self.v_abs(1, sigma_range[i]), self.hydro.mesh, L/2 * np.ones(num_vertical_samples), np.linspace(-B/2,B/2, num_vertical_samples))
                Umid[i, :] = evaluate_CF_range(self.u_abs(1, sigma_range[i]), self.hydro.mesh, L/2 * np.ones(num_vertical_samples), np.linspace(-B/2,B/2, num_vertical_samples))


            statistics['mean u'] = np.mean(Umid)
            statistics['median u'] = np.median(Umid)
            maxu_point_index = np.unravel_index(Umid.argmax(), Umid.shape) # Location of maximum is also possibly interesting
            # minu_point_index = np.unravel_index(U.argmin(), U.shape)
            statistics['max u'] = Umid[maxu_point_index]
            # statistics['min u'] = U[minu_point_index]
            statistics['max u location'] = (refined_triangulation.x[maxu_point_index[1]], refined_triangulation.y[maxu_point_index[1]], sigma_range[maxu_point_index[0]])        
            # statistics['min u location'] = (refined_triangulation.x[minu_point_index[1]], refined_triangulation.y[minu_point_index[1]], sigma_range[minu_point_index[0]])        

            statistics['mean v'] = np.mean(Vmid)
            statistics['median v'] = np.median(Vmid)
            maxv_point_index = np.unravel_index(V.argmax(), V.shape) # Location of maximum is also possibly interesting
            # minv_point_index = np.unravel_index(V.argmin(), V.shape)
            statistics['max v'] = V[maxv_point_index]
            # statistics['min v'] = V[minv_point_index]
            statistics['max v location'] = (refined_triangulation.x[maxv_point_index[1]], refined_triangulation.y[maxv_point_index[1]], sigma_range[maxv_point_index[0]])        
            # statistics['min v location'] = (refined_triangulation.x[minv_point_index[1]], refined_triangulation.y[minv_point_index[1]], sigma_range[minv_point_index[0]])        
            maxvmid_point_index = np.unravel_index(Vmid.argmax(), Vmid.shape)
            statistics['max v mid'] = Vmid[maxvmid_point_index]
            statistics['max v mid location'] = (L/2, np.linspace(-B/2,B/2, num_vertical_samples)[maxvmid_point_index[1]], sigma_range[maxvmid_point_index[0]])

        if residual_stats:
            triangulation = get_triangulation(self.hydro.mesh.ngmesh)
            refiner = tri.UniformTriRefiner(triangulation)
            refined_triangulation = refiner.refine_triangulation(subdiv=refinement_level)

            # EXCHANGE RATE: 1. COMPUTE DEPTH-INTEGRATED VELOCITY, 2. COMPUTE MAX{[U],0},  3. INTEGRATE

            H = self.hydro.spatial_physical_parameters['H'].cf
            u_DI = H * self.u_DA[0]
            u_DI_positive_part = ngsolve.IfPos(u_DI, u_DI, 0)

            y = np.linspace(-B/2, B/2, num_vertical_samples)
            dy = y[1] - y[0]

            u_DI_pos_middlesection = evaluate_CF_range(u_DI_positive_part, self.hydro.mesh, L/2*np.ones_like(y), y)
            
            statistics['middle exchange rate'] = dy * np.sum(u_DI_pos_middlesection)

            # INFLOW CENTER OF MASS: USE u_DI_pos_middlesection to evaluate lateral location inflow; vertical location is not computed here
            u_DI_pos_middlesection_probdist = u_DI_pos_middlesection / statistics['middle exchange rate']
            inflow_center = dy * np.sum(u_DI_pos_middlesection_probdist * y)
            statistics['residual inflow center'] = inflow_center


        ## RETURN STATISTICS ##

        return statistics

    def compute_flood_tidal_discharge(self, x, B, num_horizontal_points, num_vertical_points):

        y_range = np.linspace(B/2, -B/2, num_horizontal_points)
        # x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        # y_range = np.linspace(p1[1], p2[1], num_horizontal_points)
        sigma_range = np.linspace(-1, 0, num_vertical_points)

        dy = abs(y_range[1] - y_range[0])
        dsig = abs(sigma_range[1] - sigma_range[0])

        p1 = np.array([x, B/2])
        p2 = np.array([x, -B/2])

        flood_phase = 0.75
        sigma = self.hydro.constant_physical_parameters['sigma']

        H = self.hydro.spatial_physical_parameters['H'].cf

        # depth = evaluate_CF_range(H, self.hydro.mesh, x * np.ones_like(y_range), y_range)

        # y_grid, sig_grid = np.meshgrid(y_range, sigma_range)

        Q = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: H * self.u(1, sig) * self.hydro.time_basis.evaluation_function(flood_phase / sigma, 1) + \
                                                         H * self.u(-1, sig) * self.hydro.time_basis.evaluation_function(flood_phase/sigma, -1), p1, p2, num_horizontal_points, num_vertical_points)
        # area = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: H, p1, p2, num_horizontal_points, num_vertical_points)

        # print(f'Cross sectional area is {dy * dsig * np.sum(area)}')

        # fig, ax = plt.subplots()
        # ax.imshow(Q, cmap='RdBu')
        # plt.show()

        discharge = dy * dsig * np.sum(Q)
        return discharge



        

            




    def plot_vertical_profile_at_point(self, p, num_vertical_points, constituent_index, **kwargs):

        depth = evaluate_CF_point(self.hydro.spatial_physical_parameters['H'].cf, self.hydro.mesh, p[0], p[1])
        z_range = np.linspace(-depth, 0, num_vertical_points)

        if constituent_index == 0:
            u_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.u(0, sigma), p, num_vertical_points)
            v_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.v(0, sigma), p, num_vertical_points)
            w_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.w(0, sigma), p, num_vertical_points)
        elif constituent_index > 0:            
            u_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.u_abs(constituent_index, sigma), p, num_vertical_points)
            v_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.v_abs(constituent_index, sigma), p, num_vertical_points)
            w_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.w_abs(constituent_index, sigma), p, num_vertical_points)

        fig_vertical_profile_point, ax_vertical_profile_point = plt.subplots(1,3)
        ax_vertical_profile_point[0].plot(u_discrete, z_range, label='u', **kwargs)
        ax_vertical_profile_point[0].set_title('u')
        ax_vertical_profile_point[0].axvline(x=0, color='k', linewidth=1.5)
        ax_vertical_profile_point[1].plot(v_discrete, z_range, label='v', **kwargs)
        ax_vertical_profile_point[1].set_title('v')
        ax_vertical_profile_point[1].axvline(x=0, color='k', linewidth=1.5)
        ax_vertical_profile_point[2].plot(w_discrete, z_range, label='w', **kwargs)
        ax_vertical_profile_point[2].set_title('w')
        ax_vertical_profile_point[2].axvline(x=0, color='k', linewidth=1.5)
        for i in range(3):
            ax_vertical_profile_point[i].set_ylabel('Depth [m]')
            ax_vertical_profile_point[i].set_xlabel('Velocity [m/s]')

        constituent_string = f'M{2*constituent_index}'

        plt.suptitle(f'Vertical structure of {constituent_string} velocities at x={p[0]}, y={p[1]}')
        plt.tight_layout()


    def animate_vertical_profile_at_point(self, p, num_vertical_points, num_frames, **kwargs):
        depth = evaluate_CF_point(self.hydro.spatial_physical_parameters['H'].cf, self.hydro.mesh, p[0], p[1])
        z_range = np.linspace(-depth, 0, num_vertical_points)

        T = 1 / self.hydro.constant_physical_parameters['sigma']

        fig_animation, ax_animation = plt.subplots(1, 3)
        # curve_u, = ax_animation.plot([], [], lw=2, **kwargs)

        
        curve_u, = ax_animation[0].plot([], [], **kwargs)
        curve_v, = ax_animation[1].plot([], [], **kwargs)
        curve_w, = ax_animation[2].plot([], [], **kwargs)

        for i in range(3):
            ax_animation[i].set_ylabel('Depth [m]')
            ax_animation[i].set_xlabel('Velocity [m/s]')
            ax_animation[i].set_ylim((-depth, 0))
            ax_animation[i].set_xlim((-0.3,0.3))
        
        ax_animation[0].set_title('u')
        ax_animation[0].axvline(x=0, color='k', linewidth=1.5)
        ax_animation[1].set_title('v')
        ax_animation[1].axvline(x=0, color='k', linewidth=1.5)
        ax_animation[2].set_title('w')
        ax_animation[2].axvline(x=0, color='k', linewidth=1.5)

        plt.suptitle(f'Vertical structure of velocities at x={p[0]}, y={p[1]}')
        plt.tight_layout()

        def init():
            # get initial vertical structures
            u_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.u_timed(0, sigma), p, num_vertical_points)
            v_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.v_timed(0, sigma), p, num_vertical_points)
            w_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.w_timed(0, sigma), p, num_vertical_points)

            curve_u.set_data(u_discrete, z_range)
            curve_v.set_data(v_discrete, z_range)
            curve_w.set_data(w_discrete, z_range)
            return curve_u, curve_v,curve_w,

            
        def update(frame):
            u_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.u_timed(T * frame / num_frames, sigma), p, num_vertical_points)
            v_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.v_timed(T * frame / num_frames, sigma), p, num_vertical_points)
            w_discrete = evaluate_vertical_structure_at_point(self.hydro, lambda sigma: self.w_timed(T * frame / num_frames, sigma), p, num_vertical_points)

            curve_u.set_data(u_discrete, z_range)
            curve_v.set_data(v_discrete, z_range)
            curve_w.set_data(w_discrete, z_range)
            return curve_u, curve_v, curve_w,

        ani = anim.FuncAnimation(fig_animation, update, init_func=init,
                               frames=num_frames, interval=20, blit=True)
        
        ani.save('vertical_structure.mp4', fps=30)
        
        plt.show() # show the animation before the rest of the plots; otherwise the animation is deleted


    def plot_vertical_cross_section(self, quantity_function, title, clabel, p1, p2, num_horizontal_points, num_vertical_points, center_range=False, save=None, contourlines=True, num_levels=None, figsize=(12,6), **kwargs):
        
        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(width/2, -width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)
        sigma_range = np.linspace(-1, 0, num_vertical_points)

        depth = evaluate_CF_range(self.hydro.spatial_physical_parameters['H'].cf, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        Q = evaluate_vertical_structure_at_cross_section(self.hydro, quantity_function, p1, p2, num_horizontal_points, num_vertical_points)

        if center_range:
            maxamp = max(np.amax(Q), -np.amin(Q))


        fig_crosssection, ax_crosssection = plt.subplots(figsize=figsize)
        if center_range:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, **kwargs)
        else:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, **kwargs)
        cbar_crosssection = plt.colorbar(color_crosssection)
        cbar_crosssection.ax.set_ylabel(clabel)

        ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

        if contourlines:

            if num_levels is None:
                num_levels = 8
            subamplitude_lines = 2

            levels = np.linspace(np.min(Q), np.max(Q), num_levels*(subamplitude_lines+1))
            contour = ax_crosssection.contour(s_grid, z_grid, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[.5] * (1+subamplitude_lines))
            ax_crosssection.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
            
        # ax_crosssection.set_xticks(-ax_crosssection.get_xticks())
        ax_crosssection.set_title(title)
        ax_crosssection.set_xlabel('y [m]')
        ax_crosssection.set_ylabel('-Depth [m]')

        plt.tight_layout()

        if save is not None:
            fig_crosssection.savefig(save)


    def plot_cross_section_circulation(self, p1: np.ndarray, p2: np.ndarray, num_horizontal_points: int, num_vertical_points: int, stride: int, phase: float = 0, constituent='all', flowrange: tuple=None):
        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        depth = evaluate_CF_range(self.hydro.spatial_physical_parameters['H'].cf, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        if constituent == 'all':
            Q = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.u_timed(phase / self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.v_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
            W = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.w_timed(phase/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        elif constituent == 0:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            W = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
        else:
            Q = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                             self.u(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            V = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                             self.v(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
            W = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                             self.w(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                             num_horizontal_points, num_vertical_points)
        
        if flowrange is None:
            maxamp = max(np.amax(Q), -np.amin(Q))


        fig_crosssection, ax_crosssection = plt.subplots()
        if flowrange is None:
            color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q, vmin=-maxamp, vmax=maxamp, cmap='bwr')
        cbar_crosssection = plt.colorbar(color_crosssection)
        cbar_crosssection.ax.set_ylabel('Longitudinal velocity [m/s]')

        ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

        

        visual_norms = np.sqrt((V[::stride,::stride] / width)**2 + (W[::stride,::stride] / np.amax(depth))**2) # norm of the vector that we plot
        physical_norms = np.sqrt((V[::stride,::stride])**2 + (W[::stride,::stride])**2)

        # ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[::stride,::stride] / width / 10, W[::stride,::stride] / np.amax(depth) / 10, color='k')
        quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[::stride,::stride] / (width*visual_norms), W[::stride,::stride] / (np.amax(depth)*visual_norms), color='k', alpha= physical_norms / np.amax(physical_norms))


        ax_crosssection.set_title(f'Lateral flow at t = {phase}' + r'$\sigma^{-1}$' f' s\nMaximum lateral velocity = {np.round(np.amax(physical_norms),5)}')


    def plot_cross_section_residual_forcing_mechanisms(self, p1: np.ndarray, p2: np.ndarray, num_horizontal_points, num_vertical_points, figsize=(12,6), cmap='RdBu', savename=None, **kwargs):
        """Plots all of the different forcing mechanisms for along-channel residual currents, along with the total forcing and the resulting residual flow."""

        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(width/2, -width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)
        sigma_range = np.linspace(-1, 0, num_vertical_points)

        H = self.hydro.spatial_physical_parameters['H'].cf

        depth = evaluate_CF_range(H, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        uux = lambda sig: 0.5 * self.hydro.epsilon * (self.u(1, sig) * self.ux(1, sig) + self.u(-1, sig) * self.ux(-1, sig))
        vuy = lambda sig: 0.5 * self.hydro.epsilon * (self.v(1, sig) * self.uy(1, sig) + self.v(-1, sig) * self.uy(-1, sig))
        wuz = lambda sig: 0.5 / H * self.hydro.epsilon * (self.w(1, sig) * self.usig(1, sig) + self.w(-1, sig) * self.usig(-1, sig))
        zetax = lambda sig: self.hydro.constant_physical_parameters['g'] * self.gammax(0)
        fv = lambda sig: -self.hydro.constant_physical_parameters['f'] * self.v(0, sig)

        total_forcing = lambda sig: uux(sig) + vuy(sig) + wuz(sig) + zetax(sig) + fv(sig)
        total_flow = lambda sig: self.u(0, sig)


        UUx = evaluate_vertical_structure_at_cross_section(self.hydro, uux, p1, p2, num_horizontal_points, num_vertical_points)
        VUy = evaluate_vertical_structure_at_cross_section(self.hydro, vuy, p1, p2, num_horizontal_points, num_vertical_points)
        WUz = evaluate_vertical_structure_at_cross_section(self.hydro, wuz, p1, p2, num_horizontal_points, num_vertical_points)
        Zx = evaluate_vertical_structure_at_cross_section(self.hydro, zetax, p1, p2, num_horizontal_points, num_vertical_points)
        fV = evaluate_vertical_structure_at_cross_section(self.hydro, fv, p1, p2, num_horizontal_points, num_vertical_points)
        # F = evaluate_vertical_structure_at_cross_section(self.hydro, total_forcing, p1, p2, num_horizontal_points, num_vertical_points)
        F = UUx + VUy + WUz + Zx + fV
        U = evaluate_vertical_structure_at_cross_section(self.hydro, total_flow, p1, p2, num_horizontal_points, num_vertical_points)
        

        maxUUx = np.amax(np.absolute(UUx))
        maxVUy = np.amax(np.absolute(VUy))
        maxWUz = np.amax(np.absolute(WUz))
        maxZx = np.amax(np.absolute(Zx))
        maxfV = np.amax(np.absolute(fV))
        maxF = np.amax(np.absolute(F))
        maxU = np.amax(np.absolute(U))

        fig, ax = plt.subplots(3, 2, figsize=figsize)
        
        #   UUx     VUy   
        #   WUz     Zx
        #   F       U
        
        UUx_color = ax[0,0].pcolormesh(s_grid, z_grid, UUx, vmin=-maxUUx, vmax=maxUUx, cmap=cmap, **kwargs)
        VUy_color = ax[0,1].pcolormesh(s_grid, z_grid, VUy, vmin=-maxVUy, vmax=maxVUy, cmap=cmap, **kwargs)
        WUz_color = ax[1,0].pcolormesh(s_grid, z_grid, WUz, vmin=-maxWUz, vmax=maxWUz, cmap=cmap, **kwargs)
        Zx_color = ax[1,1].pcolormesh(s_grid, z_grid, Zx, vmin=-maxZx, vmax=maxZx, cmap=cmap, **kwargs)
        F_color = ax[2,1].pcolormesh(s_grid, z_grid, F, vmin=-maxF, vmax=maxF, cmap=cmap, **kwargs)
        # U_color = ax[2,1].pcolormesh(s_grid, z_grid, U, vmin=-maxU, vmax=maxU, cmap=cmap, **kwargs)
        fV_color = ax[2,0].pcolormesh(s_grid, z_grid, fV, vmin=-maxfV, vmax=maxfV, cmap=cmap, **kwargs)

        UUx_cbar = plt.colorbar(UUx_color, ax=ax[0,0])
        VUy_cbar = plt.colorbar(VUy_color, ax=ax[0,1])
        WUz_cbar = plt.colorbar(WUz_color, ax=ax[1,0])
        Zx_cbar = plt.colorbar(Zx_color, ax=ax[1,1])
        F_cbar = plt.colorbar(F_color, ax=ax[2,1])
        fV_cbar = plt.colorbar(fV_color, ax=ax[2,0])

        ax[0,0].set_ylabel('-Depth [m]')
        ax[1,0].set_ylabel('-Depth [m]')
        ax[2,0].set_ylabel('-Depth [m]')
        ax[2,0].set_xlabel('y [m]')
        ax[2,1].set_xlabel('y [m]')

        ax[0,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[1,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[2,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[2,0].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[2,1].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[1,1].plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax[0,1].plot(s_range, -depth, linewidth=1, color='k', zorder=3)

        ax[0,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[1,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[2,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[2,0].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[2,1].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[1,1].fill_between(s_range, -np.amax(depth), -depth, color='silver')
        ax[0,1].fill_between(s_range, -np.amax(depth), -depth, color='silver')

        ax[0,0].set_title(r'$\varepsilon\overline{uu_x}$')
        ax[0,1].set_title(r'$\varepsilon\overline{vu_y}$')
        ax[1,0].set_title(r'$\varepsilon\overline{wu_z}$')
        ax[1,1].set_title(r'$g\overline{\zeta_x}$')
        ax[2,0].set_title(r'$-f\overline{v}$')
        ax[2,1].set_title('All forcing')

        plt.suptitle('Residual forcing mechanisms')
        plt.tight_layout()

        if savename is not None:
            fig.savefig(savename)
    







    def animate_cross_section(self, p1, p2, num_horizontal_points, num_vertical_points, stride, num_frames, constituent='all', mode='savefigs', basename=None, variable='u'):
        
        # Initialize coordinate grids
        
        phase = np.linspace(0, 1, num_frames, endpoint=True)

        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)

        depth = evaluate_CF_range(self.hydro.spatial_physical_parameters['H'].cf, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        # Get flow variables

        if constituent == 'all':
            Q = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            V = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            W = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            for i in range(num_frames):
                Q[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.u_timed(phase[i] / self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
                V[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.v_timed(phase[i]/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
                W[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.w_timed(phase[i]/ self.hydro.constant_physical_parameters['sigma'], sig), p1, p2, num_horizontal_points, num_vertical_points)
        elif constituent == 0:
            Q = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            V = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            W = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            for i in range(num_frames):
                Q[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                V[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                W[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
        else:
            Q = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            V = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            W = np.zeros((num_frames, s_grid.shape[0], s_grid.shape[1]))
            for i in range(num_frames):
                Q[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.u(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                                self.u(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i]/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                V[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.v(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                                self.v(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i]/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                W[i,:,:] = evaluate_vertical_structure_at_cross_section(self.hydro, lambda sig: self.w(constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i] / self.hydro.constant_physical_parameters['sigma'], constituent) + \
                                                                self.w(-constituent, sig) * self.hydro.time_basis.evaluation_function(phase[i]/self.hydro.constant_physical_parameters['sigma'], -constituent), p1, p2,
                                                                num_horizontal_points, num_vertical_points)
                
        # Get maxima

        maxlongi = max(np.amax(Q), np.amax(-Q))
        maxlati = np.amax(np.absolute(V))

        if variable == 'circulation':

            visual_norms = np.sqrt((V[:,::stride,::stride] / width)**2 + (W[:,::stride,::stride] / np.amax(depth))**2) # norm of the vector that we plot
            physical_norms = np.sqrt((V[:,::stride,::stride])**2 + (W[:,::stride,::stride])**2)

            maxlati = np.amax(physical_norms)

            if mode == 'savefigs':
                for i in range(num_frames):
                    fig_crosssection, ax_crosssection = plt.subplots()
            
                    color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q[i,:,:], vmin=-maxlongi, vmax=maxlongi, cmap='bwr')
                    cbar_crosssection = plt.colorbar(color_crosssection)
                    cbar_crosssection.ax.set_ylabel('Longitudinal velocity [m/s]')

                    ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
                    ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

                    quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[i, ::stride,::stride] / (width*visual_norms[i,:,:]), W[i, ::stride,::stride] / (np.amax(depth)*visual_norms[i,:,:]), color='k', alpha= physical_norms[i,:,:] / np.amax(physical_norms))

                    phasestring = str(np.round(phase[i], 5))
                    phasestring += '0' * (7-len(phasestring))

                    ax_crosssection.set_title(f'Lateral flow at t = {phasestring}' + r'$\sigma^{-1}$' f' s\nMaximum lateral velocity = {np.round(np.amax(physical_norms),5)}')

                    fig_crosssection.savefig(f'{basename}_phase{i}.png')

        elif variable == 'u':

            if mode == 'savefigs':
                for i in range(num_frames):
                    fig_crosssection, ax_crosssection = plt.subplots()
            
                    color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, Q[i,:,:], vmin=-maxlongi, vmax=maxlongi, cmap='bwr')
                    cbar_crosssection = plt.colorbar(color_crosssection)
                    cbar_crosssection.ax.set_ylabel('Along-channel velocity [m/s]')

                    ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
                    ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

                    # quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[i, ::stride,::stride] / (width*visual_norms[i,:,:]), W[i, ::stride,::stride] / (np.amax(depth)*visual_norms[i,:,:]), color='k', alpha= physical_norms[i,:,:] / np.amax(physical_norms))

                    phasestring = str(np.round(phase[i], 5))
                    phasestring += '0' * (7-len(phasestring))

                    ax_crosssection.set_title(f'Along-channel velocity at t = {phasestring}' + r'$\sigma^{-1}$')

                    fig_crosssection.savefig(f'{basename}_phase{i}.png')
        
        elif variable == 'v':

            if mode == 'savefigs':
                for i in range(num_frames):
                    fig_crosssection, ax_crosssection = plt.subplots()
            
                    color_crosssection = ax_crosssection.pcolormesh(s_grid, z_grid, V[i,:,:], vmin=-maxlati, vmax=maxlati, cmap='bwr')
                    cbar_crosssection = plt.colorbar(color_crosssection)
                    cbar_crosssection.ax.set_ylabel('Cross-channel velocity [m/s]')

                    ax_crosssection.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
                    ax_crosssection.fill_between(s_range, -np.amax(depth), -depth, color='silver')

                    # quiv = ax_crosssection.quiver(s_grid[::stride,::stride], z_grid[::stride,::stride], V[i, ::stride,::stride] / (width*visual_norms[i,:,:]), W[i, ::stride,::stride] / (np.amax(depth)*visual_norms[i,:,:]), color='k', alpha= physical_norms[i,:,:] / np.amax(physical_norms))

                    phasestring = str(np.round(phase[i], 5))
                    phasestring += '0' * (7-len(phasestring))

                    ax_crosssection.set_title(f'Cross-channel velocity at t = {phasestring}' + r'$\sigma^{-1}$')

                    fig_crosssection.savefig(f'{basename}_phase{i}.png')


    def plot_cross_section_contours(self, quantity_function, quantity_string, num_levels, p1, p2, num_horizontal_points, num_vertical_points, subamplitude_lines=2,**kwargs):
        width = np.linalg.norm(p1-p2, 2)
        s_range = np.linspace(-width/2, width/2, num_horizontal_points)
        x_range = np.linspace(p1[0], p2[0], num_horizontal_points)
        y_range = np.linspace(p1[1], p2[1], num_horizontal_points)
        sigma_range = np.linspace(-1, 0, num_vertical_points)

        depth = evaluate_CF_range(self.hydro.spatial_physical_parameters['H'].cf, self.hydro.mesh, x_range, y_range)

        s_grid = np.tile(s_range, (num_vertical_points, 1))
        z_grid = np.array([np.linspace(-depth[i], 0, num_vertical_points) for i in range(num_horizontal_points)]).T

        Q = evaluate_vertical_structure_at_cross_section(self.hydro, quantity_function, p1, p2, num_horizontal_points, num_vertical_points)

        levels = np.linspace(np.min(Q), np.max(Q), num_levels*(subamplitude_lines+1))

        fig_crosscontour, ax_crosscontour = plt.subplots()
        contourf = ax_crosscontour.contourf(s_grid, z_grid, Q, levels, **kwargs)
        contour = ax_crosscontour.contour(s_grid, z_grid, Q, levels, colors=['k'] + ["0.4"] * subamplitude_lines, linewidths=[0.7]+[0.1]*subamplitude_lines)

        ax_crosscontour.clabel(contour, levels[0::subamplitude_lines+1], inline=1, fontsize=10, fmt='%1.4f')
        ax_crosscontour.set_title(f'Cross section {quantity_string} from ({p1[0], p1[1]}) to ({p2[0], p2[1]})')

        ax_crosscontour.plot(s_range, -depth, linewidth=1, color='k', zorder=3)
        ax_crosscontour.fill_between(s_range, -np.amax(depth), -depth, color='silver')

        cbar = fig_crosscontour.colorbar(contourf)




    

        