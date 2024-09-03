## SCRIPT TO COMPUTE THE FORCINGS ON THE SEAWARD AND RIVERINE BOUNDARY AND ADD THEM TO THE SOLVING PROCEDURE
import ngsolve
import numpy as np
import matplotlib.pyplot as plt
from hydrodynamics import Hydrodynamics
import IntegrationTools
import geometry.geometries
from geometry.create_geometry import RIVER, SEA, BOUNDARY_DICT
from netgen.geom2d import SplineGeometry
import mesh_functions

class RiverineForcing(object):

    def __init__(self, hydro: Hydrodynamics, discharge_amplitude_list, discharge_phase_list, is_constant=False):
        
        self.discharge_amplitudes = discharge_amplitude_list
        self.discharge_phases = discharge_phase_list
        self.hydro = hydro
        self.is_constant = is_constant

        self.discharge_dict = dict() # Use a dictionary to enable negative indices
        self.Q_vec = dict() # vector (\int_0^T Q h_p dt), p = -qmax, ..., qmax

        # fill amplitude and phase lists with zeros for unfilled elements unless is_constant == True and create the vector Q_vec

        if not is_constant:
            for _ in range(hydro.qmax + 1 - len(discharge_amplitude_list)):
                self.discharge_amplitudes.append(0)
                self.discharge_phases.append(0)

            self.discharge_dict[0] = self.discharge_amplitudes[0]
            self.Q_vec[0] = hydro.time_basis.inner_product(0, 0) * self.discharge_dict[0]
            for i in range(1, hydro.qmax + 1):
                self.discharge_dict[i] = self.discharge_amplitudes[i] * ngsolve.cos(self.discharge_phases[i])
                self.discharge_dict[-i] = self.discharge_amplitudes[i] * ngsolve.sin(self.discharge_phases[i])

                self.Q_vec[i] = self.discharge_dict[i] * hydro.time_basis.inner_product(i, i)
                self.Q_vec[-i] = self.discharge_dict[-i] * hydro.time_basis.inner_product(-i, -i)

        else:
            self.discharge_dict[0] = self.discharge_amplitudes[0]
            self.Q_vec[0] = (0.5 / hydro.constant_physical_parameters['sigma']) * self.discharge_dict[0]
        
        # Computation of normal components

        if is_constant and hydro.assumptions.density == 'depth-independent':
    
            d1 = [0.5*(1/hydro.constant_physical_parameters['sigma']) * hydro.constant_physical_parameters['g'] * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M+1)]
            d2 = [hydro.spatial_physical_parameters['H'].cf / (2 * hydro.constant_physical_parameters['sigma']) * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M+1)]

            sum_d1d2 = sum([d1[k]*d2[k] for k in range(hydro.M+1)])
        
            self.normal_alpha = [{0: (-d1[m]/sum_d1d2) * self.Q_vec[0]} for m in range(hydro.M + 1)]
            self.normal_alpha_boundaryCF = [{0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: self.normal_alpha[m][0]}, default=0)} for m in range(hydro.M + 1)]
            for m in range(hydro.M + 1):
                for q in range(1, hydro.qmax + 1):
                    self.normal_alpha[m][q] = 0
                    self.normal_alpha[m][-q] = 0

                    self.normal_alpha_boundaryCF[m][q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: 0}, default=0)
                    self.normal_alpha_boundaryCF[m][-q] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[RIVER]: 0}, default=0)

        elif (not is_constant) and hydro.assumptions.density == 'depth-independent':

            C = [0.25 * (1/hydro.constant_physical_parameters['sigma']) * (k+0.5)*(k+0.5) * np.pi^2 * \
                 (hydro.constant_physical_parameters['Av'] / (hydro.spatial_physical_parameters['H'].cf*hydro.spatial_physical_parameters['H'].cf)) \
                    for k in range(hydro.M + 1)]
            
            d1 = [0.5*(1/hydro.constant_physical_parameters['sigma']) * hydro.constant_physical_parameters['g'] * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M+1)]
            d2 = [hydro.spatial_physical_parameters['H'].cf / (2 * hydro.constant_physical_parameters['sigma']) * \
                  (np.power(-1, k) / ((k+0.5)*np.pi)) for k in range(hydro.M+1)]
            
            c1 = [[1 + 0.25*np.pi*np.pi*q*q*(2/(2*C[k]*C[k]-np.pi*q)) for q in range(1, hydro.qmax + 1)] for k in range(hydro.M + 1)]
            c2 = [[-0.5*np.pi*q*(2*C[k])/(2*C[k]*C[k] - np.pi*q) for q in range(1, hydro.qmax + 1)] for k in range(hydro.M + 1)]
            c3 = [[-0.5*np.pi*q/C[k] for q in range(1, hydro.qmax + 1)] for k in range(hydro.M + 1)]
            c4 = [[4*C[k] / (4*C[k]*C[k]-np.pi*np.pi*q*q) for q in range(1, hydro.qmax + 1)] for k in range(hydro.M + 1)]

            e1 = [-sum([d1[k]*d2[k]*c1[k][q] / C[k] for k in range(hydro.M + 1)]) for q in range(1, hydro.qmax + 1)]
            e2 = [-sum([d1[k]*d2[k]*c2[k][q] / C[k] for k in range(hydro.M + 1)]) for q in range(1, hydro.qmax + 1)]
            e3 = [-sum([d1[k]*d2[k]*c3[k][q] / c4[k][q] for k in range(hydro.M + 1)]) for q in range(1, hydro.qmax + 1)]
            e4 = [-sum([d1[k]*d2[k] / c4[k][q] for k in range(hydro.M + 1)]) for q in range(hydro.qmax + 1)]

            gamma = dict()
            gamma[0] = -self.Q_vec[0] / sum([d1[k]*d2[k] for k in range(hydro.M + 1)])

            for q in range(1, hydro.qmax + 1):
                gamma[q] = e1[q] / (e4[q]*e1[q] - e3[q]) * (self.Q_vec[q] - (e3[q]/e1[q])*self.Q_vec[-q])
                gamma[-q] = (self.Q_vec[-q] - e2[q]*gamma[q]) / e1[q]

            self.normal_alpha = [{0: (-d1[m]/sum_d1d2) * self.Q_vec[0]} for _ in range(hydro.M + 1)]
            self.normal_alpha_boundaryCF = [{0: hydro.mesh.BoundaryCF({RIVER: self.normal_alpha[m][0]}, default=0)} for m in range(hydro.M + 1)]
            for m in range(hydro.M + 1):
                for q in range(1, hydro.qmax + 1):
                    self.normal_alpha[m][q] = d1[m]*c3[m][q]*gamma[-q] / c4[m][q] + d2[m]*gamma[q] / c4[m][q]
                    self.normal_alpha[m][-q] = d1[m]*c1[m][q]*gamma[-q] / C[m] + d1[m]*c2[m][q]*gamma[q] / C[m]

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
        for _ in range(self.hydro.qmax + 1 - len(amplitude_list)):
            self.amplitudes.append(0)
            self.phases.append(0)

        self.cfdict = {0: self.amplitudes[0]}
        self.boundaryCFdict = {0: hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[0]}, default=0)}
        for i in range(1, self.hydro.qmax + 1):
            self.cfdict[i] = self.amplitudes[i] * ngsolve.cos(self.phases[i])
            self.cfdict[-i] = self.amplitudes[i] * ngsolve.sin(self.phases[i])

            self.boundaryCFdict[i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[i]}, default=0)
            self.boundaryCFdict[-i] = hydro.mesh.BoundaryCF({BOUNDARY_DICT[SEA]: self.cfdict[-i]}, default=0)

        
        



if __name__ == '__main__':
    geo = SplineGeometry()
    pnts = [(0,0), (1,0), (1,0.5), (1,1), (0.5,1), (0,1)]


    p1, p2, p3, p4, p5, p6 = [geo.AppendPoint(*pnt) for pnt in pnts]
    curves = [[["line",p1,p2],"bottom"],
            [["line",p2,p3],"right"],
            [["line",p5,p6],"top"],
            [["line",p6,p1],"left"]]

    [geo.Append(c,bc=bc) for c,bc in curves]
    geo.Append(['spline3', p3, p4, p5], bc='curve')
    mesh = ngsolve.Mesh(geo.GenerateMesh(maxh=0.2))

    
    

