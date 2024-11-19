import ngsolve
import ngsolve.solvers
import numpy as np
from truncationbasis import TruncationBasis
import matplotlib.pyplot as plt
import geometry.geometries
from geometry.create_geometry import RIVER, SEA, BOUNDARY_DICT
from netgen.geom2d import SplineGeometry
import mesh_functions
from minusonepower import minusonepower
        
## Public functions that call other functions depending on the assumptions
        
def add_bilinear_part(a: ngsolve.BilinearForm, model_options: dict, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions, 
                      umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax, constant_parameters: dict, spatial_parameters: dict, 
                      vertical_basis: TruncationBasis, normalalpha, forcing=True):
    if model_options['bed_bc'] == 'no_slip' and not model_options['surface_in_sigma'] and model_options['veddy_viscosity_assumption'] == 'constant' and model_options['density'] == 'depth-independent':
        return _add_bilinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                                umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax,
                                                constant_parameters, spatial_parameters, vertical_basis, normalalpha, 
                                                forcing)
    else:
        print("Bilinear part of weak form was not set up")


def add_nonlinear_part(a: ngsolve.BilinearForm, model_options: dict, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                       umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax,
                       constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                       time_basis: TruncationBasis, normalalpha, n):
    if model_options['bed_bc'] == 'no_slip' and not model_options['surface_in_sigma'] and model_options['veddy_viscosity_assumption'] == 'constant' and model_options['density'] == 'depth-independent':
        return _add_nonlinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                                umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax,
                                                constant_parameters, spatial_parameters, vertical_basis, time_basis, normalalpha, model_options['advection_epsilon'],
                                                model_options['advection_influence_matrix'], n)
    else:
        print("Nonlinear part of weak form was not set up")



def add_linearised_nonlinear_part(a: ngsolve.BilinearForm, model_options: dict, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                  umom_testfunctions, vmom_testfunctions, DIC_testfunctions, alpha0, beta0, gamma0, M, imax,
                                  constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                                  time_basis: TruncationBasis, normalalpha, advection_weighting_parameter, n):
    if model_options['bed_bc'] == 'no_slip' and not model_options['surface_in_sigma'] and model_options['veddy_viscosity_assumption'] == 'constant' and model_options['density'] == 'depth-independent':
        return _add_linearised_nonlinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                                umom_testfunctions, vmom_testfunctions, DIC_testfunctions, alpha0, beta0, gamma0, M, imax,
                                                constant_parameters, spatial_parameters, vertical_basis, time_basis, normalalpha, model_options['advection_epsilon'], 
                                                model_options['advection_influence_matrix'], n)
## Private functions that add the weak forms to the ngsolve.BilinearForm object

def _add_bilinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                     umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax,
                                     constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                                     normalalpha, forcing=True):
    
    G3 = vertical_basis.tensor_dict['G3']
    G4 = vertical_basis.tensor_dict['G4']
    G5 = vertical_basis.tensor_dict['G5']

    sig = constant_parameters['sigma']
    Av = constant_parameters['Av']
    f = constant_parameters['f']
    g = constant_parameters['g']

    H = spatial_parameters['H'].cf
    rho = spatial_parameters['density'].cf
    rho_x = spatial_parameters['density'].gradient_cf[0]
    rho_y = spatial_parameters['density'].gradient_cf[1]

    # DEPTH-INTEGRATED CONTINUITY EQUATION

    if forcing:
        a += (0.5 / sig * DIC_testfunctions[0] * H * sum([G4(m) * \
              normalalpha[m][0] for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
        for r in range(1, imax + 1):
            a += (0.5 / sig * DIC_testfunctions[-r] * H * sum([G4(m) * \
                 normalalpha[m][-r] for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
            a += (0.5 / sig * DIC_testfunctions[r] * H * sum([G4(m) * \
                 normalalpha[m][r] for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
            
    # r = 0 term
    a += (-0.5/sig * H * sum([G4(m) * (alpha_trialfunctions[m][0] * ngsolve.grad(DIC_testfunctions[0])[0] + 
                                       beta_trialfunctions[m][0] * ngsolve.grad(DIC_testfunctions[0])[1]) for m in range(M)])) * ngsolve.dx

    # r != 0-terms
    for r in range(1, imax + 1):
        a += (ngsolve.pi*r*gamma_trialfunctions[r]*DIC_testfunctions[-r] - 0.5/sig*H*sum([G4(m) * (
              alpha_trialfunctions[m][-r] * ngsolve.grad(DIC_testfunctions[-r])[0] + 
              beta_trialfunctions[m][-r] * ngsolve.grad(DIC_testfunctions[-r])[1]
        ) for m in range(M)])) * ngsolve.dx
        a += (ngsolve.pi*-r*gamma_trialfunctions[-r]*DIC_testfunctions[r] - 0.5/sig*H*sum([G4(m) * (
              alpha_trialfunctions[m][r] * ngsolve.grad(DIC_testfunctions[r])[0] + 
              beta_trialfunctions[m][r] * ngsolve.grad(DIC_testfunctions[r])[1]
        ) for m in range(M)])) * ngsolve.dx

    
    # MOMENTUM EQUATIONS
        
    for k in range(M):
        # Baroclinic forcing
        if forcing:
            a += (0.5*g*ngsolve.sqrt(2)/sig*G5(k) * H *H * umom_testfunctions[k][0] * rho_x / rho) * ngsolve.dx # U-momentum
            a += (0.5*g*ngsolve.sqrt(2)/sig*G5(k) * H *H * vmom_testfunctions[k][0] * rho_y / rho) * ngsolve.dx # V-momentum
        # r = 0 term
        # U-momentum
        a += (-0.5/sig * Av * G3(k, k) * alpha_trialfunctions[k][0]*umom_testfunctions[k][0] / (H) - 
                0.25*H*f/sig * beta_trialfunctions[k][0] * umom_testfunctions[k][0] + 
                0.5*H*g/sig * G4(k) * ngsolve.grad(gamma_trialfunctions[0])[0] * umom_testfunctions[k][0]) * ngsolve.dx
        # V-momentum
        a += (-0.5/sig * Av * G3(k, k) * beta_trialfunctions[k][0]*vmom_testfunctions[k][0] / (H) +
                0.25*H*f/sig * alpha_trialfunctions[k][0] * vmom_testfunctions[k][0] + 
                0.5*H*g/sig * G4(k) * ngsolve.grad(gamma_trialfunctions[0])[1] * vmom_testfunctions[k][0]) * ngsolve.dx
        
        # r != 0-terms
        for r in range(1, imax + 1):
            # U-momentum
            a += ((0.5*ngsolve.pi*r*H*alpha_trialfunctions[k][r] * umom_testfunctions[k][-r]- 
                    0.5/sig*Av*G3(k,k) * alpha_trialfunctions[k][-r] * umom_testfunctions[k][-r] / (H) - 
                    0.25*H*f/sig * beta_trialfunctions[k][-r] * umom_testfunctions[k][-r] + 
                    0.5*H*g/sig * G4(k) * ngsolve.grad(gamma_trialfunctions[-r])[0] * umom_testfunctions[k][-r]) + 
                    
                    (0.5*ngsolve.pi*H*-r*alpha_trialfunctions[k][-r] * umom_testfunctions[k][r] - 
                    0.5/sig*Av*G3(k,k) * alpha_trialfunctions[k][r] * umom_testfunctions[k][r] / (H) - 
                    0.25*H*f/sig * beta_trialfunctions[k][r] * umom_testfunctions[k][r] + 
                    0.5*H*g/sig * G4(k) * ngsolve.grad(gamma_trialfunctions[r])[0] * umom_testfunctions[k][r])) * ngsolve.dx
            # V-momentum
            a += ((0.5*ngsolve.pi*H*r*beta_trialfunctions[k][r] * vmom_testfunctions[k][-r] - 
                    0.5/sig*Av*G3(k,k) * beta_trialfunctions[k][-r] * vmom_testfunctions[k][-r] / (H) + 
                    0.25*f/sig * H * alpha_trialfunctions[k][-r] * vmom_testfunctions[k][-r] + 
                    0.5*H*g/sig * G4(k) * ngsolve.grad(gamma_trialfunctions[-r])[1] * vmom_testfunctions[k][-r]) + 
                    
                    (0.5*ngsolve.pi*-r*H*beta_trialfunctions[k][-r]*vmom_testfunctions[k][r] - 
                    0.5/sig*Av*G3(k,k) * beta_trialfunctions[k][r]*vmom_testfunctions[k][r] / (H) +
                    0.25*f/sig * H*alpha_trialfunctions[k][r] * vmom_testfunctions[k][r] + 
                    0.5*g/sig * H*G4(k) * ngsolve.grad(gamma_trialfunctions[r])[1] * vmom_testfunctions[k][r])) * ngsolve.dx
            

def _add_nonlinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                      umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax,
                                      constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                                      time_basis: TruncationBasis, normalalpha, advection_weighting_parameter, 
                                      advection_influence_matrix, normal):
    
    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']

    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    H = spatial_parameters['H'].cf
    
    for k in range(M):
        # component r = 0
        # U-momentum
        for p in range(-imax, imax+1):
            for q in range(-imax, imax+1):
                if H3_iszero(p, q, 0):
                    continue
                else:
                    if advection_influence_matrix[0, p] and advection_influence_matrix[0, q]:
                        a += sum([sum([G1(m, n, k) * H3(p, q, 0) * (
                            (-H * advection_weighting_parameter * alpha_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1])) * ngsolve.dx
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        ) - H3(p,q,0) * (G1(m,n,k)+G2(m,n,k)) * (
                            ((-H * advection_weighting_parameter * umom_testfunctions[k][0] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha_trialfunctions[n][p] + 
                                                                    ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta_trialfunctions[n][p]))
                            - (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + 
                                                                    beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1]))) * ngsolve.dx
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        ) for m in range(M)]) for n in range(M)])
                        # V-momentum
                        a += sum([sum([G1(m, n, k) * H3(p, q, 0) * (
                            (-H * advection_weighting_parameter * beta_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1])) * ngsolve.dx
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        ) - H3(p,q,0) * (G1(m,n,k)+G2(m,n,k)) * (
                            ((-H * advection_weighting_parameter * vmom_testfunctions[k][0] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha_trialfunctions[n][p] + 
                                                                    ngsolve.grad(beta_trialfunctions[m][q])[1] * beta_trialfunctions[n][p]))
                            - (H * advection_weighting_parameter * beta_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + 
                                                                    beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1]))) * ngsolve.dx
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        ) for m in range(M)]) for n in range(M)])

        for r in range(1, imax + 1):
            # add nonlinear part of a
            #component -r
            for p in range(-imax, imax+1):
                for q in range(-imax, imax+1):
                    if not H3_iszero(p, q, -r):
                        if advection_influence_matrix[r, p] and advection_influence_matrix[r, q]:
                            # U-momentum
                            a += sum([sum([G1(m, n, k) * H3(p, q, -r) * (
                                (-H * advection_weighting_parameter * alpha_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1])) * ngsolve.dx
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            ) - H3(p,q,-r) * (G1(m,n,k)+G2(m,n,k)) * ( 
                            ((-H * advection_weighting_parameter * umom_testfunctions[k][-r] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha_trialfunctions[n][p] + 
                                                                    ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta_trialfunctions[n][p])) 
                            - (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + 
                                                                    beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1]))) * ngsolve.dx)
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            for m in range(M)]) for n in range(M)])

                            
                            
                            # V-momentum component -r
                            a += sum([sum([G1(m, n, k) * H3(p, q, -r) * (
                                (-H * advection_weighting_parameter * beta_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1])) * ngsolve.dx
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            ) - H3(p,q,-r) * (G1(m,n,k)+G2(m,n,k)) * (
                                ((-H * advection_weighting_parameter * vmom_testfunctions[k][-r] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha_trialfunctions[n][p] + 
                                                                        ngsolve.grad(beta_trialfunctions[m][q])[1] * beta_trialfunctions[n][p]))
                            - (H * advection_weighting_parameter * beta_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1]))) * ngsolve.dx)
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            for m in range(M)]) for n in range(M)])

                        
                    
                    if not H3_iszero(p, q, r):
                        if advection_influence_matrix[r, p] and advection_influence_matrix[r, q]:
                            # U-momentum component +r
                            a += sum([sum([G1(m, n, k) * H3(p, q, r) * (
                                (-H * advection_weighting_parameter * alpha_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1])) * ngsolve.dx
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            ) - H3(p,q,r) * (G1(m,n,k)+G2(m,n,k)) * (
                                ((-H * advection_weighting_parameter * umom_testfunctions[k][r] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha_trialfunctions[n][p] + 
                                                                        ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta_trialfunctions[n][p]))
                            - (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1]))) * ngsolve.dx
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            ) for m in range(M)]) for n in range(M)])

                            
                            
                            # V-momentum component +r
                            a += sum([sum([G1(m, n, k) * H3(p, q, r) * (
                                (-H * advection_weighting_parameter * beta_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1])) * ngsolve.dx
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            ) - H3(p,q,r) * (G1(m,n,k)+G2(m,n,k)) * (
                                ((-H * advection_weighting_parameter * vmom_testfunctions[k][r] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha_trialfunctions[n][p] + 
                                                                        ngsolve.grad(beta_trialfunctions[m][q])[1] * beta_trialfunctions[n][p]))
                            - (H * advection_weighting_parameter * beta_trialfunctions[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + 
                                                                        beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1]))) * ngsolve.dx

                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * normalalpha[n][p]) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            + (H * advection_weighting_parameter * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * (normal[0]*alpha_trialfunctions[n][p] + normal[1]*beta_trialfunctions[n][p])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            ) for m in range(M)]) for n in range(M)])
                        

            
def _add_linearised_nonlinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                      umom_testfunctions, vmom_testfunctions, DIC_testfunctions, alpha0, beta0, gamma0, M, imax,
                                      constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                                      time_basis: TruncationBasis, normalalpha, advection_weighting_parameter, 
                                      advection_influence_matrix, normal):
    
    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']

    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    H = spatial_parameters['H'].cf
    
    for k in range(M):
        for p in range(-imax, imax + 1):
            for q in range(-imax, imax + 1):
                if H3_iszero(p, q, 0):
                    continue
                else:
                    if advection_influence_matrix[0, p] and advection_influence_matrix[0, q]:
                        # interior domain integration for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                            -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1]) - \
                            H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1])
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.dx
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
                            -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1]) - \
                            H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1]) - 
                            H * umom_testfunctions[k][0] * (ngsolve.grad(alpha0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                ngsolve.grad(alpha0[m][q])[1] * beta_trialfunctions[n][p]) -
                            H * umom_testfunctions[k][0] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta0[n][p])
                        ) for n in range(M)])for m in range(M)]))*ngsolve.dx
                        # integration over seaward boundary for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                            H * alpha0[m][q] * umom_testfunctions[k][0] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * alpha0[m][q] * umom_testfunctions[k][0] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        # integration over riverine boundary for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * normalalpha[n][p]
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * normalalpha[n][p]
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        # interior domain integration for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                            -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1]) - \
                            H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1])
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.dx
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
                            -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1]) - \
                            H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1]) - 
                            H * vmom_testfunctions[k][0] * (ngsolve.grad(beta0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                ngsolve.grad(beta0[m][q])[1] * beta_trialfunctions[n][p]) -
                            H * vmom_testfunctions[k][0] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                ngsolve.grad(beta_trialfunctions[m][q])[1] * beta0[n][p])
                        ) for n in range(M)])for m in range(M)]))*ngsolve.dx
                        # integration over seaward boundary for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                            H * beta0[m][q] * vmom_testfunctions[k][0] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * beta0[m][q] * vmom_testfunctions[k][0] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        # integration over riverine boundary for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * normalalpha[n][p]
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * normalalpha[n][p]
                        ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            
                # terms r != 0
        for r in range(1, imax + 1):
            for p in range(-imax, imax + 1):
                for q in range(-imax, imax + 1):
                    if H3_iszero(p, q, r):
                        continue
                    else:
                        if advection_influence_matrix[r, p] and advection_influence_matrix[r, q]:
                            # terms -r
                            # interior domain integration for u-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                                -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1]) - \
                                H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1])
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.dx
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
                                -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1]) - \
                                H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1]) - 
                                H * umom_testfunctions[k][-r] * (ngsolve.grad(alpha0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                    ngsolve.grad(alpha0[m][q])[1] * beta_trialfunctions[n][p]) -
                                H * umom_testfunctions[k][-r] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                    ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta0[n][p])
                            ) for n in range(M)])for m in range(M)]))*ngsolve.dx
                            # integration over seaward boundary for u-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                                H * alpha0[m][q] * umom_testfunctions[k][-r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * alpha0[m][q] * umom_testfunctions[k][-r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            # integration over riverine boundary for u-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            # interior domain integration for v-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                                -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1]) - \
                                H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1])
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.dx
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
                                -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1]) - \
                                H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1]) - 
                                H * vmom_testfunctions[k][-r] * (ngsolve.grad(beta0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                    ngsolve.grad(beta0[m][q])[1] * beta_trialfunctions[n][p]) -
                                H * vmom_testfunctions[k][-r] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                    ngsolve.grad(beta_trialfunctions[m][q])[1] * beta0[n][p])
                            ) for n in range(M)])for m in range(M)]))*ngsolve.dx
                            # integration over seaward boundary for v-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                                H * beta0[m][q] * vmom_testfunctions[k][-r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * beta0[m][q] * vmom_testfunctions[k][-r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            # integration over riverine boundary for v-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])

                            # terms +r
                            # interior domain integration for u-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                                -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1]) - \
                                H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1])
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.dx
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
                                -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1]) - \
                                H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1]) - 
                                H * umom_testfunctions[k][r] * (ngsolve.grad(alpha0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                    ngsolve.grad(alpha0[m][q])[1] * beta_trialfunctions[n][p]) -
                                H * umom_testfunctions[k][r] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                    ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta0[n][p])
                            ) for n in range(M)])for m in range(M)]))*ngsolve.dx
                            # integration over seaward boundary for u-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                                H * alpha0[m][q] * umom_testfunctions[k][r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * alpha0[m][q] * umom_testfunctions[k][r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            # integration over riverine boundary for u-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            # interior domain integration for v-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                                -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1]) - \
                                H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1])
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.dx
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
                                -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + 
                                                    beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1]) - \
                                H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + \
                                                                        beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1]) - 
                                H * vmom_testfunctions[k][r] * (ngsolve.grad(beta0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                    ngsolve.grad(beta0[m][q])[1] * beta_trialfunctions[n][p]) -
                                H * vmom_testfunctions[k][r] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                    ngsolve.grad(beta_trialfunctions[m][q])[1] * beta0[n][p])
                            ) for n in range(M)])for m in range(M)]))*ngsolve.dx
                            # integration over seaward boundary for v-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                                H * beta0[m][q] * vmom_testfunctions[k][r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * beta0[m][q] * vmom_testfunctions[k][r] * (
                                    alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                                ) + 
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * (
                                    alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                                )
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                            # integration over riverine boundary for v-momentum
                            a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                                H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * normalalpha[n][p]
                            ) for n in range(M)]) for m in range(M)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])



def add_weak_form(a: ngsolve.BilinearForm, model_options: dict, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                       umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax,
                       constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                       time_basis: TruncationBasis, normalalpha, only_linear = False):
    """
    Constructs a weak form of the model equations (as an ngsolve.BilinearForm-object) by adding it to an empty one.

    Arguments:
    
        - a (ngsolve.BilinearForm):                     bilinear form object on which the weak form is constructed;
        - model_options (dict):                         model options of the hydrodynamics object;
        - alpha_trialfunctions (dict):                  dictionary of trial functions representing the alpha-Fourier/eigenfunction coefficients (obtainable via hydrodynamics._setup_TnT)
        - beta_trialfunctions (dict):                   dictionary of trial functions representing the beta-Fourier/eigenfunction coefficients;
        - gamma_trialfunctions (dict):                  dictionary of trial functions representing the gamma-Fourier/eigenfunction coefficients;
        - umom_testfunctions (dict):                    dictionary of test functions for the along-channel momentum equations;
        - vmom_testfunctions (dict):                    dictionary of test functions for the lateral momentum equations;
        - DIC_testfunctions (dict):                     dictionary of test functions for the depth-integrated continuity equation;
        - M (int):                                      number of vertical basis functions;
        - imax (int):                                   number of tidal constituents excluding residual flow;
        - constant_parameters (dict):                   dictionary of constant physical parameters associated with the model;
        - spatial_parameters (dict):                    dictionary of spatial physical parameters (as SpatialParameter objects) associated with the model;
        - vertical_basis (TruncationBasis):             vertical eigenfunction basis;
        - time_basis (TruncationBasis):                 temporal Fourier basis;
        - normalalpha (dict):                           dictionary of coefficient functions containing the riverine normal flow for that boundary condition;
        - x_scaling (float):                            characteristic length scale by which the domain is scaled;
        - y_scaling (float):                            characteristic width scale by which the domain is scaled;
        - only_linear (bool):                           flag indicating whether the function should only construct the truly bilinear components of the weak form (the linear part of the equation);

    """
    
    # print("Warning: the model can currently only work if a specific temporal basis is selected. Furthermore, it assumes that the vertical basis consists of eigenfunctions of the vertical mixing operator." + \
    #       "Furthermore, it is assumed that the normal to the riverine (right) and seaward boundary (left) has no lateral component; otherwise, the riverine boundary condition cannot be implemented.")


    # Checking compatibility of model options
    if model_options['surface_in_sigma']:
        print("Adding non-linear effects stemming from the presence of the free surface in the sigma-coordinates is currently not implemented. Did not commence constructing weak form.")
        return
    
    if model_options['density'] != 'depth-independent':
        print("Depth-dependent density profiles are currently not implemented yet. Did not commence constructing weak form.")
        return

    # Defining shorthands of variables

    H = spatial_parameters['H'].cf
    Hx = spatial_parameters["H"].gradient_cf[0]
    Hy = spatial_parameters['H'].gradient_cf[1]
    rho = spatial_parameters['density'].cf
    rhox = spatial_parameters['density'].gradient_cf[0]
    rhoy = spatial_parameters['density'].gradient_cf[1]
    R = spatial_parameters['R'].cf
    Rx = spatial_parameters['R'].gradient_cf[0]
    Ry = spatial_parameters['R'].gradient_cf[1]
    ramp = spatial_parameters['non-linear_ramp'].cf

    f = constant_parameters['f']
    g = constant_parameters['g']
    Av = constant_parameters['Av']
    if model_options['horizontal_diffusion']:
        Ah = constant_parameters['Ah']
    sigma = constant_parameters['sigma']

    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']
    G3 = vertical_basis.tensor_dict['G3']
    G4 = vertical_basis.tensor_dict['G4']

    vertical_innerproduct = vertical_basis.inner_product

    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    x_scaling = model_options['x_scaling']
    y_scaling = model_options['y_scaling']

    advection_matrix = model_options['advection_influence_matrix']
    advection_epsilon = model_options['advection_epsilon']

    # 1: Depth-integrated continuity equation ====================================================================================================================
    # See project notes for an analytical expression of these weak forms
    if model_options['horizontal_diffusion']: # difference is that we do not integrate by parts in this equation if horizontal diffusion is present; boundary conditions are included in the momentum equations now
        a += sum([0.5 * G4(m) * (Hx + Rx) * alpha_trialfunctions[m][0] * DIC_testfunctions[0] / x_scaling * ngsolve.dx for m in range(M)])
        a += sum([0.5 * G4(m) * (Hy + Ry) * beta_trialfunctions[m][0] * DIC_testfunctions[0] / y_scaling * ngsolve.dx for m in range(M)])
        a += sum([0.5 * G4(m) * (H + R) * DIC_testfunctions[0] * (ngsolve.grad(alpha_trialfunctions[m][0])[0] / x_scaling + ngsolve.grad(beta_trialfunctions[m][0])[1] / y_scaling) * ngsolve.dx for m in range(M)])

        # terms l != 0
        for l in range(1, imax + 1):
            a += sigma * np.pi * l * DIC_testfunctions[-l] * gamma_trialfunctions[l] * ngsolve.dx
            a += sum([0.5 * G4(m) * (Hx + Rx) * alpha_trialfunctions[m][-l] * DIC_testfunctions[-l] / x_scaling * ngsolve.dx for m in range(M)])
            a += sum([0.5 * G4(m) * (Hy + Ry) * beta_trialfunctions[m][-l] * DIC_testfunctions[-l] / y_scaling * ngsolve.dx for m in range(M)])
            a += sum([0.5 * G4(m) * (H + R) * DIC_testfunctions[-l] * (ngsolve.grad(alpha_trialfunctions[m][-l])[0] / x_scaling + ngsolve.grad(beta_trialfunctions[m][-l])[1] / y_scaling) * ngsolve.dx for m in range(M)])

            a += sigma * np.pi * -l * DIC_testfunctions[l] * gamma_trialfunctions[-l] * ngsolve.dx
            a += sum([0.5 * G4(m) * (Hx + Rx) * alpha_trialfunctions[m][l] * DIC_testfunctions[l] / x_scaling * ngsolve.dx for m in range(M)])
            a += sum([0.5 * G4(m) * (Hy + Ry) * beta_trialfunctions[m][l] * DIC_testfunctions[l] / y_scaling * ngsolve.dx for m in range(M)])
            a += sum([0.5 * G4(m) * (H + R) * DIC_testfunctions[l] * (ngsolve.grad(alpha_trialfunctions[m][l])[0] / x_scaling + ngsolve.grad(beta_trialfunctions[m][l])[1] / y_scaling) * ngsolve.dx for m in range(M)])

    else:
        a += sum([0.5 * DIC_testfunctions[0] * (H+R) * G4(m) * normalalpha[m]  / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)])
        a += sum([G4(m) * -0.5 * (H+R) * (ngsolve.grad(DIC_testfunctions[0])[0] * alpha_trialfunctions[m][0] / x_scaling +
                                            ngsolve.grad(DIC_testfunctions[0])[1] * beta_trialfunctions[m][0] / y_scaling)  * ngsolve.dx for m in range(0, M)])
        
            # terms l != 0
        for l in range(1, imax + 1):
            a += sigma * np.pi * l * DIC_testfunctions[-l] * gamma_trialfunctions[l] * ngsolve.dx
            # a += sum([0.5 * DIC_testfunctions[-l] * (H+R) * G4(m) * normalalpha[m][-l]  / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) # --> This is zero because we assume the river discharge is stationary
            a += sum([-0.5 * (H+R) * G4(m) * (ngsolve.grad(DIC_testfunctions[-l])[0] * alpha_trialfunctions[m][-l] / x_scaling +
                                                ngsolve.grad(DIC_testfunctions[-l])[1] * beta_trialfunctions[m][-l] / y_scaling)  * ngsolve.dx for m in range(0, M)])
            # l > 0
            a += sigma * np.pi * -l * DIC_testfunctions[l] * gamma_trialfunctions[-l] * ngsolve.dx
            # a += sum([0.5 * DIC_testfunctions[l] * (H+R) * G4(m) * normalalpha[m][l]  / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) # --> Likewise
            a += sum([-0.5 * (H+R) * G4(m) * (ngsolve.grad(DIC_testfunctions[l])[0] * alpha_trialfunctions[m][l] / x_scaling +
                                                ngsolve.grad(DIC_testfunctions[l])[1] * beta_trialfunctions[m][l] / y_scaling)  * ngsolve.dx for m in range(0, M)])
    
        
    # 2: Momentum equations =========================================================================================================================
    # For analytical forms of these weak forms, see Project Notes
    if model_options['horizontal_diffusion']:
        for p in range(M): # loop through all vertical components
            # term l = 0
            if not only_linear:
                for i in range(-imax, imax + 1):
                    for j in range(-imax, imax + 1):
                        if H3_iszero(i,j,0):
                            continue
                        else:
                            if advection_matrix[0, abs(i)] and advection_matrix[0, abs(j)]:
                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling *ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (
                                    (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                                    (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (
                                    (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                                    (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
            
            # Coriolis
            a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * beta_trialfunctions[p][0] * umom_testfunctions[p][0] * ngsolve.dx
            a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * alpha_trialfunctions[p][0] * vmom_testfunctions[p][0] * ngsolve.dx
            # Barotropic pressure gradient
            a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[0])[0] * umom_testfunctions[p][0] / x_scaling * ngsolve.dx
            a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[0])[1] * vmom_testfunctions[p][0] / y_scaling * ngsolve.dx
            # Baroclinic pressure gradient
            a += (1/x_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * umom_testfunctions[p][0] * rhox / rho * ngsolve.dx # assumes density is depth-independent
            a += (1/y_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * vmom_testfunctions[p][0] * rhoy / rho * ngsolve.dx
            # Vertical eddy viscosity
            if model_options['veddy_viscosity_assumption'] == 'constant':
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][0] * umom_testfunctions[p][0] / (H+R) * ngsolve.dx# assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][0] * vmom_testfunctions[p][0] / (H+R) * ngsolve.dx
            elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][0] * umom_testfunctions[p][0] *ngsolve.dx# assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][0] * vmom_testfunctions[p][0] * ngsolve.dx
            # Horizontal eddy viscosity
            a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                ngsolve.grad(alpha_trialfunctions[p][0])[0] * ((H+R)*ngsolve.grad(umom_testfunctions[p][0])[0] / (x_scaling**2) + umom_testfunctions[p][0]*(Hx+Rx) / (x_scaling**2)) + \
                ngsolve.grad(alpha_trialfunctions[p][0])[1] * ((H+R)*ngsolve.grad(umom_testfunctions[p][0])[1] / (y_scaling**2) + umom_testfunctions[p][0]*(Hy+Ry) / (y_scaling**2))
            ) * ngsolve.dx
            a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                ngsolve.grad(beta_trialfunctions[p][0])[0] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][0])[0] / (x_scaling**2) + vmom_testfunctions[p][0]*(Hx+Rx) / (x_scaling**2)) + \
                ngsolve.grad(beta_trialfunctions[p][0])[1] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][0])[1] / (y_scaling**2) + vmom_testfunctions[p][0]*(Hy+Ry) / (y_scaling**2))
            ) * ngsolve.dx

            # Terms l != 0
            for l in range(1, imax + 1):
                # Local acceleration
                a += vertical_innerproduct(p, p) * np.pi * l * (H+R) * sigma * umom_testfunctions[p][-l] * alpha_trialfunctions[p][l] * ngsolve.dx# factor 0.5 from vertical projection coefficient
                a += vertical_innerproduct(p, p) * np.pi * -l * (H+R) * sigma * umom_testfunctions[p][l] * alpha_trialfunctions[p][-l] * ngsolve.dx
                a += vertical_innerproduct(p, p) * np.pi * l *  (H+R) * sigma * vmom_testfunctions[p][-l] * beta_trialfunctions[p][l] * ngsolve.dx # factor 0.5 from vertical projection coefficient
                a += vertical_innerproduct(p, p) * np.pi * -l *  (H+R) * sigma * vmom_testfunctions[p][l] * beta_trialfunctions[p][-l] * ngsolve.dx
                for i in range(-imax, imax + 1):
                    for j in range(-imax, imax + 1):
                        if H3_iszero(i,j,-l) and H3_iszero(i,j,l):
                            continue 
                        else:
                            if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:
                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                # Coriolis
                a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * beta_trialfunctions[p][-l] * umom_testfunctions[p][-l] * ngsolve.dx
                a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * alpha_trialfunctions[p][-l] * vmom_testfunctions[p][-l] * ngsolve.dx
                # Barotropic pressure gradient
                a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[-l])[0] * umom_testfunctions[p][-l] / x_scaling * ngsolve.dx
                a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[-l])[1] * vmom_testfunctions[p][-l] / y_scaling * ngsolve.dx
                # Vertical eddy viscosity
                if model_options['veddy_viscosity_assumption'] == 'constant':
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][-l] * umom_testfunctions[p][-l] / (H+R) * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][-l] * vmom_testfunctions[p][-l] / (H+R) * ngsolve.dx
                elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][-l] * umom_testfunctions[p][-l] *ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][-l] * vmom_testfunctions[p][-l] * ngsolve.dx
                # Horizontal eddy viscosity
                a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                    ngsolve.grad(alpha_trialfunctions[p][-l])[0] * ((H+R)*ngsolve.grad(umom_testfunctions[p][-l])[0] / (x_scaling**2) + umom_testfunctions[p][-l]*(Hx+Rx) / (x_scaling**2)) + \
                    ngsolve.grad(alpha_trialfunctions[p][-l])[1] * ((H+R)*ngsolve.grad(umom_testfunctions[p][-l])[1] / (y_scaling**2) + umom_testfunctions[p][-l]*(Hy+Ry) / (y_scaling**2))
                ) * ngsolve.dx
                a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                    ngsolve.grad(beta_trialfunctions[p][-l])[0] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][-l])[0] / (x_scaling**2) + vmom_testfunctions[p][-l]*(Hx+Rx) / (x_scaling**2)) + \
                    ngsolve.grad(beta_trialfunctions[p][-l])[1] * ((H+R)*ngsolve.grad(vmom_testfunctions[p][-l])[1] / (y_scaling**2) + vmom_testfunctions[p][-l]*(Hy+Ry) / (y_scaling**2))
                ) * ngsolve.dx

                # Coriolis
                a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * beta_trialfunctions[p][l] * umom_testfunctions[p][l] * ngsolve.dx
                a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * alpha_trialfunctions[p][l] * vmom_testfunctions[p][l] * ngsolve.dx
                # Barotropic pressure gradient
                a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[l])[0] * umom_testfunctions[p][l] / x_scaling * ngsolve.dx
                a += 0.5 * (H+R) * g * G4(p) * ngsolve.grad(gamma_trialfunctions[l])[1] * vmom_testfunctions[p][l] / y_scaling * ngsolve.dx
                # Vertical eddy viscosity
                if model_options['veddy_viscosity_assumption'] == 'constant':
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][l] * umom_testfunctions[p][l] / (H+R) * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][l] * vmom_testfunctions[p][l] / (H+R) * ngsolve.dx
                elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][l] * umom_testfunctions[p][l] *ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][l] * vmom_testfunctions[p][l] * ngsolve.dx
                # Horizontal eddy viscosity
                a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                    ngsolve.grad(alpha_trialfunctions[p][l])[0] / (x_scaling**2) * ((H+R)*ngsolve.grad(umom_testfunctions[p][l])[0] + umom_testfunctions[p][l]*(Hx+Rx)) + \
                    ngsolve.grad(alpha_trialfunctions[p][l])[1] / (y_scaling**2) * ((H+R)*ngsolve.grad(umom_testfunctions[p][l])[1] + umom_testfunctions[p][l]*(Hy+Ry))
                ) * ngsolve.dx
                a += 0.5 * vertical_innerproduct(p, p) * Ah * (
                    ngsolve.grad(beta_trialfunctions[p][l])[0] / (x_scaling**2) * ((H+R)*ngsolve.grad(vmom_testfunctions[p][l])[0] + vmom_testfunctions[p][l]*(Hx+Rx)) + \
                    ngsolve.grad(beta_trialfunctions[p][l])[1] / (y_scaling**2) * ((H+R)*ngsolve.grad(vmom_testfunctions[p][l])[1] + vmom_testfunctions[p][l]*(Hy+Ry))
                ) * ngsolve.dx

    else:
        for p in range(0, M): # loop through all vertical components
            # term l = 0
            if not only_linear:

                for i in range(-imax, imax + 1):
                    for j in range(-imax, imax + 1):
                        if H3_iszero(i,j,0): # if H^3_{i,j,l} is equal to zero, this iteration can just be skipped;
                            continue
                        else:
                            if advection_matrix[0, abs(i)] and advection_matrix[0, abs(j)]:
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling  * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * G2(m,n,p) * (
                                    alpha_trialfunctions[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][0])[0] / x_scaling +
                                    ngsolve.grad(alpha_trialfunctions[m][i])[0] * alpha_trialfunctions[n][j] * umom_testfunctions[p][0] / x_scaling +
                                    alpha_trialfunctions[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][0])[1] / y_scaling +
                                    ngsolve.grad(alpha_trialfunctions[m][i])[1] * beta_trialfunctions[n][j] * umom_testfunctions[p][0] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * G2(m,n,p) * (
                                    beta_trialfunctions[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][0])[0] / x_scaling +
                                    ngsolve.grad(beta_trialfunctions[m][i])[0] * alpha_trialfunctions[n][j] * vmom_testfunctions[p][0] / x_scaling +
                                    beta_trialfunctions[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][0])[1] / y_scaling +
                                    ngsolve.grad(beta_trialfunctions[m][i])[1] * beta_trialfunctions[n][j] * vmom_testfunctions[p][0] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G2(m,n,p) * alpha_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G2(m,n,p) * alpha_trialfunctions[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G2(m,n,p) * beta_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G2(m,n,p) * beta_trialfunctions[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                        
            a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * umom_testfunctions[p][0] * beta_trialfunctions[p][0] * ngsolve.dx
            a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * vmom_testfunctions[p][0] * alpha_trialfunctions[p][0] * ngsolve.dx
            # factor 0.25 is from the assumed projection coefficients H1 and G0

            a += 0.5 * g * (H+R) * umom_testfunctions[p][0] * G4(p) * ngsolve.grad(gamma_trialfunctions[0])[0] / x_scaling * ngsolve.dx # assumes density is depth-independent
            a += 0.5 * g * (H+R) * vmom_testfunctions[p][0] * G4(p) * ngsolve.grad(gamma_trialfunctions[0])[1] / y_scaling * ngsolve.dx

            a += (1/x_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * umom_testfunctions[p][0] * rhox / rho * ngsolve.dx# assumes density is depth-independent
            a += (1/y_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * vmom_testfunctions[p][0] * rhoy / rho * ngsolve.dx

            if model_options['veddy_viscosity_assumption'] == 'constant':
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][0] * umom_testfunctions[p][0] / (H+R) * ngsolve.dx# assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][0] * vmom_testfunctions[p][0] / (H+R) * ngsolve.dx
            elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][0] * umom_testfunctions[p][0] *ngsolve.dx# assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][0] * vmom_testfunctions[p][0] * ngsolve.dx
            
            # terms l != 0
            for l in range(1, imax + 1):

                a += vertical_innerproduct(p, p) * np.pi * l * (H+R) * sigma * umom_testfunctions[p][-l] * alpha_trialfunctions[p][l] * ngsolve.dx# factor 0.5 from vertical projection coefficient
                a += -vertical_innerproduct(p, p) * np.pi * l * (H+R) * sigma * umom_testfunctions[p][l] * alpha_trialfunctions[p][-l] * ngsolve.dx
                a += vertical_innerproduct(p, p) * np.pi * l *  (H+R) * sigma * vmom_testfunctions[p][-l] * beta_trialfunctions[p][l] * ngsolve.dx # factor 0.5 from vertical projection coefficient
                a += -vertical_innerproduct(p, p) * np.pi * l *  (H+R) * sigma * vmom_testfunctions[p][l] * beta_trialfunctions[p][-l] * ngsolve.dx

                if not only_linear:
                    for i in range(-imax, imax + 1):
                        for j in range(-imax, imax + 1):

                            if H3_iszero(i,j,-l): # if H^3_{i,j,l} is equal to zero, this iteration can just be skipped;
                                continue
                            else:
                                if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * G2(m,n,p) * (
                                        alpha_trialfunctions[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][-l])[0] / x_scaling +
                                        ngsolve.grad(alpha_trialfunctions[m][i])[0] * alpha_trialfunctions[n][j] * umom_testfunctions[p][0] / x_scaling +
                                        alpha_trialfunctions[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][-l])[1] / y_scaling +
                                        ngsolve.grad(alpha_trialfunctions[m][i])[1] * beta_trialfunctions[n][j] * umom_testfunctions[p][0] / y_scaling
                                    ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * G2(m,n,p) * (
                                        beta_trialfunctions[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][-l])[0] / x_scaling +
                                        ngsolve.grad(beta_trialfunctions[m][i])[0] * alpha_trialfunctions[n][j] * vmom_testfunctions[p][0] / x_scaling +
                                        beta_trialfunctions[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][-l])[1] / y_scaling +
                                        ngsolve.grad(beta_trialfunctions[m][i])[1] * beta_trialfunctions[n][j] * vmom_testfunctions[p][0] / y_scaling
                                    ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G2(m,n,p) * alpha_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G2(m,n,p) * alpha_trialfunctions[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G2(m,n,p) * beta_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G2(m,n,p) * beta_trialfunctions[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]

                            if H3_iszero(i,j,l): # if H^3_{i,j,l} is equal to zero, this iteration can just be skipped;
                                continue
                            else:
                                if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:                              
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * G2(m,n,p) * (
                                        alpha_trialfunctions[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][l])[0] / x_scaling +
                                        ngsolve.grad(alpha_trialfunctions[m][i])[0] * alpha_trialfunctions[n][j] * umom_testfunctions[p][0] / x_scaling +
                                        alpha_trialfunctions[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][l])[1] / y_scaling +
                                        ngsolve.grad(alpha_trialfunctions[m][i])[1] * beta_trialfunctions[n][j] * umom_testfunctions[p][0] / y_scaling
                                    ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * G2(m,n,p) * (
                                        beta_trialfunctions[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][l])[0] / x_scaling +
                                        ngsolve.grad(beta_trialfunctions[m][i])[0] * alpha_trialfunctions[n][j] * vmom_testfunctions[p][0] / x_scaling +
                                        beta_trialfunctions[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][l])[1] / y_scaling +
                                        ngsolve.grad(beta_trialfunctions[m][i])[1] * beta_trialfunctions[n][j] * vmom_testfunctions[p][0] / y_scaling
                                    ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][-l] * G2(m,n,p) * alpha_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * umom_testfunctions[p][-l] * G2(m,n,p) * alpha_trialfunctions[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                    a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][-l] * G2(m,n,p) * beta_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                    a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * vmom_testfunctions[p][-l] * G2(m,n,p) * beta_trialfunctions[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                
                a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * umom_testfunctions[p][-l] * beta_trialfunctions[p][-l] * ngsolve.dx
                a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * vmom_testfunctions[p][-l] * alpha_trialfunctions[p][-l] * ngsolve.dx
                # factor 0.25 is from the assumed projection coefficients H1 and G0

                a += 0.5 * g * (H+R) * umom_testfunctions[p][-l] * G4(p) * ngsolve.grad(gamma_trialfunctions[-l])[0] / x_scaling * ngsolve.dx # assumes density is depth-independent
                a += 0.5 * g * (H+R) * vmom_testfunctions[p][-l] * G4(p) * ngsolve.grad(gamma_trialfunctions[-l])[1] / y_scaling * ngsolve.dx

                a += (1/x_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * umom_testfunctions[p][-l] * rhox / rho * ngsolve.dx # assumes density is depth-independent
                a += (1/y_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * vmom_testfunctions[p][-l] * rhoy / rho * ngsolve.dx

                if model_options['veddy_viscosity_assumption'] == 'constant':
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][-l] * umom_testfunctions[p][-l] / (H+R) * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][-l] * vmom_testfunctions[p][-l] / (H+R) * ngsolve.dx
                elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][-l] * umom_testfunctions[p][-l] * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][-l] * vmom_testfunctions[p][-l] * ngsolve.dx

                a += -0.5 * vertical_innerproduct(p, p) * f * (H+R) * umom_testfunctions[p][l] * beta_trialfunctions[p][l] * ngsolve.dx
                a += 0.5 * vertical_innerproduct(p, p) * f * (H+R) * vmom_testfunctions[p][l] * alpha_trialfunctions[p][l] * ngsolve.dx
                # factor 0.25 is from the assumed projection coefficients H1 and G0

                a += 0.5 * g * (H+R) * umom_testfunctions[p][l] * G4(p) * ngsolve.grad(gamma_trialfunctions[l])[0] / x_scaling * ngsolve.dx # assumes density is depth-independent
                a += 0.5 * g * (H+R) * vmom_testfunctions[p][l] * G4(p) * ngsolve.grad(gamma_trialfunctions[l])[1] / y_scaling * ngsolve.dx

                a += (1/x_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * umom_testfunctions[p][l] * rhox / rho * ngsolve.dx # assumes density is depth-independent
                a += (1/y_scaling) * 0.5 * np.sqrt(2) * G4(p) * (H+R) * (H+R) * vmom_testfunctions[p][l] * rhoy / rho * ngsolve.dx

                if model_options['veddy_viscosity_assumption'] == 'constant':
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][l] * umom_testfunctions[p][l] / (H+R) * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][l] * vmom_testfunctions[p][l] / (H+R) * ngsolve.dx
                elif model_options['veddy_viscosity_assumption'] == 'depth-scaled&constantprofile': 
                    a += -0.5 * Av * G3(p, p) * alpha_trialfunctions[p][l] * umom_testfunctions[p][l] * ngsolve.dx # assumes that vertical basis consists of eigenfunctions of the vertical mixing operator
                    a += -0.5 * Av * G3(p, p) * beta_trialfunctions[p][l] * vmom_testfunctions[p][l] * ngsolve.dx


def add_linearised_nonlinear_terms(a: ngsolve.BilinearForm, model_options: dict, alpha_trialfunctions, alpha0, beta_trialfunctions, beta0, gamma_trialfunctions,
                       gamma0, umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, imax,
                       constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                       time_basis: TruncationBasis, normalalpha):
    """Adds the (Fréchet/Gâteaux) linearisation of the nonlinear terms (advection and/or surface_in_sigma) to a bilinear form.
    
    Arguments:
    
        - a (ngsolve.BilinearForm):                     bilinear form object on which the weak form is constructed;
        - model_options (dict):                         model options of the hydrodynamics object;
        - alpha_trialfunctions (dict):                  dictionary of trial functions representing the alpha-Fourier/eigenfunction coefficients (obtainable via hydrodynamics._setup_TnT)
        - alpha0 (dict):                                dictionary of gridfunctions containing the value of the alpha coefficients at the current Newton iteration, at which the form is linearised;
        - beta_trialfunctions (dict):                   dictionary of trial functions representing the beta-Fourier/eigenfunction coefficients;
        - beta0 (dict):                                 dictionary of gridfunctions containing the value of the beta coefficients at the current Newton iteration, at which the form is linearised;
        - gamma_trialfunctions (dict):                  dictionary of trial functions representing the gamma-Fourier/eigenfunction coefficients;
        - gamma0 (dict):                                dictionary of gridfunctions containing the value of the gamma coefficients at the current Newton iteration, at which the form is linearised;
        - umom_testfunctions (dict):                    dictionary of test functions for the along-channel momentum equations;
        - vmom_testfunctions (dict):                    dictionary of test functions for the lateral momentum equations;
        - DIC_testfunctions (dict):                     dictionary of test functions for the depth-integrated continuity equation;
        - M (int):                                      number of vertical basis functions;
        - imax (int):                                   number of tidal constituents excluding residual flow;
        - constant_parameters (dict):                   dictionary of constant physical parameters associated with the model;
        - spatial_parameters (dict):                    dictionary of spatial physical parameters (as SpatialParameter objects) associated with the model;
        - vertical_basis (TruncationBasis):             vertical eigenfunction basis;
        - time_basis (TruncationBasis):                 temporal Fourier basis;
        - normalalpha (dict):                           dictionary of coefficient functions containing the riverine normal flow for that boundary condition;
        - x_scaling (float):                            characteristic length scale by which the domain is scaled;
        - y_scaling (float):                            characteristic width scale by which the domain is scaled;
    
    """

    # print("Warning: the model can currently only work if a specific temporal basis is selected. Furthermore, it assumes that the vertical basis consists of eigenfunctions of the vertical mixing operator." + \
    #       "Furthermore, it is assumed that the normal to the riverine (right) and seaward boundary (left) has no lateral component; otherwise, the riverine boundary condition cannot be implemented.")


    # Checking compatibility of model options
    if model_options['surface_in_sigma']:
        print("Adding non-linear effects stemming from the presence of the free surface in the sigma-coordinates is currently not implemented. Did not commence constructing weak form.")
        return
    
    if model_options['density'] != 'depth-independent':
        print("Depth-dependent density profiles are currently not implemented yet. Did not commence constructing weak form.")
        return

    # Defining shorthands of variables

    H = spatial_parameters['H'].cf
    Hx = spatial_parameters["H"].gradient_cf[0]
    Hy = spatial_parameters['H'].gradient_cf[1]
    rho = spatial_parameters['density'].cf
    rhox = spatial_parameters['density'].gradient_cf[0]
    rhoy = spatial_parameters['density'].gradient_cf[1]
    R = spatial_parameters['R'].cf
    Rx = spatial_parameters['R'].gradient_cf[0]
    Ry = spatial_parameters['R'].gradient_cf[1]
    ramp = spatial_parameters['non-linear_ramp'].cf

    f = constant_parameters['f']
    g = constant_parameters['g']
    Av = constant_parameters['Av']
    sigma = constant_parameters['sigma']

    x_scaling = model_options['x_scaling']
    y_scaling = model_options['y_scaling']

    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']
    G3 = vertical_basis.tensor_dict['G3']
    G4 = vertical_basis.tensor_dict['G4']
    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    advection_matrix = model_options['advection_influence_matrix']
    advection_epsilon = model_options['advection_epsilon']

    if model_options['horizontal_diffusion']:
        for p in range(M):
            for i in range(-imax, imax+1):
                for j in range(-imax, imax+1):
                    if H3_iszero(i, j, 0):
                        continue
                    else:
                        if advection_matrix[0, abs(i)] and advection_matrix[0, abs(j)]:
                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * alpha_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[0] / x_scaling *ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha_trialfunctions[m][i] * (
                                (Hx + Rx) * alpha0[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + \
                                (Hy + Ry) * beta0[n][j] / y_scaling + (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta_trialfunctions[m][i] * (
                                (Hx + Rx) * alpha0[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + \
                                (Hy + Ry) * beta0[n][j] / y_scaling + (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                            # Along-channel advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling *ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Lateral advection
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * umom_testfunctions[p][0] * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * G1(m,n,p) * vmom_testfunctions[p][0] * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                            # Vertical advection
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * umom_testfunctions[p][0] * alpha0[m][i] * (
                                (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                                (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                            a += sum([sum([advection_epsilon * ramp * H3(i,j,0) * G2(m,n,p) * vmom_testfunctions[p][0] * beta0[m][i] * (
                                (Hx + Rx) * alpha_trialfunctions[n][j] / x_scaling + (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + \
                                (Hy + Ry) * beta_trialfunctions[n][j] / y_scaling + (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling
                            ) * ngsolve.dx for n in range(M)]) for m in range(M)])
            
            for l in range(1, imax+1):
                for i in range(-imax, imax + 1):
                    for j in range(-imax, imax + 1):
                        if H3_iszero(i,j,-l) and H3_iszero(i,j,l):
                            continue 
                        else:
                            if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:
                                # term -l
                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta0[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + (Hx+Rx) * alpha0[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling + (Hy+Ry) * beta0[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta_trialfunctions[m][i] * (
                                    (H+R) * ngsolve.grad(alpha0[n][j])[0] / x_scaling + (Hx+Rx) * alpha0[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta0[n][j])[1] / y_scaling + (Hy+Ry) * beta0[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * umom_testfunctions[p][-l] * alpha0[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][-l] * beta0[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                                # term +l
                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha0[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,-l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta0[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

                                # Along-channel advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Lateral advection
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(M)]) for m in range(M)])
                                # Vertical advection
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * umom_testfunctions[p][l] * alpha0[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])
                                a += sum([sum([advection_epsilon * ramp * H3(i,j,l) * G2(m,n,p) * vmom_testfunctions[p][l] * beta0[m][i] * (
                                    (H+R) * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling + (Hx+Rx) * alpha_trialfunctions[n][j] / x_scaling + \
                                    (H+R) * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling + (Hy+Ry) * beta_trialfunctions[n][j] / y_scaling
                                ) * ngsolve.dx for n in range(M)]) for m in range(M)])

    else:
        for p in range(0, M): # loop through all vertical components

            # term l = 0

            for i in range(-imax, imax + 1):
                for j in range(-imax, imax + 1):
                    if H3_iszero(i,j,0): # if H^3_{i,j,l} is equal to zero, this iteration can just be skipped;
                        continue
                    else:
                        if advection_matrix[0, abs(i)] and advection_matrix[0, abs(j)]:
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * G2(m,n,p) * (
                                alpha0[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][0])[0] / x_scaling +
                                ngsolve.grad(alpha0[m][i])[0] * alpha_trialfunctions[n][j] * umom_testfunctions[p][0] / x_scaling +
                                alpha0[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][0])[1] / y_scaling +
                                ngsolve.grad(alpha0[m][i])[1] * beta_trialfunctions[n][j] * umom_testfunctions[p][0] / y_scaling
                            ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * G2(m,n,p) * (
                                alpha_trialfunctions[m][i] * alpha0[n][j] * ngsolve.grad(umom_testfunctions[p][0])[0] / x_scaling +
                                ngsolve.grad(alpha_trialfunctions[m][i])[0] * alpha0[n][j] * umom_testfunctions[p][0] / x_scaling +
                                alpha_trialfunctions[m][i] * beta0[n][j] * ngsolve.grad(umom_testfunctions[p][0])[1] / y_scaling +
                                ngsolve.grad(alpha_trialfunctions[m][i])[1] * beta0[n][j] * umom_testfunctions[p][0] / y_scaling
                            ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * G2(m,n,p) * (
                                beta0[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][0])[0] / x_scaling +
                                ngsolve.grad(beta0[m][i])[0] * alpha_trialfunctions[n][j] * vmom_testfunctions[p][0] / x_scaling +
                                beta0[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][0])[1] / y_scaling +
                                ngsolve.grad(beta0[m][i])[1] * beta_trialfunctions[n][j] * vmom_testfunctions[p][0] / y_scaling
                            ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * G2(m,n,p) * (
                                beta_trialfunctions[m][i] * alpha0[n][j] * ngsolve.grad(vmom_testfunctions[p][0])[0] / x_scaling +
                                ngsolve.grad(beta_trialfunctions[m][i])[0] * alpha0[n][j] * vmom_testfunctions[p][0] / x_scaling +
                                beta_trialfunctions[m][i] * beta0[n][j] * ngsolve.grad(vmom_testfunctions[p][0])[1] / y_scaling +
                                ngsolve.grad(beta_trialfunctions[m][i])[1] * beta0[n][j] * vmom_testfunctions[p][0] / y_scaling
                            ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G2(m,n,p) * alpha_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G2(m,n,p) * alpha0[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                            a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G2(m,n,p) * beta_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G2(m,n,p) * beta0[m][i] * alpha_trialfunctions[n][j] for m in range(0, M)]) / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                            
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * umom_testfunctions[p][0] * G2(m,n,p) * alpha_trialfunctions[m][i] * alpha0[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                            a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,0) * vmom_testfunctions[p][0] * G2(m,n,p) * beta_trialfunctions[m][i] * alpha0[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]

            # terms l!=0
            for l in range(1, imax + 1):
                
                for i in range(-imax, imax + 1):
                    for j in range(-imax, imax + 1):

                        if H3_iszero(i,j,-l): # if H^3_{i,j,l} is equal to zero, this iteration can just be skipped;
                            continue
                        else:
                            if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * G2(m,n,p) * (
                                    alpha0[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][-l])[0] / x_scaling +
                                    ngsolve.grad(alpha0[m][i])[0] * alpha_trialfunctions[n][j] * umom_testfunctions[p][-l] / x_scaling +
                                    alpha0[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][-l])[1] / y_scaling +
                                    ngsolve.grad(alpha0[m][i])[1] * beta_trialfunctions[n][j] * umom_testfunctions[p][-l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * G2(m,n,p) * (
                                    alpha_trialfunctions[m][i] * alpha0[n][j] * ngsolve.grad(umom_testfunctions[p][-l])[0] / x_scaling +
                                    ngsolve.grad(alpha_trialfunctions[m][i])[0] * alpha0[n][j] * umom_testfunctions[p][-l] / x_scaling +
                                    alpha_trialfunctions[m][i] * beta0[n][j] * ngsolve.grad(umom_testfunctions[p][-l])[1] / y_scaling +
                                    ngsolve.grad(alpha_trialfunctions[m][i])[1] * beta0[n][j] * umom_testfunctions[p][-l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * G2(m,n,p) * (
                                    beta0[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][-l])[0] / x_scaling +
                                    ngsolve.grad(beta0[m][i])[0] * alpha_trialfunctions[n][j] * vmom_testfunctions[p][-l] / x_scaling +
                                    beta0[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][-l])[1] / y_scaling +
                                    ngsolve.grad(beta0[m][i])[1] * beta_trialfunctions[n][j] * vmom_testfunctions[p][-l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * G2(m,n,p) * (
                                    beta_trialfunctions[m][i] * alpha0[n][j] * ngsolve.grad(vmom_testfunctions[p][-l])[0] / x_scaling +
                                    ngsolve.grad(beta_trialfunctions[m][i])[0] * alpha0[n][j] * vmom_testfunctions[p][-l] / x_scaling +
                                    beta_trialfunctions[m][i] * beta0[n][j] * ngsolve.grad(vmom_testfunctions[p][-l])[1] / y_scaling +
                                    ngsolve.grad(beta_trialfunctions[m][i])[1] * beta0[n][j] * vmom_testfunctions[p][-l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G2(m,n,p) * alpha_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G2(m,n,p) * alpha0[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G2(m,n,p) * beta_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G2(m,n,p) * beta0[m][i] * alpha_trialfunctions[n][j] for m in range(0, M)]) / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * umom_testfunctions[p][-l] * G2(m,n,p) * alpha_trialfunctions[m][i] * alpha0[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,-l) * vmom_testfunctions[p][-l] * G2(m,n,p) * beta_trialfunctions[m][i] * alpha0[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]

                        if H3_iszero(i,j,l): # if H^3_{i,j,l} is equal to zero, this iteration can just be skipped;
                            continue
                        else:
                            if advection_matrix[l, abs(i)] and advection_matrix[l, abs(j)]:
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(alpha_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * G2(m,n,p) * (
                                    alpha0[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][l])[0] / x_scaling +
                                    ngsolve.grad(alpha0[m][i])[0] * alpha_trialfunctions[n][j] * umom_testfunctions[p][l] / x_scaling +
                                    alpha0[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(umom_testfunctions[p][l])[1] / y_scaling +
                                    ngsolve.grad(alpha0[m][i])[1] * beta_trialfunctions[n][j] * umom_testfunctions[p][l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(alpha0[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * G2(m,n,p) * (
                                    alpha_trialfunctions[m][i] * alpha0[n][j] * ngsolve.grad(umom_testfunctions[p][l])[0] / x_scaling +
                                    ngsolve.grad(alpha_trialfunctions[m][i])[0] * alpha0[n][j] * umom_testfunctions[p][l] / x_scaling +
                                    alpha_trialfunctions[m][i] * beta0[n][j] * ngsolve.grad(umom_testfunctions[p][l])[1] / y_scaling +
                                    ngsolve.grad(alpha_trialfunctions[m][i])[1] * beta0[n][j] * umom_testfunctions[p][l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta0[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * G2(m,n,p) * (
                                    beta0[m][i] * alpha_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][l])[0] / x_scaling +
                                    ngsolve.grad(beta0[m][i])[0] * alpha_trialfunctions[n][j] * vmom_testfunctions[p][l] / x_scaling +
                                    beta0[m][i] * beta_trialfunctions[n][j] * ngsolve.grad(vmom_testfunctions[p][l])[1] / y_scaling +
                                    ngsolve.grad(beta0[m][i])[1] * beta_trialfunctions[n][j] * vmom_testfunctions[p][l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * alpha_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[0] / x_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G1(m,n,p) * beta_trialfunctions[m][i] * ngsolve.grad(beta_trialfunctions[n][j])[1] / y_scaling * ngsolve.dx for n in range(0, M)]) for m in range(0, M)])
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * G2(m,n,p) * (
                                    beta_trialfunctions[m][i] * alpha0[n][j] * ngsolve.grad(vmom_testfunctions[p][l])[0] / x_scaling +
                                    ngsolve.grad(beta_trialfunctions[m][i])[0] * alpha0[n][j] * vmom_testfunctions[p][l] / x_scaling +
                                    beta_trialfunctions[m][i] * beta0[n][j] * ngsolve.grad(vmom_testfunctions[p][l])[1] / y_scaling +
                                    ngsolve.grad(beta_trialfunctions[m][i])[1] * beta0[n][j] * vmom_testfunctions[p][l] / y_scaling
                                ) * ngsolve.dx for m in range(0, M)]) for n in range(0, M)])

                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G2(m,n,p) * alpha_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G2(m,n,p) * alpha0[m][i] * alpha_trialfunctions[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                a += sum([sum([advection_epsilon * ramp * (H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G2(m,n,p) * beta_trialfunctions[m][i] * normalalpha[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[RIVER]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at river boundary to be [1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G2(m,n,p) * beta0[m][i] * alpha_trialfunctions[n][j] for m in range(0, M)]) / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * umom_testfunctions[p][l] * G2(m,n,p) * alpha_trialfunctions[m][i] * alpha0[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
                                a += sum([sum([advection_epsilon * ramp * -(H+R) * H3(i,j,l) * vmom_testfunctions[p][l] * G2(m,n,p) * beta_trialfunctions[m][i] * alpha0[n][j] / x_scaling * ngsolve.ds(BOUNDARY_DICT[SEA]) for m in range(0, M)]) for n in range(0, M)]) # assumes outward normal at sea boundary to be [-1 0]
        


    

     