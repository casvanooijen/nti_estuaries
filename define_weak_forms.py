import ngsolve
import ngsolve.solvers
import numpy as np
from TruncationBasis import TruncationBasis
import matplotlib.pyplot as plt
import geometry.geometries
from geometry.create_geometry import RIVER, SEA, BOUNDARY_DICT
from netgen.geom2d import SplineGeometry
import mesh_functions
from assumptions import ModelOptions
from minusonepower import minusonepower
        
## Public functions that call other functions depending on the assumptions
        
def add_bilinear_part(a: ngsolve.BilinearForm, model_options: ModelOptions, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions, 
                      umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, qmax, constant_parameters: dict, spatial_parameters: dict, 
                      vertical_basis: TruncationBasis, normalalpha, forcing=True):
    if model_options.bed_bc == 'no_slip' and model_options.leading_order_surface and model_options.veddy_viscosity_assumption == 'constant' and model_options.density == 'depth-independent':
        return _add_bilinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                                umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, qmax,
                                                constant_parameters, spatial_parameters, vertical_basis, normalalpha, forcing)
    else:
        print("Bilinear part of weak form was not set up")


def add_nonlinear_part(a: ngsolve.BilinearForm, model_options: ModelOptions, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                       umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, qmax,
                       constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                       time_basis: TruncationBasis, normalalpha, n):
    if model_options.bed_bc == 'no_slip' and model_options.leading_order_surface and model_options.veddy_viscosity_assumption == 'constant' and model_options.density == 'depth-independent':
        return _add_nonlinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                                umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, qmax,
                                                constant_parameters, spatial_parameters, vertical_basis, time_basis, normalalpha, model_options.advection_epsilon, n)
    else:
        print("Nonlinear part of weak form was not set up")



def add_linearised_nonlinear_part(a: ngsolve.BilinearForm, model_options: ModelOptions, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                  umom_testfunctions, vmom_testfunctions, DIC_testfunctions, alpha0, beta0, gamma0, M, qmax,
                                  constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                                  time_basis: TruncationBasis, normalalpha, advection_weighting_parameter, n):
    if model_options.bed_bc == 'no_slip' and model_options.leading_order_surface and model_options.veddy_viscosity_assumption == 'constant' and model_options.density == 'depth-independent':
        return _add_linearised_nonlinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                                umom_testfunctions, vmom_testfunctions, DIC_testfunctions, alpha0, beta0, gamma0, M, qmax,
                                                constant_parameters, spatial_parameters, vertical_basis, time_basis, normalalpha, model_options.advection_epsilon, n)
## Private functions that add the weak forms to the ngsolve.BilinearForm object

def _add_bilinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                     umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, qmax,
                                     constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, normalalpha, forcing=True):
    
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
              normalalpha[m][0] for m in range(M + 1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
        for r in range(1, qmax + 1):
            a += (0.5 / sig * DIC_testfunctions[-r] * H * sum([G4(m) * \
                 normalalpha[m][-r] for m in range(M + 1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
            a += (0.5 / sig * DIC_testfunctions[r] * H * sum([G4(m) * \
                 normalalpha[m][r] for m in range(M + 1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
            
    # r = 0 term
    a += (-0.5/sig * H * sum([G4(m) * (alpha_trialfunctions[m][0] * ngsolve.grad(DIC_testfunctions[0])[0] + 
                                       beta_trialfunctions[m][0] * ngsolve.grad(DIC_testfunctions[0])[1]) for m in range(M + 1)])) * ngsolve.dx

    # r != 0-terms
    for r in range(1, qmax + 1):
        a += (ngsolve.pi*r*gamma_trialfunctions[r]*DIC_testfunctions[-r] - 0.5/sig*H*sum([G4(m) * (
              alpha_trialfunctions[m][-r] * ngsolve.grad(DIC_testfunctions[-r])[0] + 
              beta_trialfunctions[m][-r] * ngsolve.grad(DIC_testfunctions[-r])[1]
        ) for m in range(M + 1)])) * ngsolve.dx
        a += (ngsolve.pi*-r*gamma_trialfunctions[-r]*DIC_testfunctions[r] - 0.5/sig*H*sum([G4(m) * (
              alpha_trialfunctions[m][r] * ngsolve.grad(DIC_testfunctions[r])[0] + 
              beta_trialfunctions[m][r] * ngsolve.grad(DIC_testfunctions[r])[1]
        ) for m in range(M + 1)])) * ngsolve.dx

    
    # MOMENTUM EQUATIONS
        
    for k in range(M + 1):
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
        for r in range(1, qmax + 1):
            # U-momentum
            a += ((0.5*ngsolve.pi*r*H*alpha_trialfunctions[k][r] *umom_testfunctions[k][-r]- 
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
                                      umom_testfunctions, vmom_testfunctions, DIC_testfunctions, M, qmax,
                                      constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                                      time_basis: TruncationBasis, normalalpha, advection_weighting_parameter, normal):
    
    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']

    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    H = spatial_parameters['H'].cf
    
    for k in range(M+1):
        # component r = 0
        # U-momentum
        for p in range(-qmax, qmax+1):
            for q in range(-qmax, qmax+1):
                if H3_iszero(p, q, 0):
                    continue
                else:
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
                    ) for m in range(M + 1)]) for n in range(M + 1)])
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
                    ) for m in range(M + 1)]) for n in range(M + 1)])

        for r in range(1, qmax + 1):
            # add nonlinear part of a
            #component -r
            for p in range(-qmax, qmax+1):
                for q in range(-qmax, qmax+1):
                    if not H3_iszero(p, q, -r):
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
                        for m in range(M + 1)]) for n in range(M + 1)])

                        
                        
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
                        for m in range(M + 1)]) for n in range(M + 1)])

                        
                    
                    if not H3_iszero(p, q, r):
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
                        ) for m in range(M + 1)]) for n in range(M + 1)])

                        
                        
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
                        ) for m in range(M + 1)]) for n in range(M + 1)])
                        

            
def _add_linearised_nonlinear_part_NS_RL_EVC_DDI(a, alpha_trialfunctions, beta_trialfunctions, gamma_trialfunctions,
                                      umom_testfunctions, vmom_testfunctions, DIC_testfunctions, alpha0, beta0, gamma0, M, qmax,
                                      constant_parameters, spatial_parameters, vertical_basis: TruncationBasis, 
                                      time_basis: TruncationBasis, normalalpha, advection_weighting_parameter, normal):
    
    G1 = vertical_basis.tensor_dict['G1']
    G2 = vertical_basis.tensor_dict['G2']

    H3 = time_basis.tensor_dict['H3']
    H3_iszero = time_basis.tensor_dict['H3_iszero']

    H = spatial_parameters['H'].cf
    
    for k in range(M + 1):
        for p in range(-qmax, qmax + 1):
            for q in range(-qmax, qmax + 1):
                if H3_iszero(p, q, 0):
                    continue
                else:
                    # interior domain integration for u-momentum
                    a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                        -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + 
                                            beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1]) - \
                        H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + \
                                                                beta0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1])
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.dx
                    a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
                        -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + 
                                            beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1]) - \
                        H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[0] + \
                                                                beta0[n][p] * ngsolve.grad(umom_testfunctions[k][0])[1]) - 
                        H * umom_testfunctions[k][0] * (ngsolve.grad(alpha0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                            ngsolve.grad(alpha0[m][q])[1] * beta_trialfunctions[n][p]) -
                        H * umom_testfunctions[k][0] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                            ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta0[n][p])
                    ) for n in range(M+1)])for m in range(M+1)]))*ngsolve.dx
                    # integration over seaward boundary for u-momentum
                    a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                        H * alpha0[m][q] * umom_testfunctions[k][0] * (
                            alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                        ) + 
                        H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * (
                            alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                        )
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                    a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                        H * alpha0[m][q] * umom_testfunctions[k][0] * (
                            alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                        ) + 
                        H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * (
                            alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                        )
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                    # integration over riverine boundary for u-momentum
                    a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                        H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * normalalpha[n][p]
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                    a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                        H * alpha_trialfunctions[m][q] * umom_testfunctions[k][0] * normalalpha[n][p]
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                    # interior domain integration for v-momentum
                    a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                        -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + 
                                            beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1]) - \
                        H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + \
                                                                beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1])
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.dx
                    a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k))*(
                        -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + 
                                            beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1]) - \
                        H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[0] + \
                                                                beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][0])[1]) - 
                        H * vmom_testfunctions[k][0] * (ngsolve.grad(beta0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                            ngsolve.grad(beta0[m][q])[1] * beta_trialfunctions[n][p]) -
                        H * vmom_testfunctions[k][0] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                            ngsolve.grad(beta_trialfunctions[m][q])[1] * beta0[n][p])
                    ) for n in range(M+1)])for m in range(M+1)]))*ngsolve.dx
                    # integration over seaward boundary for v-momentum
                    a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                        H * beta0[m][q] * vmom_testfunctions[k][0] * (
                            alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                        ) + 
                        H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * (
                            alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                        )
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                    a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                        H * beta0[m][q] * vmom_testfunctions[k][0] * (
                            alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                        ) + 
                        H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * (
                            alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                        )
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                    # integration over riverine boundary for v-momentum
                    a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,0)*(
                        H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * normalalpha[n][p]
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                    a += (advection_weighting_parameter * sum([sum([-H3(p,q,0)*(G1(m,n,k)+G2(m,n,k)) * (
                        H * beta_trialfunctions[m][q] * vmom_testfunctions[k][0] * normalalpha[n][p]
                    ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                            
                # terms r != 0
        for r in range(1, qmax + 1):
            for p in range(-qmax, qmax + 1):
                for q in range(-qmax, qmax + 1):
                    if H3_iszero(p, q, r):
                        continue
                    else:
                        # terms -r
                        # interior domain integration for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                            -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1]) - \
                            H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1])
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.dx
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
                            -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1]) - \
                            H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(umom_testfunctions[k][-r])[1]) - 
                            H * umom_testfunctions[k][-r] * (ngsolve.grad(alpha0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                ngsolve.grad(alpha0[m][q])[1] * beta_trialfunctions[n][p]) -
                            H * umom_testfunctions[k][-r] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta0[n][p])
                        ) for n in range(M+1)])for m in range(M+1)]))*ngsolve.dx
                        # integration over seaward boundary for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                            H * alpha0[m][q] * umom_testfunctions[k][-r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * alpha0[m][q] * umom_testfunctions[k][-r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        # integration over riverine boundary for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][-r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        # interior domain integration for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                            -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1]) - \
                            H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1])
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.dx
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k))*(
                            -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1]) - \
                            H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][-r])[1]) - 
                            H * vmom_testfunctions[k][-r] * (ngsolve.grad(beta0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                ngsolve.grad(beta0[m][q])[1] * beta_trialfunctions[n][p]) -
                            H * vmom_testfunctions[k][-r] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                ngsolve.grad(beta_trialfunctions[m][q])[1] * beta0[n][p])
                        ) for n in range(M+1)])for m in range(M+1)]))*ngsolve.dx
                        # integration over seaward boundary for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                            H * beta0[m][q] * vmom_testfunctions[k][-r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * beta0[m][q] * vmom_testfunctions[k][-r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        # integration over riverine boundary for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,-r)*(
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,-r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][-r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])

                        # terms +r
                        # interior domain integration for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                            -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1]) - \
                            H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1])
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.dx
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
                            -H * alpha0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1]) - \
                            H * alpha_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(umom_testfunctions[k][r])[1]) - 
                            H * umom_testfunctions[k][r] * (ngsolve.grad(alpha0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                ngsolve.grad(alpha0[m][q])[1] * beta_trialfunctions[n][p]) -
                            H * umom_testfunctions[k][r] * (ngsolve.grad(alpha_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                ngsolve.grad(alpha_trialfunctions[m][q])[1] * beta0[n][p])
                        ) for n in range(M+1)])for m in range(M+1)]))*ngsolve.dx
                        # integration over seaward boundary for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                            H * alpha0[m][q] * umom_testfunctions[k][r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * alpha0[m][q] * umom_testfunctions[k][r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        # integration over riverine boundary for u-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * alpha_trialfunctions[m][q] * umom_testfunctions[k][r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        # interior domain integration for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                            -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1]) - \
                            H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1])
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.dx
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k))*(
                            -H * beta0[m][q] * (alpha_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + 
                                                beta_trialfunctions[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1]) - \
                            H * beta_trialfunctions[m][q] * (alpha0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[0] + \
                                                                    beta0[n][p] * ngsolve.grad(vmom_testfunctions[k][r])[1]) - 
                            H * vmom_testfunctions[k][r] * (ngsolve.grad(beta0[m][q])[0] * alpha_trialfunctions[n][p] + \
                                                                ngsolve.grad(beta0[m][q])[1] * beta_trialfunctions[n][p]) -
                            H * vmom_testfunctions[k][r] * (ngsolve.grad(beta_trialfunctions[m][q])[0] * alpha0[n][p] + 
                                                                ngsolve.grad(beta_trialfunctions[m][q])[1] * beta0[n][p])
                        ) for n in range(M+1)])for m in range(M+1)]))*ngsolve.dx
                        # integration over seaward boundary for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                            H * beta0[m][q] * vmom_testfunctions[k][r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * beta0[m][q] * vmom_testfunctions[k][r] * (
                                alpha_trialfunctions[n][p] * normal[0] + beta_trialfunctions[n][p] * normal[1]
                            ) + 
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * (
                                alpha0[n][p] * normal[0] + beta0[n][p] * normal[1]
                            )
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[SEA])
                        # integration over riverine boundary for v-momentum
                        a += (advection_weighting_parameter * sum([sum([G1(m,n,k)*H3(p,q,r)*(
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])
                        a += (advection_weighting_parameter * sum([sum([-H3(p,q,r)*(G1(m,n,k)+G2(m,n,k)) * (
                            H * beta_trialfunctions[m][q] * vmom_testfunctions[k][r] * normalalpha[n][p]
                        ) for n in range(M+1)]) for m in range(M+1)])) * ngsolve.ds(BOUNDARY_DICT[RIVER])





     