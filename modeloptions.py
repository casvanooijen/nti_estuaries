# class Assumptions(object):

#     """Object containing the assumptions """

#     def __init__(self, bed_bc:str, rigid_lid:bool, eddy_viscosity_assumption:str, density:str):

#         """Arguments: ('...'  means that there will possibly be future options added)
        
#         - bed_bc:                       ('no_slip' or ...)
#         - rigid_lid:                    flag to indicate whether a rigid lid assumption is used;
#         - eddy_viscosity_assumption:    ('constant' or ...);
#         - density:                      ('depth-independent' or ...);
        
#         """
        
#         self.bed_bc = bed_bc
#         self.rigid_lid = rigid_lid
#         self.eddy_viscosity_assumption = eddy_viscosity_assumption
#         self.density = density


class ModelOptions(object):

    """Object containing user-provided options/assumptions for the model equations, e.g. whether non-linear advection should be included or not, or whether different modules of the 
    code (hydrodynamics & sediment) should be coupled or not."""

    def __init__(self, bed_bc:str = 'no-slip', leading_order_surface:bool = True, veddy_viscosity_assumption:str = 'constant', density:str = 'depth-independent',
                 advection_epsilon:float = 1):

        """Arguments: ('...'  means that there will possibly be future options added)
        
        - bed_bc:                       indicates what type of boundary condition is used at the river bed ('no_slip' or ...);
        - leading_order_surface (bool): flag to indicate whether non-linear effects stemming from the varying water surface should be included;
        - veddy_viscosity_assumption:   structure of the vertical eddy viscosity parameter ('constant' or ...);
        - density:                      indicates what type of water density field is used ('depth-independent' or ...);
        - advection_epsilon (float):    scalar by which the advective terms in the momentum equations are multiplied; if set to zero, advective terms are skipped;     
                                        if set to one, advective terms are fully included;       
        
        """

        self.bed_bc = bed_bc
        self.leading_order_surface = leading_order_surface
        self.veddy_viscosity_assumption = veddy_viscosity_assumption
        self.density = density
        self.advection_epsilon = advection_epsilon