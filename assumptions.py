class Assumptions(object):

    def __init__(self, bed_bc:str, rigid_lid:bool, eddy_viscosity_assumption:str, density:str):

        """Arguments:
        
        - bed_bc:                       ('no_slip' or ...)
        - rigid_lid:                    flag to indicate whether a rigid lid assumption is used;
        - eddy_viscosity_assumption:    ('constant' or ...);
        - density:                      ('depth-independent' or ...);
        
        """
        
        self.bed_bc = bed_bc
        self.rigid_lid = rigid_lid
        self.eddy_viscosity_assumption = eddy_viscosity_assumption
        self.density = density