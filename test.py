import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from netgen.geom2d import unit_square
import ngsolve

mesh = ngsolve.Mesh(unit_square.GenerateMesh(maxh=0.2))





