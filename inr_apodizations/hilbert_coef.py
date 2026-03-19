import numpy as np
from scipy import signal

# ------------------- COEFCICIENTES FILTRO HILBERT ----------------------------------------------------------------
# coeficientes de filtro Hilbert que me pasó J. Camacho
# WARNING: este filtro está hardcoded
# tiene 63 coeficientes, por lo cual se debe usar taps=62
# TODO: usar distinto nro de taps para el filtro pasabando y para Hilbert
# modo de no obligar a que el pasabanda tenga necesariamente taps=62
q = [-28, 0, -18, 0, -51, 0, -104, 0, -179, 0, -280, 0, -409, 0, -575, 0, -786, 0, -1058, 0, -1417, 0, -1914, 0, -2659,
     0, -3939, 0, -6812, 0, -20813, 0, 20813, 0, 6812, 0, 3939, 0, 2659, 0, 1914, 0, 1417, 0, 1058, 0, 786, 0, 575, 0,
     409, 0, 280, 0, 179, 0, 104, 0, 51, 0, 18, 0, 28]
# normalizar filtro para que tenga gananacia=1
temp = signal.freqz(q)
coef = q / np.abs(temp[1]).max()
