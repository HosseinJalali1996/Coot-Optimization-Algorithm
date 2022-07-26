from timer import Timer
from select_functions import Function_Name
from coot import COOT
from structure_class import struct
import matplotlib.pyplot as plt

t = Timer()
t.start()

parameters = struct()

parameters.N = 30         # Number of search agents
parameters.Max_iter = 500 # Maximum number of iterations
parameters.LP = 0.1       # Leader Percentage


problem = Function_Name('F1') # You can choose the functions from 1 to 13.

Convergence_curve, best_Position, best_cost, Iter = COOT(parameters,problem)

t.stop()
print()
print(best_Position,end='\n\n')

plt.plot(Iter, Convergence_curve)
plt.xlabel('Iteration')
plt.ylabel('Convergence_curve')
plt.show()
