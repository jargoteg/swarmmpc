from casadi import *
import numpy as np


#Define enviorment
T = 0.2 #[s]
N = 10  #prediction horizon
rob_diam = 0.3 #[m]
n_robots = 3 #number of robots
obss = np.array([[5,5]]) #Position of obstacles
n_obs = len(obss[:][0]) #number of obstacles
obs_diam = 0.3 #[m]

#Define state constraints
x_max = 10
x_min = -10    #Size of the arena
y_max = 1
y_min = -1
#limits = [x_min x_max ; y_min y_max]

v_max = 0.6
v_min = -v_max
omega_max = pi/4
omega_min = -omega_max

x = []
y = []
theta = []
v = []
omega = []
states = []
inputs = []
rhs = []



for p in range(0,n_robots): #define states, inputs and dynamics for each robot
    x.append(SX.sym('x'+str(p)))
    y.append(SX.sym('y'+str(p)))
    theta.append(SX.sym('theta'+str(p)))
    states.extend([x[p],y[p],theta[p]])

    v.append(SX.sym('v'+str(p)))
    omega.append(SX.sym('omega'+str(p)))
    inputs.extend([v[p],omega[p]])

    rhs.extend([v[p]*cos(theta[p]),v[p]*sin(theta[p]),omega[p]])  #system r.h.s (right hand side of system dynamics)

print(states)
print(inputs)
print(rhs)

n_states = len(states)
n_inputs = len(inputs) #Get number of states and inputs

f = Function('f',states+inputs,rhs) #nonlinear mapping function f(x,u)
U = SX.sym('U',n_inputs,N) #Decision variables (inputs)
P = SX.sym('P',n_states + n_states) #parameters (which include at the initial state of the robot and the reference state)

X = SX.sym('X',n_states,(N+1)) #A vector that represents the states over the optimization problem.

obj = 0; #Objective function
g = [];  #Constraints vector

Q = np.zeros((n_states,n_states)) #weighting matrices (states)
R = np.zeros((n_inputs,n_inputs)) #weighting matrices (inputs)


