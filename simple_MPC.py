from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from simulation_code import simulate
import numpy as np


#Define enviorment
T = 0.2 #[s]
N = 1  #prediction horizon
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

def shift_timestep(sampling_time, t0,x0,u,f):
    f_value = f(x0, u[:,0])
    next_state = ca.DM.full(x0 + (sampling_time * f_value))

    t0 = t0 + sampling_time
    u0 = ca.horzcat(
            u[:, 1:],
            reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0

def DM2Arr(dm):
    return np.array(dm.full())



for p in range(0,n_robots): #define states, inputs and dynamics for each robot
    x.append(SX.sym('x'+str(p)))
    y.append(SX.sym('y'+str(p)))
    theta.append(SX.sym('theta'+str(p)))
    states.extend([x[p],y[p],theta[p]])
    print(states)
    v.append(SX.sym('v'+str(p)))
    omega.append(SX.sym('omega'+str(p)))
    inputs.extend([v[p],omega[p]])

    rhs.extend([v[p]*cos(theta[p]),v[p]*sin(theta[p]),omega[p]])  #system r.h.s (right hand side of system dynamics)


print(x)
print(y)

#x = SX.sym('x',5)
print(x)


n_states = len(states)
n_inputs = len(inputs) #Get number of states and inputs

f = Function('f',[states, inputs],[rhs]) #nonlinear mapping function f(x,u)
U = SX.sym('U',n_inputs,N) #Decision variables (inputs)
P = SX.sym('P',n_states + n_states) #parameters (which include at the initial state of the robot and the reference state)

X = SX.sym('X',n_states,(N+1)) #A vector that represents the states over the optimization problem.

cost_fn = 0 #cost function
g = []  #Constraints vector ~~check

Q = np.zeros((n_states,n_states)) #weighting matrices (states)
R = np.zeros((n_inputs,n_inputs)) #weighting matrices (inputs)

for nj in range(0,n_robots): #%Populate Q and R matrices for all robots (for 3 states and 2 inputs per robot)
    Q[3*(nj),3*(nj)] = 1
    Q[1+3*(nj),1+3*(nj)] = 5
    Q[2+3*(nj),2+3*(nj)] = 0.1 
    R[2*(nj),2*(nj)] = 0.5
    R[1+2*(nj),1+2*(nj)] = 0.05 


st  = X[:,0] #initial state
g.append(st-P[0:n_robots*3]) #initial condition constraints for initial state (3 is the amount of states per robot)

for k in range(0,N):
    st = X[:,k]
    inp = U[:,k]; #st=state; inp=input
    st_error = (st-P[n_robots*3:n_robots*3+n_robots*3])
    obj = obj+st_error.T @ Q @ st_error + inp.T @ R @ inp #calculate obj
    st_next = X[:,k+1]
    f_value = f(st,inp) #~~check
    st_next_euler = st+ (T*f_value)
    g.append(st_next-st_next_euler) #compute constraints for equality state constraints

# runge kutta
for k in range(N):
    st = X[:, k]
    inp = U[:, k] #st = state; inp = input
    cost_fn = cost_fn \
        + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = vertcat(g, st_next - st_next_RK4)

OPT_variables = vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,






