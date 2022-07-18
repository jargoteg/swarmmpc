from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi, arctan2
import matplotlib.pyplot as plt
from simulation_code import simulate

#Single robot MPC using relative distance and orientation instead of global info.

# setting matrix_weights' variables
Q_x = 100
Q_y = 100
Q_theta = 0
R1 = 1
R2 = 0.01

step_horizon = 0.1  # time between steps in seconds
N = 10              # number of look ahead steps
rob_diam = 0.3      # diameter of the robot
sim_time = 200      # simulation time

# specs
x_init = 0
y_init = 0
theta_init = 0

x_target = 0
y_target = 10
theta_target = pi

v_max = 0.6
v_min = -v_max
omega_max = pi/4 
omega_min = -omega_max

#Define state constraints
x_max = 10      #Size of the arena
x_min = -10   
y_max = 10
y_min = -10


def shift_timestep(step_horizon, t0, state_init, u, f):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0


def DM2Arr(dm):
    return np.array(dm.full())


# state symbolic variables
x = ca.SX.sym('x')
y = ca.SX.sym('y')
theta = ca.SX.sym('theta')
states = ca.vertcat(
    x,
    y,
    theta
)
n_states = states.numel()

# control symbolic variables
v = ca.SX.sym('v')
omega = ca.SX.sym('omega')

controls = ca.vertcat(
    v,
    omega,
)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# state weights matrix (Q_X, Q_Y, Q_THETA)
Q = ca.diagcat(Q_x, Q_y, Q_theta)

# controls weights matrix
# controls weights matrix
R = ca.diagcat(R1, R2)

RHS = ca.vertcat(
    v*cos(theta),
    v*sin(theta),
    omega
)

# maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
f = ca.Function('f', [states, controls], [RHS])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation


# runge kutta
for k in range(N):
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[n_states:]).T @ Q @ (st - P[n_states:]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
    k1 = f(st, con)
    k2 = f(st + step_horizon/2*k1, con)
    k3 = f(st + step_horizon/2*k2, con)
    k4 = f(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    g = ca.vertcat(g, st_next - st_next_RK4)

m = [-omega_max/v_max, omega_max/-v_min, -omega_min/v_max, omega_min/-v_min]

for p in range(0,4): #Diff drive constraint in the input
    for k in range(0,N):
        con = U[:,k]; #con=control
        g = ca.vertcat(g, con[1]-m[p]*con[0])

OPT_variables = ca.vertcat(
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
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = x_min     # X lower bound
lbx[1: n_states*(N+1): n_states] = y_min     # Y lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # theta lower bound

ubx[0: n_states*(N+1): n_states] = x_max      # X upper bound
ubx[1: n_states*(N+1): n_states] = y_max      # Y upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # theta upper bound

lbx[n_states*(N+1)::n_controls] = v_min       # v lower bound for all V
lbx[n_states*(N+1)+1::n_controls] = omega_min       # v lower bound for all V

ubx[n_states*(N+1)::n_controls] = v_max       # v lower bound for all V
ubx[n_states*(N+1)+1::n_controls] = omega_max       # v lower bound for all V


lbg = ca.DM.zeros((n_states*(N+1) + 4*(N), 1))  # constraints lower bound
ubg = ca.DM.zeros((n_states*(N+1) + 4*(N), 1))  # constraints upper bound

lbg[n_states*(N+1):n_states*(N+1)+2*N] = -ca.inf
lbg[n_states*(N+1)+2*N:] = omega_min

ubg[n_states*(N+1):n_states*(N+1)+2*N] = omega_max
ubg[n_states*(N+1)+2*N:] = ca.inf

args = {
    'lbg': lbg,  # constraints lower bound
    'ubg': ubg,  # constraints upper bound
    'lbx': lbx,
    'ubx': ubx
}

t0 = 0
state_init = ca.DM([x_init, y_init, theta_init])        # initial state
state_init0 = ca.DM([0,0,0])        # initial state
state_real_target = ca.DM([x_target, y_target, theta_target])  # target state
state_target = state_real_target - state_init  # target state

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1)         # initial state full


mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])


###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    #sensor measurements
    rel_distance = ca.norm_2(state_init[0:2] - state_target[0:2]) #distance to target
    #heading to target
    orientation = arctan2(state_target[0]-state_init[0],state_target[1]-state_init[1])
    print(rel_distance)
    print(state_init)
    while (ca.norm_2(state_init - state_real_target) > 1e-1) and (mpc_iter * step_horizon < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init0,    # current state
            state_target   # target state
        )
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )

        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)

        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))

        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init0, u, f)
        state_target = state_real_target - state_init  # target state
        print(X0)
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        # xx ...
        t2 = time()
        #print(mpc_iter)
        #print(t2-t1)
        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(cat_states, cat_controls, times, step_horizon, N,
             np.array([x_init, y_init, theta_init, x_target, y_target, theta_target]), save=False)
