from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
from simulation_code import simulate

#Main MPC class where all variables are defined. Intended to work for multi-robot case
#Robot limits in velocity
MAX_SPEED = 100
MAX_ROTATION = pi/4

ARENA_SIZE_X = 100
ARENA_SIZE_Y = 100
class MPC:

    #MPC weights
    Q_x = 100
    Q_y = 100
    Q_theta = 2
    R1 = 1
    R2 = 0.01
    
    N = 40              # number of look ahead steps
    rob_diam = 0.3      # diameter of the robot
    sim_time = 200      # simulation time
    step_horizon = 0.1  # time between steps in seconds
    def __init__(self,posx,posy,postheta,target_x,target_y,target_theta):
        self.x_init = posx
        self.y_init = posy
        self.theta_init = postheta

        self.x_target = target_x
        self.y_target = target_y
        self.theta_target = target_theta

        #Robot specific parameters
        self.v_max = MAX_SPEED
        self.v_min = -self.v_max
        self.omega_max = MAX_ROTATION
        self.omega_min = -self.omega_max

        #Size of the arena
        self.x_max = ARENA_SIZE_X    
        self.x_min = -ARENA_SIZE_X   
        self.y_max = ARENA_SIZE_Y
        self.y_min = -ARENA_SIZE_Y

        # state symbolic variables
        self.x = ca.SX.sym('x')
        self.y = ca.SX.sym('y')
        self.theta = ca.SX.sym('theta')
        self.states = ca.vertcat(
            self.x,
            self.y,
            self.theta
        )
        self.n_states = self.states.numel()

        # control symbolic variables
        self.v = ca.SX.sym('v')
        self.omega = ca.SX.sym('omega')

        self.controls = ca.vertcat(
            self.v,
            self.omega,
        )
        self.n_controls = self.controls.numel()

        # matrix containing all states over all time steps +1 (each column is a state vector)
        self.X = ca.SX.sym('X', self.n_states, self.N + 1)

        # matrix containing all control actions over all time steps (each column is an action vector)
        self.U = ca.SX.sym('U', self.n_controls, self.N)

        # coloumn vector for storing initial state and target state
        self.P = ca.SX.sym('P', self.n_states + self.n_states)

        # state weights matrix (Q_X, Q_Y, Q_THETA)
        self.Q = ca.diagcat(self.Q_x, self.Q_y, self.Q_theta)

        # controls weights matrix
        # controls weights matrix
        self.R = ca.diagcat(self.R1, self.R2)

        self.RHS = ca.vertcat(
            self.v*cos(self.theta),
            self.v*sin(self.theta),
            self.omega
        )
        # maps controls from [v,w].T [x,y,theta].T
        self.f = ca.Function('f', [self.states, self.controls], [self.RHS])
        
        self.cost_fn = 0  # cost function
        self.g = self.X[:, 0] - self.P[:self.n_states]  # constraints in the equation
        # runge kutta
        for k in range(self.N):
            self.st = self.X[:, k]
            self.con = self.U[:, k]
            self.cost_fn = self.cost_fn \
                + (self.st - self.P[self.n_states:]).T @ self.Q @ (self.st - self.P[self.n_states:]) \
                + self.con.T @ self.R @ self.con
            st_next = self.X[:, k+1]
            k1 = self.f(self.st, self.con)
            k2 = self.f(self.st + self.step_horizon/2*k1, self.con)
            k3 = self.f(self.st + self.step_horizon/2*k2, self.con)
            k4 = self.f(self.st + self.step_horizon * k3, self.con)
            st_next_RK4 = self.st + (self.step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, st_next - st_next_RK4)

        m = [-self.omega_max/self.v_max, self.omega_max/-self.v_min, -self.omega_min/self.v_max, self.omega_min/-self.v_min]

        for p in range(0,4): #Diff drive constraint in the input
            for k in range(0,self.N):
                self.con = self.U[:,k]; #con=control
                self.g = ca.vertcat(self.g, self.con[1]-m[p]*self.con[0])

        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        self.nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': self.g,
            'p': self.P
        }

        self.opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp_prob, self.opts)

        lbx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))
        ubx = ca.DM.zeros((self.n_states*(self.N+1) + self.n_controls*self.N, 1))

        lbx[0: self.n_states*(self.N+1): self.n_states] = self.x_min     # X lower bound
        lbx[1: self.n_states*(self.N+1): self.n_states] = self.y_min     # Y lower bound
        lbx[2: self.n_states*(self.N+1): self.n_states] = -ca.inf     # theta lower bound

        ubx[0: self.n_states*(self.N+1): self.n_states] = self.x_max      # X upper bound
        ubx[1: self.n_states*(self.N+1): self.n_states] = self.y_max      # Y upper bound
        ubx[2: self.n_states*(self.N+1): self.n_states] = ca.inf      # theta upper bound

        lbx[self.n_states*(self.N+1)::self.n_controls] = self.v_min       # v lower bound for all V
        lbx[self.n_states*(self.N+1)+1::self.n_controls] = self.omega_min       # v lower bound for all V

        ubx[self.n_states*(self.N+1)::self.n_controls] = self.v_max       # v lower bound for all V
        ubx[self.n_states*(self.N+1)+1::self.n_controls] = self.omega_max       # v lower bound for all V

        lbg = ca.DM.zeros((self.n_states*(self.N+1) + 4*(self.N), 1))  # constraints lower bound
        ubg = ca.DM.zeros((self.n_states*(self.N+1) + 4*(self.N), 1))  # constraints upper bound

        lbg[self.n_states*(self.N+1):self.n_states*(self.N+1)+2*self.N] = -ca.inf
        lbg[self.n_states*(self.N+1)+2*self.N:] = self.omega_min

        ubg[self.n_states*(self.N+1):self.n_states*(self.N+1)+2*self.N] = self.omega_max
        ubg[self.n_states*(self.N+1)+2*self.N:] = ca.inf

        self.args = {
            'lbg': lbg,  # constraints lower bound
            'ubg': ubg,  # constraints upper bound
            'lbx': lbx,
            'ubx': ubx
        }

    def simulation(self):
        self.t0 = 0
        self.u = 0
        self.current_state = ca.DM([self.x_init,self.y_init,self.theta_init])        # initial state
        self.target_state = ca.DM([self.x_target, self.y_target, self.theta_target])  # target state

        self.t = ca.DM(self.t0)

        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.current_state, 1, self.N+1)         # initial state full

        self.mpc_iter = 0
        self.cat_states = DM2Arr(self.X0) #saves states for plotting

        self.cat_controls = DM2Arr(self.u0[:, 0]) #saves controls for plotting
        self.times = np.array([[0]])
    def saveForPlot(self):
        self.cat_states = np.dstack((
            self.cat_states,
            DM2Arr(self.X0)
        ))

        self.cat_controls = np.vstack((
            self.cat_controls,
            DM2Arr(self.u[:, 0])
        ))
        self.t = np.vstack((
            self.t,
            self.t0
        ))
    def shift_timestep(self):
        f_value = self.f(self.current_state, self.u[:, 0])
        self.current_state = ca.DM.full(self.current_state + (self.step_horizon * f_value))

        self.t0 = self.t0 + self.step_horizon
        self.u0 = ca.horzcat(
            self.u[:, 1:],
            ca.reshape(self.u[:, -1], -1, 1)
        )
    def update_args(self):
        self.args['p'] = ca.vertcat(
            self.current_state,    # current state
            self.target_state   # target state
        )
        # optimization variable current state
        self.args['x0'] = ca.vertcat(
            ca.reshape(self.X0, self.n_states*(self.N+1), 1),
            ca.reshape(self.u0, self.n_controls*self.N, 1)
        )

def DM2Arr(dm):
    return np.array(dm.full())
 
###############################################################################

if __name__ == '__main__':
    robotA = MPC(0,0,0,3,4,pi)
    robotA.simulation() #initialize simulation values
    main_loop = time()  # return time in sec
    while (ca.norm_2(robotA.current_state - robotA.target_state) > 1e-1) and (robotA.mpc_iter * robotA.step_horizon < robotA.sim_time):
        t1 = time()
        robotA.update_args()

        sol = robotA.solver(
            x0=robotA.args['x0'],
            lbx=robotA.args['lbx'],
            ubx=robotA.args['ubx'],
            lbg=robotA.args['lbg'],
            ubg=robotA.args['ubg'],
            p=robotA.args['p']
        )

        robotA.u = ca.reshape(sol['x'][robotA.n_states * (robotA.N + 1):], robotA.n_controls, robotA.N)
        robotA.X0 = ca.reshape(sol['x'][: robotA.n_states * (robotA.N+1)], robotA.n_states, robotA.N+1)

        robotA.saveForPlot()

        robotA.shift_timestep()

        # print(X0)
        robotA.X0 = ca.horzcat(
            robotA.X0[:, 1:],
            ca.reshape(robotA.X0[:, -1], -1, 1)
        )
        t2 = time()

        robotA.times = np.vstack((
            robotA.times,
            t2-t1
        ))

        robotA.mpc_iter = robotA.mpc_iter + 1

    main_loop_time = time()
    ss_error = ca.norm_2(robotA.current_state - robotA.target_state)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(robotA.times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

    # simulate
    simulate(robotA.cat_states, robotA.cat_controls, robotA.times, robotA.step_horizon, robotA.N,
             np.array([robotA.x_init, robotA.y_init, robotA.theta_init, robotA.x_target, robotA.y_target, robotA.theta_target]), save=True)
