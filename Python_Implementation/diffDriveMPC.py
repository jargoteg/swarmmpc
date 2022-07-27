from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
#from robot_client import DM2Arr
from simulation_code import simulate

#Main MPC class where all variables are defined. Intended to work for multi-robot case
#Robot limits in velocity
MAX_SPEED = 100
MAX_ROTATION = pi/4

ARENA_SIZE_X = 100
ARENA_SIZE_Y = 100

class Constraints: #Handles the constraints for the MPC controller
    def __init__(self):
        self.lbx = ca.DM([])
        self.ubx = ca.DM([])
        self.lbg = ca.DM([])
        self.ubg = ca.DM([])
        self.g = ca.DM([])

    def add_lbx_and_ubx(self,robot):
        self.lbx = ca.DM.zeros((robot.n_states*(robot.N+1) + robot.n_controls*robot.N, 1))
        self.ubx = ca.DM.zeros((robot.n_states*(robot.N+1) + robot.n_controls*robot.N, 1))

        self.lbx[0: robot.n_states*(robot.N+1): robot.n_states] = robot.x_min     # X lower bound
        self.lbx[1: robot.n_states*(robot.N+1): robot.n_states] = robot.y_min     # Y lower bound
        self.lbx[2: robot.n_states*(robot.N+1): robot.n_states] = -ca.inf     # theta lower bound

        self.ubx[0: robot.n_states*(robot.N+1): robot.n_states] = robot.x_max      # X upper bound
        self.ubx[1: robot.n_states*(robot.N+1): robot.n_states] = robot.y_max      # Y upper bound
        self.ubx[2: robot.n_states*(robot.N+1): robot.n_states] = ca.inf      # theta upper bound

        self.lbx[robot.n_states*(robot.N+1)::robot.n_controls] = robot.v_min       # v lower bound for all V
        self.lbx[robot.n_states*(robot.N+1)+1::robot.n_controls] = robot.omega_min       # v lower bound for all V

        self.ubx[robot.n_states*(robot.N+1)::robot.n_controls] = robot.v_max       # v lower bound for all V
        self.ubx[robot.n_states*(robot.N+1)+1::robot.n_controls] = robot.omega_max       # v lower bound for all V
    def add_g_equality_cons(self,robot): #equality constraints between next state and predicted next state
        self.g = robot.X[:, 0] - robot.P[:robot.n_states]  # constraints in the equation
        # runge kutta
        for k in range(robot.N):
            robot.st = robot.X[:, k]
            robot.con = robot.U[:, k]
            robot.cost_fn = robot.cost_fn \
                + (robot.st - robot.P[robot.n_states:]).T @ robot.Q @ (robot.st - robot.P[robot.n_states:]) \
                + robot.con.T @ robot.R @ robot.con
            st_next = robot.X[:, k+1]
            k1 = robot.f(robot.st, robot.con)
            k2 = robot.f(robot.st + robot.step_horizon/2*k1, robot.con)
            k3 = robot.f(robot.st + robot.step_horizon/2*k2, robot.con)
            k4 = robot.f(robot.st + robot.step_horizon * k3, robot.con)
            st_next_RK4 = robot.st + (robot.step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            self.g = ca.vertcat(self.g, st_next - st_next_RK4)
            #current_length = self.g.size()[0]
            this_lbg = ca.DM.zeros((robot.n_states*(robot.N+1),1))
            print(this_lbg)
            this_ubg = ca.DM.zeros((robot.n_states*(robot.N+1),1))
            self.lbg = appendDM(self.lbg,this_lbg)
            self.ubg = appendDM(self.ubg,this_ubg)

    def add_g_dynamics(self,robot):
        m = [-robot.omega_max/robot.v_max, robot.omega_max/-robot.v_min, -robot.omega_min/robot.v_max, robot.omega_min/-robot.v_min]

        for p in range(0,4): #Diff drive constraint in the input
            for k in range(0,robot.N):
                robot.con = robot.U[:,k]; #con=control
                self.g = ca.vertcat(self.g, robot.con[1]-m[p]*robot.con[0])
        self.lbg = appendDM(self.lbg,ca.DM.ones((2*robot.N),1)*(-ca.inf))
        self.lbg = appendDM(self.lbg,ca.DM.ones((2*robot.N),1)*(robot.omega_min))
        self.ubg = appendDM(self.ubg,ca.DM.ones((2*robot.N),1)*(robot.omega_max))
        self.ubg = appendDM(self.ubg,ca.DM.ones((2*robot.N),1)*(ca.inf))

class MPC:

    #MPC weights
    Q_x = 100
    Q_y = 100
    Q_theta = 2
    R1 = 1
    R2 = 0.01
    
    N = 1               # number of look ahead steps
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
        constraint = Constraints()
        constraint.add_g_equality_cons(self)
        constraint.add_g_dynamics(self)


        self.OPT_variables = ca.vertcat(
            self.X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
            self.U.reshape((-1, 1))
        )
        self.nlp_prob = {
            'f': self.cost_fn,
            'x': self.OPT_variables,
            'g': constraint.g,
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

        
        constraint.add_lbx_and_ubx(self)
        constraint.add_g_equality_cons(self)
        constraint.add_g_dynamics(self)

        self.args = {
            'lbg': constraint.lbg,  # constraints lower bound
            'ubg': constraint.ubg,  # constraints upper bound
            'lbx': constraint.lbx,
            'ubx': constraint.ubx
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

def Arr2DM(arr):
    return ca.DM(arr)

def appendDM(dm,ap):
    arr = DM2Arr(dm)
    arr = np.append(arr,ap)
    return Arr2DM(arr)
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
