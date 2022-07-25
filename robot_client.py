#!/usr/bin/env python3

from robots import robots, server_none, server_york, server_manchester, server_sheffield

import asyncio
import websockets
import json
import signal
import time
import sys
from enum import Enum
import time
import random
import inspect
import casadi as ca
from casadi import sin, cos, pi
import numpy as np
import colorama
from colorama import Fore



colorama.init(autoreset=True)


##
# Replace `server_none` with one of `server_york`, `server_sheffield`, or `server_manchester` here,
#  or specify a custom server IP address as a string.
# All ports should remain at 80.
##
server_address = server_sheffield
server_port = 80
robot_port = 80
##

if len(server_address) == 0:
    raise Exception(f"Enter local tracking server address on line {inspect.currentframe().f_lineno - 6}, "
                    f"then re-run this script.")


# Persistent Websockets!!!!!!!!!!!!!!!!
# https://stackoverflow.com/questions/59182741/python-websockets-lib-client-persistent-connection-with-class-implementation


##
# Handle Ctrl+C termination
# https://stackoverflow.com/questions/2148888/python-trap-all-signals

SIGNALS_TO_NAMES_DICT = dict((getattr(signal, n), n) \
    for n in dir(signal) if n.startswith('SIG') and '_' not in n)

# https://github.com/aaugustin/websockets/issues/124
__kill_now = False


def __set_kill_now(signum, frame):
    print('\nReceived signal:', SIGNALS_TO_NAMES_DICT[signum], str(signum))
    global __kill_now
    __kill_now = True


signal.signal(signal.SIGINT, __set_kill_now)
signal.signal(signal.SIGTERM, __set_kill_now)


def kill_now() -> bool:
    global __kill_now
    return __kill_now

# Ctrl+C termination handled
##


server_connection = None
active_robots = {}
ids = []


# Robot states to use in the controller
class RobotState(Enum):
    FORWARDS = 1
    BACKWARDS = 2
    LEFT = 3
    RIGHT = 4
    STOP = 5


# Main Robot class to keep track of robot states
class Robot:

    # 3.6V should give an indication that the battery is getting low, but this value can be experimented with.
    # Battery percentage might be a better
    BAT_LOW_VOLTAGE = 3.6

    # Firmware on both robots accepts wheel velocities between -100 and 100.
    # This limits the controller to fit within that.
    MAX_SPEED = 100

    def __init__(self, robot_id):
        self.id = robot_id
        self.connection = None

        self.position = []
        self.orientation = 0
        self.neighbours = {}
        self.tasks = {}

        self.teleop = False
        self.state = RobotState.STOP
        self.ir_readings = []
        self.battery_charging = False
        self.battery_voltage = 0
        self.battery_percentage = 0

        self.turn_time = time.time()


        ##########################################
        #MPC init
        # setting matrix_weights' variables
        self.Q_x = 100
        self.Q_y = 100
        self.Q_theta = 2
        self.R1 = 1
        self.R2 = 0.01

        self.step_horizon = 0.1  # time between steps in seconds
        self.N = 1              # number of look ahead steps

        # specs
        self.x_init = 1
        self.y_init = 0
        self.theta_init = 0
        self.x_target = 7
        self.y_target = 10
        self.theta_target = pi

        #Robot specific parameters
        self.v_max = 100
        self.v_min = -self.v_max
        self.omega_max = pi/4 
        self.omega_min = -self.omega_max

        #Size of the arena
        self.x_max = 100      
        self.x_min = -100   
        self.y_max = 100
        self.y_min = -100

        
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
        # maps controls from [va, vb, vc, vd].T to [vx, vy, omega].T
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

        t0 = 0
        self.state_init = ca.DM([0,0,0])        # initial state
        self.state_target = ca.DM([self.x_target, self.y_target, self.theta_target])  # target state

        self.t = ca.DM(t0)

        self.u0 = ca.DM.zeros((self.n_controls, self.N))  # initial control
        self.X0 = ca.repmat(self.state_init, 1, self.N+1)         # initial state full

        # Pi-puck IR is more sensitive than Mona, so use higher threshold for obstacle detection
        if robot_id < 31:
            # Pi-puck
            self.ir_threshold = 200
        else:
            # Mona
            self.ir_threshold = 80

# Connect to websocket server of tracking server
async def connect_to_server():
    uri = f"ws://{server_address}:{server_port}"
    connection = await websockets.connect(uri)

    print("Opening connection to server: " + uri)

    awake = await check_awake(connection)

    if awake:
        print("Server is awake")
        global server_connection
        server_connection = connection
    else:
        print("Server did not respond")


# Connect to websocket server running on each of the robots
async def connect_to_robots():
    for id in active_robots.keys():
        ip = robots[id]
        if ip != '':
            uri = f"ws://{ip}:{robot_port}"
            connection = await websockets.connect(uri)

            print("Opening connection to robot:", uri)

            awake = await check_awake(connection)

            if awake:
                print(f"Robot {id} is awake")
                active_robots[id].connection = connection
            else:
                print(f"Robot {id} did not respond")
        else:
            print(f"No IP defined for robot {id}")


# Check if robot is awake by sending the "check_awake" command to its websocket server
async def check_awake(connection):
    awake = False

    try:
        message = {"check_awake": True}

        # Send request for data and wait for reply
        await connection.send(json.dumps(message))
        reply_json = await connection.recv()
        reply = json.loads(reply_json)

        # Reply should contain "awake" with value True
        awake = reply["awake"]

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

    return awake


# Ask a list of robot IDs for all their sensor data (proximity + battery)
async def get_robot_data(ids):
    await message_robots(ids, get_data)


# Send all commands to a list of robots IDs (motors + LEDs)
async def send_robot_commands(ids):
    await message_robots(ids, send_commands)


# Tell a list of robot IDs to stop
async def stop_robots(ids):
    await message_robots(ids, stop_robot)


# Send a message to a list of robot IDs
# Uses multiple websockets code from:
# https://stackoverflow.com/questions/49858021/listen-to-multiple-socket-with-websockets-and-asyncio
async def message_robots(ids, function):
    loop = asyncio.get_event_loop()
    tasks = []
    for id, robot in active_robots.items():
        if id in ids:
            tasks.append(loop.create_task(function(robot)))
    await asyncio.gather(*tasks)


# Get robots' virtual sensor data from the tracking server, for our active robots
async def get_server_data():
    try:
        global ids
        message = {"get_robots": True}

        # Send request for data and wait for reply
        await server_connection.send(json.dumps(message))
        reply_json = await server_connection.recv()
        reply = json.loads(reply_json)

        # Filter reply from the server, based on our active robots of interest
        filtered_reply = {int(k): v for (k, v) in reply.items() if int(k) in active_robots.keys()}

        ids = list(filtered_reply.keys())

        # Receive robot virtual sensor data from the server
        for id, robot in filtered_reply.items():
            active_robots[id].position = robot["position"]
            active_robots[id].orientation = robot["orientation"]
            # Filter out any neighbours that aren't our active robots
            active_robots[id].neighbours = {k: v for (k, v) in robot["neighbours"].items() if int(k) in active_robots.keys()}
            active_robots[id].tasks = robot["tasks"]
            print(f"Robot {id}")
            print(f"Position = {active_robots[id].position}")
            print(f"Orientation = {active_robots[id].orientation}")
            print(f"Neighbours = {active_robots[id].neighbours}")
            print(f"Tasks = {active_robots[id].tasks}")
            print()

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


# Stop robot from moving and turn off its LEDs
async def stop_robot(robot):
    try:
        # Turn off LEDs and motors when killed
        message = {"set_leds_colour": "off", "set_motor_speeds": {}}
        message["set_motor_speeds"]["left"] = 0
        message["set_motor_speeds"]["right"] = 0
        await robot.connection.send(json.dumps(message))

        # Send command message
        await robot.connection.send(json.dumps(message))
    except Exception as e:
        print(f"{type(e).__name__}: {e}")


# Get IR and battery readings from robot
async def get_data(robot):
    try:
        message = {"get_ir": True, "get_battery": True}

        # Send request for data and wait for reply
        await robot.connection.send(json.dumps(message))
        reply_json = await robot.connection.recv()
        reply = json.loads(reply_json)

        robot.ir_readings = reply["ir"]

        robot.battery_voltage = reply["battery"]["voltage"]
        robot.battery_percentage = reply["battery"]["percentage"]

        print(f"[Robot {robot.id}] IR readings: {robot.ir_readings}")
        print("[Robot {}] Battery: {:.2f}V, {}%" .format(robot.id,
                                                         robot.battery_voltage,
                                                         robot.battery_percentage))

    except Exception as e:
        print(f"{type(e).__name__}: {e}")


# Send motor and LED commands to robot
# This function also performs the obstacle avoidance and teleop algorithm state machines
async def send_commands(robot):
    try:
        # Turn off LEDs and motors when killed
        if kill_now():
            message = {"set_leds_colour": "off", "set_motor_speeds": {}}
            message["set_motor_speeds"]["left"] = 0
            message["set_motor_speeds"]["right"] = 0
            await robot.connection.send(json.dumps(message))

        # Construct command message
        message = {}

        if robot.teleop:
            # Teleoperation mode
            message["set_leds_colour"] = "blue"
            if robot.state == RobotState.FORWARDS:
                left = right = robot.MAX_SPEED
            elif robot.state == RobotState.BACKWARDS:
                left = right = -robot.MAX_SPEED
            elif robot.state == RobotState.LEFT:
                left = -robot.MAX_SPEED * 0.8
                right = robot.MAX_SPEED * 0.8
            elif robot.state == RobotState.RIGHT:
                left = robot.MAX_SPEED * 0.8
                right = -robot.MAX_SPEED * 0.8
            elif robot.state == RobotState.STOP:
                left = right = 0
        else:
            # Autonomous mode
            if robot.state == RobotState.FORWARDS:
                left = right = robot.MAX_SPEED
                if (time.time() - robot.turn_time > 0.5) and any(ir > robot.ir_threshold for ir in robot.ir_readings):
                    robot.turn_time = time.time()
                    robot.state = random.choice((RobotState.LEFT, RobotState.RIGHT))
            elif robot.state == RobotState.BACKWARDS:
                left = right = -robot.MAX_SPEED
                robot.turn_time = time.time()
                robot.state = RobotState.FORWARDS
            elif robot.state == RobotState.LEFT:
                left = -robot.MAX_SPEED
                right = robot.MAX_SPEED
                if time.time() - robot.turn_time > random.uniform(0.5, 1.0):
                    robot.turn_time = time.time()
                    robot.state = RobotState.FORWARDS
            elif robot.state == RobotState.RIGHT:
                left = robot.MAX_SPEED
                right = -robot.MAX_SPEED
                if time.time() - robot.turn_time > random.uniform(0.5, 1.0):
                    robot.turn_time = time.time()
                    robot.state = RobotState.FORWARDS
            elif robot.state == RobotState.STOP:
                left = right = 0
                robot.turn_time = time.time()
                robot.state = RobotState.FORWARDS

        #MPC(robot.state)
        left, right = await my_MPC(robot)
        
                
        print("------")
        print("posx:",robot.position[0])
        print("posy:",robot.position[1])
        #near_robots = list(robot.neighbours.keys())
        near_tasks = list(robot.tasks.keys())
        print(near_tasks)
        print("")
        
        if near_tasks:
            for i in near_tasks:
                print(i)
                rg = robot.tasks[i]['range']
                br = robot.tasks[i]['bearing']
                print("range: ",rg)
                print("bearing: ",br)
             
        #if near_tasks:
            #print("tasks",near_tasks[0])
            #print("------")
        print("Motor values: L:",left)
        print("R:",right)
        message["set_motor_speeds"] = {}
        message["set_motor_speeds"]["left"] = left
        message["set_motor_speeds"]["right"] = right

        # Set RGB LEDs based on battery voltage
        if robot.battery_voltage < robot.BAT_LOW_VOLTAGE:
            message["set_leds_colour"] = "red"
        else:
            message["set_leds_colour"] = "green"

        # Send command message
        await robot.connection.send(json.dumps(message))

    except Exception as e:
        print(f"{type(e).__name__}: {e}")

def DM2Arr(dm):
    return np.array(dm.full())

async def my_MPC(robot):
    left = 0
    right = 0

    near_tasks = list(robot.tasks.keys())
    if near_tasks:
            for i in near_tasks:
                print(i)
                rel_distance = robot.tasks[i]['range']
                bearing = robot.tasks[i]['bearing']
                print("range: ",rel_distance)
                print("bearing: ",bearing)
    
    relative_target_state = ca.DM([robot.position[0]+rel_distance*cos(bearing),robot.position[1]+rel_distance*sin(bearing),0])  #target x and y measured states
    robot_current_state = ca.DM([robot.position[0],robot.position[1],robot.orientation]) 
    print("robotpos",robot_current_state)
    print("targetpos",relative_target_state)
    robot.args['p'] = ca.vertcat(
        robot_current_state,    # current state
        relative_target_state   # target state
    )
    # optimization variable current state (for warm start probably)
    robot.args['x0'] = ca.vertcat(
        ca.reshape(robot.X0, robot.n_states*(robot.N+1), 1),
        ca.reshape(robot.u0, robot.n_controls*robot.N, 1)
    )
    #print(args['p'])
    sol = robot.solver(
        x0=robot.args['x0'],
        lbx=robot.args['lbx'],
        ubx=robot.args['ubx'],
        lbg=robot.args['lbg'],
        ubg=robot.args['ubg'],
        p=robot.args['p']
    )                           #solve optimal control problem

    u = ca.reshape(sol['x'][robot.n_states * (robot.N + 1):], robot.n_controls, robot.N) #optimal controls
    left = DM2Arr(u[0, 0])
    right = DM2Arr(u[0, 0])
    return float(left),float(right)

# Menu state for teleop control input
class MenuState(Enum):
    START = 1
    SELECT = 2
    DRIVE = 3


# Send message to a websocket server
async def send_message(websocket, message):
    await websocket.send(json.dumps({"prompt": message}))


# Handle message received on the websocket server
# Used for teleoperation code, which is controlled by running teleop_client.py in a separate terminal.
async def handler(websocket):
    state = MenuState.START
    robot_id = ""
    valid_robots = list(active_robots.keys())
    forwards = "w"
    backwards = "s"
    left = "a"
    right = "d"
    stop = " "
    release = "q"

    async for packet in websocket:
        message = json.loads(packet)
        if "key" in message:
            key = message["key"]
            if key == "teleop_start":
                state = MenuState.START
            elif key == "teleop_stop":
                if state == MenuState.DRIVE:
                    id = int(robot_id)
                    active_robots[id].teleop = False
                    active_robots[id].state = RobotState.STOP

            if state == MenuState.START:
                await send_message(websocket, f"\r\nEnter robot ID ({valid_robots}), then press return: ")
                robot_id = ""
                state = MenuState.SELECT
            elif state == MenuState.SELECT:
                if key == "\r":
                    valid = False
                    try:
                        if int(robot_id) in valid_robots:
                            valid = True
                            await send_message(websocket, f"\r\nControlling robot ({release} to release): " + robot_id)
                            await send_message(websocket, f"\r\nControls: Forwards = {forwards}; Backwards = {backwards}; Left = {left}; Right = {right}; Stop = SPACE")
                            active_robots[int(robot_id)].teleop = True
                            state = MenuState.DRIVE
                    except ValueError:
                        pass

                    if not valid:
                        await send_message(websocket, "\r\nInvalid robot ID, try again: ")
                        robot_id = ""
                        state = MenuState.SELECT
                else:
                    await send_message(websocket, key)
                    robot_id = robot_id + key
            elif state == MenuState.DRIVE:
                id = int(robot_id)
                if key == release:
                    await send_message(websocket, "\r\nReleasing control of robot: " + robot_id)
                    active_robots[id].teleop = False
                    active_robots[id].state = RobotState.STOP
                    state = MenuState.START
                elif key == forwards:
                    await send_message(websocket, "\r\nDriving forwards")
                    active_robots[id].state = RobotState.FORWARDS
                elif key == backwards:
                    await send_message(websocket, "\r\nDriving backwards")
                    active_robots[id].state = RobotState.BACKWARDS
                elif key == left:
                    await send_message(websocket, "\r\nTurning left")
                    active_robots[id].state = RobotState.LEFT
                elif key == right:
                    await send_message(websocket, "\r\nTurning right")
                    active_robots[id].state = RobotState.RIGHT
                elif key == stop:
                    await send_message(websocket, "\r\nStopping")
                    active_robots[id].state = RobotState.STOP
                else:
                    await send_message(websocket, "\r\nUnrecognised command")


# Main entry point for robot control client sample code
if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    loop.run_until_complete(connect_to_server())

    if server_connection is None:
        print(Fore.RED + "[ERROR]: No connection to server")
        sys.exit(1)

    # Specify robot IDs to work with here. For example for robots 11-15 use:
    #  robot_ids = range(11, 16)
    robot_ids = range(43, 44)

    if len(robot_ids) == 0:
        raise Exception(f"Enter range of robot IDs to control on line {inspect.currentframe().f_lineno - 3}, "
                        f"then re-run this script.")

    # Create Robot objects
    for robot_id in robot_ids:
        if robots[robot_id] != '':
            active_robots[robot_id] = Robot(robot_id)
        else:
            print(f"No IP defined for robot {robot_id}")

    # Create websockets connections to robots
    print(Fore.GREEN + "[INFO]: Connecting to robots")
    loop.run_until_complete(connect_to_robots())

    if not active_robots:
        print(Fore.RED + "[ERROR]: No connection to robots")
        sys.exit(1)

    # Listen for any keyboard input from teleop websocket client
    print(Fore.GREEN + "[INFO]: Starting teleop server")
    start_server = websockets.serve(ws_handler=handler, host=None, port=7000, ping_interval=None, ping_timeout=None)
    loop.run_until_complete(start_server)

    # Only communicate with robots that were successfully connected to
    while True:
        # Request all robot virtual sensor data from the tracking server
        print(Fore.GREEN + "[INFO]: Requesting data from tracking server")
        loop.run_until_complete(get_server_data())

        # Request sensor data from detected robots
        print(Fore.GREEN + "[INFO]: Robots detected:", ids)
        print(Fore.GREEN + "[INFO]: Requesting data from detected robots")
        loop.run_until_complete(get_robot_data(ids))

        # Calculate next step of control algorithm, and send commands to robots
        print(Fore.GREEN + "[INFO]: Sending commands to detected robots")
        loop.run_until_complete(send_robot_commands(ids))

        print()

        # TODO: Close websocket connections
        if kill_now():
            loop.run_until_complete(stop_robots(robot_ids))  # Kill all robots, even if not visible
            break

        # Sleep until next control cycle
        time.sleep(0.1)
