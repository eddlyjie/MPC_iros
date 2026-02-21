"""
Configuration file for the NMPC simulation.
Contains vehicle parameters, track settings, and MPC controller tuning.
"""
import numpy as np
import casadi as ca

DEBUG = False  # Set to True to enable debug messages

# --- Simulation Settings ---
T = 0.1  # Time step [s]
N = 30  # Prediction horizon
SIM_TIME = 40.0  # Total simulation time [s]
USE_LAST_FEASIBLE_ON_FAIL = False # Strategy for solver failures
DELAY_TIME = 0.00  # Delay time for the second vehicle [s]
CAR1_SOLVER = 1
CAR2_SOLVER = 2 # 0: Original Solver, 1: Enhanced Solver, 2: Enhanced Hard Constraint Solver

# RL Parameters
RL_PARAMS = {
    'alpha': 0.0003,
    'beta': 0.0003,
    'batch_size': 256
}

# --- Vehicle Parameters ---
VEHICLE_PARAMS = {
    'la': 1.3776,         # CoG to front axle [m]
    'lb': 1.4924,         # CoG to rear axle [m]
    'hcg': 0.36,          # CoG height [m]
    'Izz': 4760.0,        # Yaw moment of inertia [kg*m^2]
    'muf': 0.98 * 1.05,   # Front tire friction coefficient
    'mur': 1.03 * 1.05,   # Rear tire friction coefficient
    'M': 1970.0,          # Total mass [kg]
    'g': 9.81,            # Gravitational acceleration [m/s^2]
    'Caf': 222000.0,      # Front cornering stiffness [N/rad]
    'Car': 284000.0,      # Rear cornering stiffness [N/rad]
    'ratio': 0.72,        # Front-wheel drive torque distribution ratio
    'max_speed': 35.0     # Maximum longitudinal speed [m/s]
}

# --- Vehicle 2 Parameters ---
VEHICLE_PARAMS_2 = {
    'la': 1.3776,         # CoG to front axle [m]
    'lb': 1.4924,         # CoG to rear axle [m]
    'hcg': 0.36,          # CoG height [m]
    'Izz': 4760.0,        # Yaw moment of inertia [kg*m^2]
    'muf': 0.98 * 1.3,   # Front tire friction coefficient
    'mur': 1.03 * 1.3,   # Rear tire friction coefficient
    'M': 1970.0,          # Total mass [kg]
    'g': 9.81,            # Gravitational acceleration [m/s^2]
    'Caf': 222000.0,      # Front cornering stiffness [N/rad]
    'Car': 284000.0,      # Rear cornering stiffness [N/rad]
    'ratio': 0.72,        # Front-wheel drive torque distribution ratio
    'max_speed': 35.0     # Maximum longitudinal speed [m/s]
}

# --- Track Data ---
TRACK_PARAMS = {
    'maxNumBlocks': 35,
    'mat_file': 'track/map/track_02.mat'
    # 'mat_file': 'track/thunder/processed_ThunderHill.mat'
}

# --- MPC Cost Function Weights ---
MPC_COSTS = {
    'R': np.diag([0.01, 0.01]),    # Control input cost
    'w_v': 0.5,                   # Lateral velocity cost
    'w_sa': 0.0,                  # Steering angle cost
    'w_r': 1.0,                   # Yaw rate cost
    'w_ax': 0.02,                 # Longitudinal acceleration cost
    'w_tube': 10.0,               # Track boundary cost
    'w_goal': 10.0,               # Terminal cost (progress)
    # --- MODIFIED/NEW PARAMETERS ---
    'w_avoid': 1500.0,              # Global weight for the avoidance cost term
    'w_reward': 0.0,                # Global weight for the reward cost term (currently off)
    'AVOID_RADIUS': 4.0,            # Radius [m] for the avoidance penalty function
    "W_AVOID" : 1000.0
}

MPC_COSTS_2 = {
    'R': np.diag([0.01, 0.01]),    # Control input cost
    'w_v': 0.5,                   # Lateral velocity cost
    'w_sa': 0.0,                  # Steering angle cost
    'w_r': 1.0,                   # Yaw rate cost
    'w_ax': 0.02,                 # Longitudinal acceleration cost
    'w_tube': 10.0,               # Track boundary cost
    'w_goal': 10.0,               # Terminal cost (progress)
    # --- MODIFIED/NEW PARAMETERS ---
    'w_avoid': 1500.0,              # Global weight for the avoidance cost term
    'w_reward': 0.0,                # Global weight for the reward cost term (currently off)
    'AVOID_RADIUS': 4.0,            # Radius [m] for the avoidance penalty function
    "W_AVOID" : 1000.0
}

# --- State and Control Boundaries ---
STATE_BOUNDS = [
    (-ca.inf, ca.inf),         # x (pos)
    (-ca.inf, ca.inf),         # y (pos)
    (-7.0, 7.0),               # v (lat_vel)
    (-2 * np.pi, 2 * np.pi),   # r (yaw_rate)
    (-ca.inf, ca.inf),         # psi (yaw_angle)
    (5.0, VEHICLE_PARAMS['max_speed']), # ux (long_vel)
    (-np.pi / 9, np.pi / 9),   # sa (steer_angle)
    (-ca.inf, ca.inf)          # ax (long_accel)
]

STATE_BOUNDS_2 = [
    (-ca.inf, ca.inf),         # x (pos)
    (-ca.inf, ca.inf),         # y (pos)
    (-7.0, 7.0),               # v (lat_vel)
    (-2 * np.pi, 2 * np.pi),   # r (yaw_rate)
    (-ca.inf, ca.inf),         # psi (yaw_angle)
    (5.0, VEHICLE_PARAMS_2['max_speed']), # ux (long_vel)
    (-np.pi / 9, np.pi / 9),   # sa (steer_angle)
    (-ca.inf, ca.inf)          # ax (long_accel)
]

CONTROL_BOUNDS = [
    (-0.15, 0.15),   # sr (steer_rate)
    (-30.0, 30.0)    # jx (jerk)
]

CONTROL_BOUNDS_2 = [
    (-0.15, 0.15),   # sr (steer_rate)
    (-30.0, 30.0)    # jx (jerk)
]

#change the map to test
CKPT = "models/map/02"


# --- Initial State ---
# x0 = [x, y, v, r, psi, ux, sa, ax]
# training values
#thunderhill
# X0_INIT = np.array([349.0, -577.0, 0.0, 0.0, -3.0227, 15.0, 0.0, 0.0]).reshape(-1, 1)
# X0_INIT_2 = np.array([355.0, -577.0, 0.0, 0.0, -3.0227, 15.0, 0.0, 0.0]).reshape(-1, 1)
#08
# X0_INIT = np.array([143, 370, 0.0, 0.0, 0.5, 15.0, 0.0, 0.0]).reshape(-1, 1)
# X0_INIT_2 = np.array([141, 365, 0.0, 0.0, 0.5, 15.0, 0.0, 0.0]).reshape(-1, 1)
#02
X0_INIT = np.array([-24, -380, 0.0, 0.0, -1.5, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([-24, -375, 0.0, 0.0, -1.5, 15.0, 0.0, 0.0]).reshape(-1, 1)


# # testing values
# X0_INIT = np.array([313.0, -891.0, 0.0, 0.0, 0.7297, 15.0, 0.0, 0.0]).reshape(-1, 1)
# X0_INIT_2 = np.array([307.69, -895.0, 0.0, 0.0, 0.5591, 15.0, 0.0, 0.0]).reshape(-1, 1)
# X0_INIT = np.array([2.780309713107110e+02, -9.036261981441817e+02, 0.0, 0.0, 0.089998862428691, 5.0, 0.0, 0.0]).reshape(-1, 1)
