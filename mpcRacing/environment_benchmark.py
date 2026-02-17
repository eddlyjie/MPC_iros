import numpy as np
import gymnasium as gym
from gymnasium import spaces
import casadi as ca
import os

from solver_class_avoidance import MPC_Solver as Enhanced_MPC_Solver
from solver_class import MPC_Solver as Original_MPC_Solver
from solver_class_benchmark import MPC_Solver as RL_MPC_Solver
import simulation_vehicle_model as vehicle_model
import track_utils_rl as track_utils
import config
from utils import debug

print("INFO: Using 'environment_benchmark.py' for the RL Environment.")

class RLEnvironment(gym.Env):
    """
    Gymnasium-compatible Reinforcement Learning environment for the dual-vehicle racing simulation.
    
    This version trains an agent to output 12 values, representing reward targets (s_rel, d_rel, w)
    at 4 specific collocation points.
    """
    def __init__(self, vehicle_params, initial_state_car1, initial_state_car2, T, track_mat_file):
        super(RLEnvironment, self).__init__()
        
        self.T = T
        self.N = config.N # Store prediction horizon for use in step method
        self.time_elapsed = 0.0
        self.vehicle_params = vehicle_params
        
        # --- NEW: Define the specific collocation points for the RL agent to control ---
        self.rl_collocation_indices = [1, 4, 10, 20]
        if max(self.rl_collocation_indices) >= self.N:
            raise ValueError("A specified collocation index is outside the prediction horizon N.")
            
        self.num_rl_points = len(self.rl_collocation_indices)
        
        # --- Load and Process Track Data ---
        self.track_data = track_utils.load_and_process_track(track_mat_file)
        if self.track_data is None:
            raise FileNotFoundError(f"Fatal: Could not load or process track data from {track_mat_file}")
            
        # --- NEW: Check for required track interpolators ---
        required_interpolators = ['curvature_interp', 'd_left_interp', 'd_right_interp']
        for key in required_interpolators:
            if key not in self.track_data:
                raise KeyError(f"Fatal: 'track_data' dictionary loaded from 'track_utils_rl.py' is missing the required interpolator: '{key}'. "
                               f"Please update 'load_and_process_track' to provide this.")
        # --- END OF NEW CHECK ---

        self.centerline = self.track_data['center_line']
        self.left_bound = self.track_data['left_bound']
        self.right_bound = self.track_data['right_bound']
        
        # --- NEW: Store track and reward/termination state variables ---
        self.track_length = self.track_data['s_ref'][-1]
        self.track_half_width = 6 # Based on previous logic
        self.boundary_penalty_start_distance = self.track_half_width - 0.3
        self.car2_lead_timer = 0.0
        self.prev_s1 = 0.0
        
        # --- NEW: Track sampling parameters for observation ---
        self.num_track_points = 15
        self.track_point_spacing = 10.0  # 10m intervals
        
        self.initial_state_car1 = initial_state_car1
        self.initial_state_car2 = initial_state_car2

        # --- FIX: Ensure states are flattened 1D arrays ---
        self.state_car1 = np.array(self.initial_state_car1).flatten()
        self.state_car2 = np.array(self.initial_state_car2).flatten()
        
        # --- Solvers ---
        self.solver_car1 = RL_MPC_Solver(T = T, N = config.N, vehicle_params = config.VEHICLE_PARAMS, track_params = config.TRACK_PARAMS, mpc_costs = config.MPC_COSTS, state_bounds = config.STATE_BOUNDS, control_bounds = config.CONTROL_BOUNDS)
        
        # Conditionally initialize the solver for Car 2 based on config
        self.car2_use_enhanced_solver = (hasattr(config, 'CAR2_SOLVER') and (config.CAR2_SOLVER == 1 or config.CAR2_SOLVER == 2))
        if self.car2_use_enhanced_solver and config.CAR2_SOLVER == 1:
            print("INFO: Car 2 is using the Enhanced MPC Solver for avoidance.")
            self.solver_car2 = Enhanced_MPC_Solver(T=config.T, N=config.N, vehicle_params=config.VEHICLE_PARAMS_2, track_params=config.TRACK_PARAMS, mpc_costs=config.MPC_COSTS_2, state_bounds=config.STATE_BOUNDS_2, control_bounds=config.CONTROL_BOUNDS_2)
        elif self.car2_use_enhanced_solver and config.CAR2_SOLVER == 2:
            print("INFO: Car 2 is using the Enhanced Hard Constraint MPC Solver for avoidance.")
            from solver_class_avoidance_hard import MPC_Solver as Enhanced_Hard_MPC_Solver
            self.solver_car2 = Enhanced_Hard_MPC_Solver(T=config.T, N=config.N, vehicle_params=config.VEHICLE_PARAMS_2, track_params=config.TRACK_PARAMS, mpc_costs=config.MPC_COSTS_2, state_bounds=config.STATE_BOUNDS_2, control_bounds=config.CONTROL_BOUNDS_2)
        else:
            print("INFO: Car 2 is using the Original MPC Solver.")
            self.solver_car2 = Original_MPC_Solver(T=config.T, N=config.N, vehicle_params=config.VEHICLE_PARAMS_2, track_params=config.TRACK_PARAMS, mpc_costs=config.MPC_COSTS_2)

        # --- Vehicle Dynamics Model (for state propagation) ---
        self.dynamics_func = self._create_callable_model_function(vehicle_params)
        self.dynamics_func_2 = self._create_callable_model_function(config.VEHICLE_PARAMS_2)
        
        # --- RL Spaces (MODIFIED) ---
        action_low = np.tile(np.array([0, -5]), self.num_rl_points)  # s_low, d_low
        action_high = np.tile(np.array([50, 5]), self.num_rl_points) # s_high, d_high
        
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float32)
        
        # --- MODIFICATION: Observation space shape changed ---
        # Base observation size = 10
        # Track observation size = 15 points * 3 values/point (kappa, d_left, d_right) = 45
        # Total observation size = 10 + 45 = 55
        self.observation_space_size = 10 + self.num_track_points * 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_size,), dtype=np.float32)
        # --- END OF MODIFICATION ---

    def _create_callable_model_function(self, vehicle_params):
        x_sym = ca.SX.sym('x', 8)
        u_sym = ca.SX.sym('u', 2)
        model_expr_func = vehicle_model.get_vehicle_model(vehicle_params)
        rhs = model_expr_func(x_sym, u_sym)
        return ca.Function('f', [x_sym, u_sym], [rhs])

    def _rk4_step(self, state, control):
        k1 = self.dynamics_func(state, control)
        k2 = self.dynamics_func(state + self.T / 2 * k1, control)
        k3 = self.dynamics_func(state + self.T / 2 * k2, control)
        k4 = self.dynamics_func(state + self.T * k3, control)
        next_state = state + self.T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_state.full().flatten()
    
    def _rk4_step_2(self, state, control):
        k1 = self.dynamics_func_2(state, control)
        k2 = self.dynamics_func_2(state + self.T / 2 * k1, control)
        k3 = self.dynamics_func_2(state + self.T / 2 * k2, control)
        k4 = self.dynamics_func_2(state + self.T * k3, control)
        next_state = state + self.T / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_state.full().flatten()
        
    # --- MODIFIED FUNCTION ---
    def _get_observation(self):
        """
        Calculates the observation vector in Frenet coordinates, including future track geometry.
        
        Observation:
        [
            --- Base (10 values) ---
            d1, phi_e1, vs1, vd1, r1,         # Ego state (5)
            rel_s, d2,                        # Relative state (2)
            phi_e2, vs2, vd2,                 # Other vehicle state (3)
            
            --- Track Geometry (45 values) ---
            kappa_1, d_left_1, d_right_1,     # Point @ s + 10m
            kappa_2, d_left_2, d_right_2,     # Point @ s + 20m
            ...
            kappa_15, d_left_15, d_right_15   # Point @ s + 150m
        ]
        """
        # --- 1. EGO VEHICLE (CAR 1) ---
        s1, d1, phi_e1 = track_utils.cartesian_to_frenet(
            self.state_car1[0], self.state_car1[1], self.state_car1[4], self.track_data
        )
        s1, d1, phi_e1 = float(s1), float(d1), float(phi_e1)

        vx1 = self.state_car1[5] # ego_vx
        vy1 = self.state_car1[2] # ego_vy
        r1  = self.state_car1[3] # ego_r

        cos_phi_e1 = np.cos(phi_e1)
        sin_phi_e1 = np.sin(phi_e1)
        vs1 = vx1 * cos_phi_e1 - vy1 * sin_phi_e1
        vd1 = vx1 * sin_phi_e1 + vy1 * cos_phi_e1

        # --- 2. OTHER VEHICLE (CAR 2) ---
        s2, d2, phi_e2 = track_utils.cartesian_to_frenet(
            self.state_car2[0], self.state_car2[1], self.state_car2[4], self.track_data
        )
        s2, d2, phi_e2 = float(s2), float(d2), float(phi_e2)
        
        vx2 = self.state_car2[5] # other_vx
        vy2 = self.state_car2[2] # other_vy
        
        cos_phi_e2 = np.cos(phi_e2)
        sin_phi_e2 = np.sin(phi_e2)
        vs2 = vx2 * cos_phi_e2 - vy2 * sin_phi_e2
        vd2 = vx2 * sin_phi_e2 + vy2 * cos_phi_e2

        # --- 3. RELATIVE STATE ---
        rel_s = s2 - s1
        if rel_s > self.track_length / 2.0:
            rel_s -= self.track_length
        if rel_s < -self.track_length / 2.0:
            rel_s += self.track_length

        # --- 4. ASSEMBLE BASE OBSERVATION ---
        base_observation = np.array([
            d1, phi_e1, vs1, vd1, r1,       # Ego
            rel_s, d2,                      # Relative
            phi_e2, vs2, vd2                # Other
        ], dtype=np.float32)
        
        # --- 5. GET FUTURE TRACK GEOMETRY ---
        track_obs_list = []
        # Get interpolator functions from track_data
        kappa_interp = self.track_data['curvature_interp']
        d_left_interp = self.track_data['d_left_interp']
        d_right_interp = self.track_data['d_right_interp']

        for i in range(1, self.num_track_points + 1):
            # Calculate the future 's' value, handling track wraparound
            s_future = (s1 + i * self.track_point_spacing) % self.track_length
            
            # Get track properties at the future 's' value
            # .full().item() converts from CasADi DM/SX to a standard Python float
            kappa = kappa_interp(s_future).full().item()
            d_left = d_left_interp(s_future).full().item()
            d_right = d_right_interp(s_future).full().item()
            
            track_obs_list.extend([kappa, d_left, d_right])
            
        track_observation = np.array(track_obs_list, dtype=np.float32)

        # --- 6. COMBINE AND RETURN ---
        full_observation = np.concatenate([base_observation, track_observation])
        
        # Final check on shape
        if full_observation.shape[0] != self.observation_space_size:
             debug(f"WARNING: Observation shape mismatch. Expected {self.observation_space_size}, got {full_observation.shape[0]}")

        return full_observation
    # --- END OF MODIFICATION ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state_car1 = np.copy(self.initial_state_car1).flatten()
        self.state_car2 = np.copy(self.initial_state_car2).flatten()
        self.time_elapsed = 0.0
        # --- NEW: Reset stateful reward variables ---
        self.car2_lead_timer = 0.0
        # Get initial progress for car 1
        s1, _, _ = track_utils.cartesian_to_frenet(self.state_car1[0], self.state_car1[1], self.state_car1[4], self.track_data)
        self.prev_s1 = float(s1)
        
        observation = self._get_observation()
        # Return observation and an empty info dict (standard for gym.reset)
        return observation, {}

    def step(self, action):
        debug(f"DEBUG: Received action from RL agent: {action}")
        self.time_elapsed += self.T

        rl_actions = np.reshape(action, (self.num_rl_points, 2)) # Shape: (4, 2)
        s_ego, _, _ = track_utils.cartesian_to_frenet(self.state_car1[0], self.state_car1[1], self.state_car1[4], self.track_data)
        s_ego = float(s_ego)
        
        avoid_pts = np.zeros((self.N, 3))
        reward_pts = np.zeros((self.N, 3))

        s_target = s_ego
        for i, idx in enumerate(self.rl_collocation_indices):
            s_rel, d_rel = rl_actions[i]
            w = 1.0  # Dummy
            s_target += s_rel
            x_target, y_target = track_utils.frenet_to_cartesian(s_target, d_rel, self.track_data)
            reward_pts[idx] = [float(x_target), float(y_target), float(w)]

        # 4. Solve for Car 1's control input
        solution1, solve_time = self.solver_car1.solve_and_update(self.state_car1, avoid_points=avoid_pts, reward_points=reward_pts)
        
        if solution1 is not None:
            n_states = self.solver_car1.n_states
            n_controls = self.solver_car1.n_controls
            N = self.solver_car1.N
            u_optimal = ca.reshape(solution1['x'][n_states * (N + 1):], n_controls, N)
            u1 = u_optimal.full()[:, 0]
            debug(f"DEBUG: Car 1 control input: {u1}")
            pred1 = self.solver_car1.X0_pred.full()
        else:
            debug("WARNING: Solver for Car 1 failed. Applying zero control.")
            u1 = np.zeros(2)
            pred1 = None
            
        # 5. Solve for Car 2's control input
        if self.time_elapsed < config.DELAY_TIME:
            u2 = np.zeros(2) 
            pred2 = None
        else:
            if self.car2_use_enhanced_solver:
                avoid_pts_for_car2 = np.zeros((self.N, 3))
                if pred1 is None:
                    avoid_pts_for_car2[:, 0] = self.state_car1[0]
                    avoid_pts_for_car2[:, 1] = self.state_car1[1]
                    avoid_pts_for_car2[:, 2] = config.MPC_COSTS['W_AVOID']
                else:
                    avoid_pts_for_car2[:, 0] = pred1[0, 1:].flatten()
                    avoid_pts_for_car2[:, 1] = pred1[1, 1:].flatten()
                    avoid_pts_for_car2[:, 2] = config.MPC_COSTS['W_AVOID']

                solution2 = self.solver_car2.solve_and_update(self.state_car2, avoid_points=avoid_pts_for_car2)
                if solution2 is not None:
                    n_states = self.solver_car2.n_states
                    n_controls = self.solver_car2.n_controls
                    N = self.solver_car2.N
                    u_optimal_2 = ca.reshape(solution2['x'][n_states * (N + 1):], n_controls, N)
                    u2 = u_optimal_2.full()[:, 0]
                    pred2 = self.solver_car2.X0_pred.full()
                    debug(f"DEBUG: Car 2 control input (enhanced solver): {u2}")
                else:
                    debug("WARNING: Enhanced solver for Car 2 failed. Applying zero control.")
                    u2 = np.zeros(2)
                    pred2 = None
            else:
                solution2 = self.solver_car2.solve_and_update(self.state_car2)
                if solution2:
                    u2 = solution2['u'][:, 0]
                else:
                    debug("WARNING: Original solver for Car 2 failed. Applying zero control.")
                    u2 = np.zeros(2)
                    pred2 = None # Make sure pred2 is None if solver fails

        # 6. Propagate vehicle states
        self.state_car1 = self._rk4_step(self.state_car1, u1)
        if self.time_elapsed >= config.DELAY_TIME: # Only update Car 2 after delay
             self.state_car2 = self._rk4_step_2(self.state_car2, u2)
        # Handle case where car 2 is delayed but we still need its state for info
        elif pred2 is None:
             pred2 = np.tile(self.state_car2, (self.solver_car2.n_states, self.N + 1))


        # 7. Calculate Reward and Termination Conditions
        reward, terminated = self._calculate_reward_and_terminal()
        truncated = False # Standard gym practice
        
        observation = self._get_observation()
        s1 = float(track_utils.cartesian_to_frenet(self.state_car1[0], self.state_car1[1], self.state_car1[4], self.track_data)[0])
        s2 = float(track_utils.cartesian_to_frenet(self.state_car2[0], self.state_car2[1], self.state_car2[4], self.track_data)[0])
        info = {
            'state_car1_cartesian': self.state_car1,
            'state_car2_cartesian': self.state_car2,
            'predicted_states_car1': pred1,
            'predicted_states_car2': pred2,
            'reward_pts': reward_pts,
            'solve_time_car1': solve_time,
            's_car1': s1,
            's_car2': s2
        }
        
        return observation, reward, terminated, truncated, info
        
    def _calculate_reward_and_terminal(self):
        
        reward = 0.0
        terminated = False

        s1, d1, _ = track_utils.cartesian_to_frenet(self.state_car1[0], self.state_car1[1], self.state_car1[4], self.track_data)
        s2, d2, _ = track_utils.cartesian_to_frenet(self.state_car2[0], self.state_car2[1], self.state_car2[4], self.track_data)
        s1, d1, s2 = float(s1), float(d1), float(s2)

        # REWARD 1: Track Progression
        delta_s1 = s1 - self.prev_s1
        if delta_s1 < -self.track_length / 2.0:
            delta_s1 += self.track_length
        elif delta_s1 > self.track_length / 2.0:
            delta_s1 -= self.track_length
            
        reward += delta_s1 * 1.0
        self.prev_s1 = s1 

        # REWARDS 2, 4, 5 & TERMINATE 2: Progress vs. Car 2
        progress_diff = s1 - s2
        if progress_diff > self.track_length / 2.0:
            progress_diff -= self.track_length
        if progress_diff < -self.track_length / 2.0:
            progress_diff += self.track_length

        if progress_diff > 0.1: # Car 1 is ahead
            reward += progress_diff * 0.1
            reward += 0.1
            self.car2_lead_timer = 0.0
        else: # Car 1 is behind
            reward -= 0.1
            self.car2_lead_timer += self.T

        if self.car2_lead_timer > 3.0:
            reward -= 300
            terminated = True

        # REWARD 3: Minimize Speed Difference
        dist = np.linalg.norm(self.state_car1[:2] - self.state_car2[:2])
        if dist < 40.0:
            speed1 = self.state_car1[5] # ux
            speed2 = self.state_car2[5] # ux
            speed_diff = abs(speed1 - speed2)
            reward -= speed_diff * 0.05 

        # REWARD 6 & TERMINATE 1: Boundary Penalty
        # --- MODIFICATION: Use dynamic boundaries if they are narrower than the fixed one ---
        # We still use self.track_half_width for termination, as changing that
        # logic is a bigger step. But we can add penalties for the *actual* boundaries.
        abs_d = abs(d1)
        
        # Get current boundaries
        current_d_left = self.track_data['d_left_interp'](s1).full().item()
        current_d_right = self.track_data['d_right_interp'](s1).full().item()
        
        # Check if penetrating the *actual* boundaries
        if d1 > (current_d_left - 0.3): # Too far left
             penetration = d1 - (current_d_left - 0.3)
             reward -= (penetration / 0.3)**2 * 5.0
        elif d1 < (current_d_right + 0.3): # Too far right
             penetration = (current_d_right + 0.3) - d1
             reward -= (penetration / 0.3)**2 * 5.0

        # TERMINATE 1: Out of bounds (using the original fixed width)
        if abs_d > self.track_half_width:
            reward -= 200
            terminated = True
            
        # TERMINATE 3: Collision
        if dist < 0.5: 
            reward += 500  
            terminated = True

        # TERMINATE: Time limit
        if self.time_elapsed >= config.SIM_TIME:
            terminated = True
            reward += 1000.0
            
        return reward, terminated

    def render(self):
        pass

    def close(self):
        del self.solver_car1
        del self.solver_car2
        print("Environment closed.")

