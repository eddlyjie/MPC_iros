import numpy as np
import casadi as ca

# Assuming vehicle_model.py contains the get_vehicle_model function
import simulation_vehicle_model as vehicle_model
import config
# from solver_class import MPC_Solver

class Environment:
    """
    Manages the state of the vehicle and the simulation environment.

    This class can either propagate the vehicle's state using a provided
    vehicle model or simulate communication with an external, real-time
    simulation environment.
    """

    def __init__(self, vehicle_params, initial_state, T, mpc_solver, use_real_time_sim=False):
        """
        Initializes the environment.

        Args:
            vehicle_params (dict): A dictionary of vehicle parameters.
            initial_state (np.ndarray): The starting state of the vehicle.
            T (float): The sampling time (time step) for the simulation.
            use_real_time_sim (bool): If True, the environment will simulate
                                      communication with an external simulator.
                                      If False, it will use the internal model.
        """
        self.use_real_time_sim = use_real_time_sim
        self.vehicle_params = vehicle_params
        self.T = T
        self.state = np.array(initial_state).flatten()
        
        # NOTE: State definition is ['x', 'y', 'v', 'r', 'psi', 'ux', 'sa', 'ax']
        # The yaw angle 'psi' is at index 4.
        self.yaw_index = 4
        self.prev_yaw = self.state[self.yaw_index]

        # Car 2 (Delayed Vehicle) State ---
        self.state_car2 = np.array(initial_state).flatten()
        self.prev_yaw_car2 = self.state_car2[self.yaw_index]
        self.solver = mpc_solver # Store the solver instance
        self.time_elapsed = 0.0 # Timer to manage the 1s delay

        if not self.use_real_time_sim:
            # Get the vehicle model function for internal propagation
            self.model_func = vehicle_model.get_vehicle_model(self.vehicle_params)

        print(f"Environment initialized. Using real-time simulation: {self.use_real_time_sim}")

    def _propagate_state(self, current_state, prev_yaw, control_input):
        """
        Propagates a given state forward by one time step using RK4 integration.
        This is a helper function to avoid code duplication for the two cars.
        """
        # Ensure control is a CasADi DM type for the model function
        control_input_dm = ca.DM(control_input)

        # 4th Order Runge-Kutta (RK4) integration
        k1 = self.model_func(current_state, control_input_dm)
        k2 = self.model_func(current_state + self.T/2 * k1, control_input_dm)
        k3 = self.model_func(current_state + self.T/2 * k2, control_input_dm)
        k4 = self.model_func(current_state + self.T * k3, control_input_dm)

        next_state_dm = current_state + self.T/6 * (k1 + 2*k2 + 2*k3 + k4)
        next_state = next_state_dm.full().flatten()

        # Handle yaw angle wrapping to keep it within [-pi, pi]
        # This is important for consistency and preventing numerical issues
        current_yaw = next_state[self.yaw_index]
        if abs(current_yaw - prev_yaw) > np.pi:
            if current_yaw > prev_yaw:
                next_state[self.yaw_index] -= 2 * np.pi
            else:
                next_state[self.yaw_index] += 2 * np.pi
        
        new_prev_yaw = next_state[self.yaw_index]
        
        return next_state, new_prev_yaw
    
    def step(self, control_car1):
        """
        Propagates the environment one time step forward.

        Args:
            control_car1 (np.ndarray or ca.DM): The control input for car 1 to apply.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The new state of the vehicle.
                - float: The reward for the step.
        """
        if self.use_real_time_sim:
            # Option 1: Communicate with a real-time simulation environment
            print("\n--- Communicating with external simulator ---")
            print(f"Sending control input: {control_car1}")
            
            # This is a placeholder for your communication logic (e.g., via ROS, UDP)
            # You would send 'control_car1' and wait for a response.
            # For this example, we'll just return a dummy state.
            print("Waiting for response from simulator...")
            # Pretend we received a new state after some delay
            new_state = self._get_state_from_external_sim()
            self.state = new_state
            print(f"Received new state: {self.state.flatten()}")

        else:
            self.time_elapsed += self.T
            # --- Car 1 (Ego Vehicle) Propagation ---
            # Propagates its state using the control input from main_simulation.py
            next_state_car1, self.prev_yaw = self._propagate_state(self.state, self.prev_yaw, control_car1)
            self.state = next_state_car1

            # --- Car 2 (Delayed Vehicle) Propagation ---
            if self.time_elapsed < config.DELAY_TIME:
                # While delayed, car 2's state remains frozen
                next_state_car2 = self.state_car2
            else:
                # After DELAY_TIME, car 2 starts moving, controlled by its own solver call
                if config.CAR2_SOLVER == 0:
                    solution_car2 = self.solver.solve_and_update(self.state_car2)
                elif config.CAR2_SOLVER == 1:
                    avoid_points = np.zeros((3, config.N))
                    reward_points = np.zeros((3, config.N))
                    if next_state_car1 is not None:
                        avoid_x = next_state_car1[0]
                        avoid_y = next_state_car1[1]
                        avoid_weight = 15.0

                        avoid_points[0, :] = avoid_x
                        avoid_points[1, :] = avoid_y
                        avoid_points[2, :] = avoid_weight
                    solution_car2 = self.solver.solve_and_update(self.state_car2, avoid_points=avoid_points, reward_points=reward_points)
                if solution_car2:
                    # Extract the optimal control sequence for car 2
                    u_optimal_car2 = ca.reshape(solution_car2['x'][self.solver.n_states * (self.solver.N + 1):], self.solver.n_controls, self.solver.N)
                    control_car2 = u_optimal_car2[:, 0]
                else:
                    # If solver fails for car 2, apply zero control as a fallback
                    print("Warning: Solver failed for Car 2. Applying zero control.")
                    control_car2 = np.zeros(self.solver.n_controls)
                
                # Propagate car 2's state using its own calculated control
                next_state_car2, self.prev_yaw_car2 = self._propagate_state(self.state_car2, self.prev_yaw_car2, control_car2)
            
            self.state_car2 = next_state_car2

            # --- Return values ---
            reward = self._calculate_reward()
        
        self._unwrap_yaw()
        self._unwrap_yaw_car2()
        # The function now returns the states of both cars
        return self.state, reward, self.state_car2

    def _unwrap_yaw(self):
        """
        Unwraps the vehicle's yaw angle to be continuous.
        This modifies the internal state directly.
        """
        current_yaw = self.state[self.yaw_index]
        yaw_diff = current_yaw - self.prev_yaw
        
        # Check for wrap-around (jump from +pi to -pi or vice-versa)
        if yaw_diff > np.pi:
            self.state[self.yaw_index] -= 2 * np.pi
        elif yaw_diff < -np.pi:
            self.state[self.yaw_index] += 2 * np.pi
            
        # Update the previous yaw for the next step's calculation
        self.prev_yaw = self.state[self.yaw_index]

    def _unwrap_yaw_car2(self):
        """
        Unwraps the delayed vehicle's yaw angle to be continuous.
        This modifies the internal state directly.
        """
        current_yaw = self.state_car2[self.yaw_index]
        yaw_diff = current_yaw - self.prev_yaw_car2
        
        # Check for wrap-around (jump from +pi to -pi or vice-versa)
        if yaw_diff > np.pi:
            self.state_car2[self.yaw_index] -= 2 * np.pi
        elif yaw_diff < -np.pi:
            self.state_car2[self.yaw_index] += 2 * np.pi
            
        # Update the previous yaw for the next step's calculation
        self.prev_yaw_car2 = self.state_car2[self.yaw_index]

    def _get_state_from_external_sim(self):
        """
        Placeholder function to simulate receiving a state from an external source.
        In a real implementation, this would contain your communication logic.
        """
        # In a real scenario, you'd parse an incoming message.
        # Here, we'll just slightly modify the current state as a dummy response.
        dummy_response_state = self.state + np.random.randn(len(self.state)) * 0.01
        return dummy_response_state.flatten()

    def _calculate_reward(self):
        """
        Calculates the reward based on the current state.
        
        This is a placeholder and should be implemented with your desired
        reward logic (e.g., based on progress, stability, staying on track).
        """
        # For now, as requested, it returns 0.
        return 0.0

    def reset(self, initial_state):
        """
        Resets the environment to a given initial state.

        Args:
            initial_state (np.ndarray): The state to reset to.
        """
        self.state = np.array(initial_state).flatten()
        # Also reset the previous yaw tracker
        self.prev_yaw = self.state[self.yaw_index]
        print(f"\nEnvironment reset to state: {self.state}")
        return self.state

