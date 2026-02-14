import numpy as np
import time
import matplotlib.pyplot as plt
import casadi as ca

# MODIFIED: Import both the enhanced and original MPC solver classes with aliases
from solver_class_avoidance import MPC_Solver as Enhanced_MPC_Solver
from solver_class import MPC_Solver as Original_MPC_Solver

# Import the other necessary components of our simulation
from environment import Environment
from visualization import Visualization
import config
import track_utils

def main():
    """
    Main function to run the MPC simulation with real-time visualization.
    This version uses an enhanced solver for the primary vehicle and the
    original solver for the secondary vehicle.
    """
    print("--- Initializing Simulation Components ---\n")

    # 1. Load Track Data
    track_data = track_utils.load_track_data(config.TRACK_PARAMS['mat_file'])
    if track_data is None:
        print("Exiting due to track loading error.")
        return

    # 2. Initialize MPC Solvers for both cars
    # MODIFIED: The primary vehicle (Car 1) uses the enhanced solver
    print("Initializing Enhanced Solver for Car 1...")
    mpc_solver = Enhanced_MPC_Solver(
        T=config.T, N=config.N, vehicle_params=config.VEHICLE_PARAMS,
        track_params=config.TRACK_PARAMS, mpc_costs=config.MPC_COSTS,
        state_bounds=config.STATE_BOUNDS, control_bounds=config.CONTROL_BOUNDS
    )
    
    # MODIFIED: The second car uses the original solver
    if config.CAR2_SOLVER == 0:
        print("Initializing Original Solver for Car 2...")
        mpc_solver2 = Original_MPC_Solver(
            T=config.T, N=config.N, vehicle_params=config.VEHICLE_PARAMS_2,
            track_params=config.TRACK_PARAMS, mpc_costs=config.MPC_COSTS,
            state_bounds=config.STATE_BOUNDS_2, control_bounds=config.CONTROL_BOUNDS
        )
    elif config.CAR2_SOLVER == 1:
        print("Initializing Enhanced Solver for Car 2...")
        mpc_solver2 = Enhanced_MPC_Solver(
            T=config.T, N=config.N, vehicle_params=config.VEHICLE_PARAMS_2,
            track_params=config.TRACK_PARAMS, mpc_costs=config.MPC_COSTS,
            state_bounds=config.STATE_BOUNDS_2, control_bounds=config.CONTROL_BOUNDS
        )

    # 3. Initialize the Environment
    # Find a starting position on the track
    start_idx = 10
    initial_pos = track_data['center_line'][start_idx, :]
    target_pos = track_data['center_line'][start_idx + 100, :]
    initial_yaw = track_data['lane_yaw'][start_idx, 0]
    initial_state = np.array([initial_pos[0], initial_pos[1], 0, 0, initial_yaw, 5, 0, 0])

    env = Environment(
        vehicle_params=config.VEHICLE_PARAMS,
        initial_state=initial_state,
        T=config.T,
        mpc_solver=mpc_solver2, # Pass the original solver for car 2
    )
    
    # 4. Initialize Visualization
    vis = Visualization(track_data)

    # --- Simulation Loop ---
    print("\n--- Starting Simulation Loop ---\n")
    start_time = time.time()
    num_steps = int(config.SIM_TIME / config.T)
    
    current_state = env.state
    current_state_car2 = env.state_car2

    for i in range(num_steps):
        # Create dynamic avoidance points for the enhanced solver
        # check if plot window is closed
        if not plt.fignum_exists(vis.fig.number):
            print("Visualization window closed. Ending simulation.")
            break
        avoid_points = np.zeros((3, config.N))
        reward_points = np.zeros((3, config.N)) 
        # to avoid numerical issues if no points are set, fill all columns of avoid_points with initial_pos = track_data['center_line'][start_idx-2, :]
        initial_pos = track_data['center_line'][start_idx-5, :]
        for col in range(config.N):
            avoid_points[0, col] = initial_pos[0]
            avoid_points[1, col] = initial_pos[1]
            avoid_points[2, col] = 0.0  # zero weight means no avoidance initially
        # fill reward points with current position
        for col in range(config.N):
            reward_points[0, col] = current_state[0]
            reward_points[1, col] = current_state[1]
            reward_points[2, col] = 0.0  # constant reward weight

        if current_state_car2 is not None and i > 20:
            avoid_x = current_state_car2[0]
            avoid_y = current_state_car2[1]
            avoid_weight = 15.0 
            
            avoid_points[0, :] = avoid_x
            avoid_points[1, :] = avoid_y
            avoid_points[2, :] = avoid_weight

        # a. Solve MPC for Car 1 using the enhanced solver with avoidance points
        # The method name 'solve' is assumed to be correct for your solver version
        solution = mpc_solver.solve_and_update(current_state, avoid_points=avoid_points, reward_points=reward_points)

        if solution is not None:
            # Extract first control input and predicted trajectory
            control_to_apply = mpc_solver.u0[:, 0].full().flatten()
            print(f"Step {i}: Control Applied to Car 1: {control_to_apply}")
            predicted_states = mpc_solver.X0_pred.full()
        else:
            print(f"Warning: Solver failed for Car 1 at step {i}. Applying zero control.")
            control_to_apply = np.zeros(mpc_solver.n_controls)
            predicted_states = None

        # b. Apply control to the environment and get states for BOTH cars
        # The environment will internally use the original solver for car 2
        next_state, reward, next_state_car2 = env.step(control_to_apply)

        # c. Update the visualization
        vis.update_plot(current_state, predicted_states, current_state_car2)

        # d. Update the states for the next iteration
        current_state = next_state
        current_state_car2 = next_state_car2
        
        if i % 20 == 0:
            speed_ms_car1 = current_state[5] if current_state is not None else 0
            speed_ms_car2 = current_state_car2[5] if current_state is not None else 0
            print(f"Sim Time: {i * config.T:.2f}s | Car 1 Speed: {speed_ms_car1:.2f} m/s | Car 2 Speed: {speed_ms_car2:.2f} m/s")

    end_time = time.time()
    total_time = end_time - start_time
    avg_step_time = total_time / num_steps if num_steps > 0 else 0
    print(f"\n--- Simulation Finished ---")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average step time: {avg_step_time*1000:.2f} ms")
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()

