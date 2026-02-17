import numpy as np
import casadi as ca
import time

# Import the new solver class and required configs/modules
from solver_class import MPC_Solver
import config as mpc_config  # Assuming config.py exists with all parameters

def run_simulation():
    """
    An example of how to initialize and use the MPC_Solver class to run a short simulation.
    """
    print("--- Initializing MPC Solver ---")
    
    # 1. Initialize the Solver Class
    # This single step creates the solver, loads track data, and sets up everything needed.
    mpc_solver = MPC_Solver(
        T=mpc_config.T,
        N=mpc_config.N,
        vehicle_params=mpc_config.VEHICLE_PARAMS,
        track_params=mpc_config.TRACK_PARAMS,
        mpc_costs=mpc_config.MPC_COSTS
    )
    
    # --- Simulation Setup ---
    x0 = mpc_config.X0_INIT.copy()
    x_history = [np.array(x0).flatten()]
    u_history = []
    
    # The initial guesses for the solver are now managed internally by the MPC_Solver class
    
    print("--- Starting Simulation Loop ---")
    sim_time_seconds = mpc_config.SIM_TIME
    mpc_iter = 0
    start_time = time.time()
    
    while mpc_iter * mpc_config.T < sim_time_seconds:
        # 2. Get the solution from the solver using the updated method
        # This function handles initialization and warm-starting automatically.
        solution = mpc_solver.solve_and_update(x0)
        
        if solution is None:
            print(f"Stopping simulation due to solver failure at t={mpc_iter * mpc_config.T:.2f}s.")
            break
            
        # Extract the optimal control sequence 
        u_optimal = ca.reshape(solution['x'][mpc_solver.n_states * (mpc_solver.N + 1):], mpc_solver.n_controls, mpc_solver.N)
        
        # The first control input in the sequence is the one we apply to the vehicle
        control_to_apply = u_optimal[:, 0]
        u_history.append(control_to_apply.full().flatten())
        
        # The warm-start variables (u0, X0_pred) are updated automatically inside solve_and_update.

        # 3. Apply the control to the vehicle model to simulate one step forward
        # This uses RK4 integration, just like in your original train.py file.
        model_func = mpc_solver.model_func
        k1 = model_func(x0, control_to_apply)
        k2 = model_func(x0 + mpc_config.T/2 * k1, control_to_apply)
        k3 = model_func(x0 + mpc_config.T/2 * k2, control_to_apply)
        k4 = model_func(x0 + mpc_config.T * k3, control_to_apply)
        
        # Update state for the next iteration. .full() converts from CasADi DM to a NumPy array.
        x0 = (x0 + (mpc_config.T / 6) * (k1 + 2*k2 + 2*k3 + k4)).full()
        
        x_history.append(x0.flatten())
        mpc_iter += 1
        
        if mpc_iter % 10 == 0:
            # FIX: Access the scalar value from the numpy array for formatting.
            # x0 is a 2D numpy array (e.g., shape (8,1)), so we use flatten() or direct indexing.
            current_speed = x0.flatten()[5]
            print(f"Sim Time: {mpc_iter * mpc_config.T:.2f}s, Current Speed: {current_speed:.2f} m/s")

    end_time = time.time()
    print(f"--- Simulation Finished ---")
    print(f"Ran {mpc_iter} MPC steps in {end_time - start_time:.2f} seconds.")
    # The x_history and u_history arrays can now be used for analysis or plotting.
    

if __name__ == '__main__':
    # NOTE: To run this example, you must have a 'config.py' file in the same
    # directory with all the required dictionaries (VEHICLE_PARAMS, TRACK_PARAMS, 
    # MPC_COSTS, T, N, X0_INIT, etc.), as well as the 'track_utils.py', 
    # 'vehicle_model.py', and the track's .mat file.
    try:
        run_simulation()
    except (ImportError, AttributeError, FileNotFoundError) as e:
        print("\n" + "="*50)
        print(f"Could not run example: {e}")
        print("Please ensure you have all required dependency files and configurations.")
        print("="*50)

