import numpy as np
import time
from environment import Environment
import config as env_config

def run_environment_test():
    """
    An example of how to initialize and use the Environment class.
    """
    # --- Test 1: Using the internal vehicle model ---
    print("="*50)
    print("--- Testing Environment with Internal Model ---")
    print("="*50)

    # Initialize the environment to use the internal model
    env_model = Environment(
        vehicle_params=env_config.VEHICLE_PARAMS,
        initial_state=env_config.X0_INIT,
        T=env_config.T,
        use_real_time_sim=False
    )
    
    # Simulate a few steps
    num_steps = 5
    # Define some control inputs [steering_rate, jerk]
    control_sequence = [
        np.array([0.1, 0.5]),
        np.array([0.1, 0.5]),
        np.array([-0.05, -0.2]),
        np.array([-0.05, -0.2]),
        np.array([0.0, 0.0])
    ]

    for i in range(num_steps):
        control = control_sequence[i]
        print(f"\nStep {i+1}/{num_steps}")
        print(f"Applying control: {control}")
        
        start_time = time.time()
        new_state, reward = env_model.step(control)
        end_time = time.time()
        
        print(f"New State: {np.round(new_state, 4)}")
        print(f"Reward: {reward}")
        print(f"Step computation time: {end_time - start_time:.6f}s")


    # --- Test 2: Simulating the real-time environment communication ---
    print("\n" + "="*50)
    print("--- Testing Environment with Real-Time Simulation ---")
    print("="*50)
    
    # Reset the environment to use the real-time sim flag
    env_real_time = Environment(
        vehicle_params=env_config.VEHICLE_PARAMS,
        initial_state=env_config.X0_INIT,
        T=env_config.T,
        use_real_time_sim=True
    )
    
    # Simulate a few steps
    for i in range(3):
        control = control_sequence[i]
        print(f"\nStep {i+1}/{3}")
        
        start_time = time.time()
        # The step function will now print messages simulating communication
        new_state, reward = env_real_time.step(control)
        end_time = time.time()

        print(f"Reward: {reward}")
        print(f"Step computation time: {end_time - start_time:.6f}s")


if __name__ == '__main__':
    run_environment_test()
