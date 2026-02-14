import numpy as np
import time
from sac_agent import SACAgent
from environment_rl import RLEnvironment
from visualization import Visualization
import track_utils_rl as track_utils
import config

def main():
    """
    Main function to deploy and visualize the trained SAC agent.
    """
    print("--- Initializing RL Agent Deployment ---")

    # 1. Load Track and Agent
    track_data = track_utils.load_track_data(config.TRACK_PARAMS['mat_file'])
    if track_data is None:
        return

    # Use the same environment as in training to get states
    initial_state_car1 = np.array([track_data['center_line'][0,0], track_data['center_line'][0,1], 0, 0, track_data['lane_yaw'][0,0], 10, 0, 0])
    initial_state_car2 = np.array([track_data['center_line'][100,0], track_data['center_line'][100,1], 0, 0, track_data['lane_yaw'][100,0], 10, 0, 0])
    
    env = RLEnvironment(
        vehicle_params=config.VEHICLE_PARAMS,
        initial_state_car1=initial_state_car1,
        initial_state_car2=initial_state_car2,
        T=config.T,
        track_data=track_data
    )

    agent = SACAgent(
        input_dims=env.observation_space.shape,
        n_actions=env.action_space.shape[0]
    )
    agent.load_models() # Load the trained models

    # 2. Initialize Visualization
    vis = Visualization(track_data)
    
    # 3. Main Simulation Loop
    observation = env.reset()
    done = False
    
    print("\n--- Running Simulation with Trained Agent ---")
    while not done:
        # Choose the deterministic action
        action = agent.choose_action(observation) 
        
        # The environment step will internally call the NMPC solver
        observation_, reward, done, info = env.step(action)

        # Get Cartesian states for visualization from the info dict
        current_state_car1 = info['state_car1_cartesian']
        current_state_car2 = info['state_car2_cartesian']
        predicted_states_car1 = info['predicted_states_car1']
        
        vis.update_plot(current_state_car1, predicted_states_car1, current_state_car2)
        
        observation = observation_
        
        time.sleep(config.T) # Run in pseudo real-time

if __name__ == '__main__':
    main()
