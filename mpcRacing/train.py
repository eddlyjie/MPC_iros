import numpy as np
from sac_agent import SACAgent
from environment_rl import RLEnvironment # We will create this modified environment
import track_utils_rl as track_utils # We will create this modified track_utils
import config

def main():
    """
    Main function to train the Soft Actor-Critic agent.
    """
    print("--- Initializing RL Training Environment ---")

    # 1. Load Track Data
    track_data = track_utils.load_and_process_track(config.TRACK_PARAMS['mat_file'])
    if track_data is None:
        return

    # 2. Initialize RL Environment
    # The initial state needs to be defined, e.g., starting line
    initial_state_car1 = np.array([track_data['center_line'][0,0], track_data['center_line'][0,1], 0, 0, track_data['lane_yaw'][0,0], 10, 0, 0])
    initial_state_car2 = np.array([track_data['center_line'][100,0], track_data['center_line'][100,1], 0, 0, track_data['lane_yaw'][100,0], 10, 0, 0])

    env = RLEnvironment(
        vehicle_params=config.VEHICLE_PARAMS,
        initial_state_car1=initial_state_car1,
        initial_state_car2=initial_state_car2,
        T=config.T,
        track_mat_file=config.TRACK_PARAMS['mat_file']
    )

    # 3. Initialize SAC Agent
    # The input_dims should match the observation space from the environment
    agent = SACAgent(
        env=env,
        gamma=config.RL_PARAMS['gamma'],
        batch_size=config.RL_PARAMS['batch_size'],
        alpha=config.RL_PARAMS['alpha'],
        beta=config.RL_PARAMS['beta']
    )

    n_games = 1000
    best_score = -np.inf
    
    print("\n--- Starting Training Loop ---")
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            
            score += reward
            observation = observation_

        if score > best_score:
            best_score = score
            agent.save_models()

        avg_score = score # In a real scenario, you'd average over many episodes
        print(f'Episode {i}, Score: {score:.2f}, Avg Score: {avg_score:.2f}')

if __name__ == '__main__':
    main()
