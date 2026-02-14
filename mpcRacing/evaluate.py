import numpy as np
import time
import os
import re # Added for parsing filenames
import argparse # For command-line arguments

# Assuming these modules are available in your directory
from environment_rl import RLEnvironment
from sac_agent import SACAgent
from visualization_rl import Visualization # Use the visualization file from context
import config

def find_latest_model(chkpt_dir):
    """
    Finds the latest saved model in the checkpoint directory.
    Returns the suffix (e.g., "save10") and the corresponding number (10).
    """
    if not os.path.exists(chkpt_dir):
        return None, 0

    files = os.listdir(chkpt_dir)
    # Regex to find files like 'actor_sac_save10.pth'
    save_files = [f for f in files if re.match(r'actor_sac_save(\d+)\.pth', f)]
    
    if not save_files:
        return None, 0

    # Extract numbers and find the max
    max_num = -1
    for f in save_files:
        match = re.search(r'(\d+)', f)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
                
    if max_num == -1:
        return None, 0
        
    latest_suffix = f"save{max_num}"
    return latest_suffix, max_num

def evaluate():
    """
    Main function to run RL model evaluation.
    """
    print("--- Initializing RL Evaluation Environment ---")

    # --- Configurable Parameters ---
    N_EVAL_EPISODES = 10 # Number of episodes to run for evaluation
    MODEL_SUFFIX = None # Set this to a specific suffix like 'save10' to load a specific model
    SAVE_VIDEO = False # Whether to save videos of the episodes
    VIDEO_DIR = "eval_videos" # Directory to save videos

    # 1. Initialize Environment
    track_mat_file = config.TRACK_PARAMS['mat_file']
    initial_state_car1 = config.X0_INIT
    initial_state_car2 = config.X0_INIT_2

    env = RLEnvironment(
        vehicle_params=config.VEHICLE_PARAMS,
        initial_state_car1=initial_state_car1,
        initial_state_car2=initial_state_car2,
        T=config.T,
        track_mat_file=track_mat_file
    )
    
    # 2. Initialize RL Agent
    # Note: We still initialize the SAC agent fully, but will only use the actor for inference.
    agent = SACAgent(
        env=env,
        alpha=config.RL_PARAMS['alpha'],
        beta=config.RL_PARAMS['beta'],
        batch_size=config.RL_PARAMS['batch_size'],
        fc1_dims=512,
        fc2_dims=512
    )
    
    # 3. Initialize Visualization
    print("Initializing Visualization...")
    vis = Visualization(env.track_data)

    # 4. Load Model
    if MODEL_SUFFIX is None:
        # Find and use the latest model if no specific suffix is provided
        latest_suffix, _ = find_latest_model(agent.actor.checkpoint_dir)
        MODEL_SUFFIX = latest_suffix

    if MODEL_SUFFIX:
        print(f"Attempting to load model: {MODEL_SUFFIX}...")
        try:
            agent.load_models(MODEL_SUFFIX)
            print("... Models loaded successfully.")
        except Exception as e:
            print(f"Error loading models for suffix '{MODEL_SUFFIX}': {e}")
            print("Cannot proceed with evaluation. Exiting.")
            vis.close()
            return
    else:
        print("No trained models found to load. Exiting.")
        vis.close()
        return

    # --- 5. Create Video Directory ---
    if SAVE_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        print(f"Videos will be saved to '{VIDEO_DIR}' directory.")

    total_scores = []
    
    print(f"\n--- Starting Evaluation for {N_EVAL_EPISODES} Episodes ---")
    try:
        for i in range(N_EVAL_EPISODES):
            observation, _ = env.reset()
            terminated = False
            truncated = False
            score = 0
            step_count = 0
            
            print(f"--- Running Episode {i+1}/{N_EVAL_EPISODES} ---")

            # --- Start Video Recording for this episode ---
            if SAVE_VIDEO and vis.is_open:
                video_filename = os.path.join(VIDEO_DIR, f"model_{MODEL_SUFFIX}_episode_{i+1}.mp4")
                print(f"Starting recording for episode {i+1} -> {video_filename}")
                vis.start_recording(filename=video_filename, fps=30, dpi=150)
            # ---

            while not terminated and not truncated:
                # Agent chooses the action using the loaded model (no exploration)
                # Setting evaluate=True ensures deterministic action (mean of the policy)
                action = agent.choose_action(observation, evaluate=True) 
                
                observation_, reward, terminated, truncated, info = env.step(action)
                
                score += reward
                observation = observation_
                step_count += 1
                
                # Update visualization
                try:
                    if vis.is_open: # Check if visualization window is still open
                        vis.update_plot(
                            current_state=info['state_car1_cartesian'],
                            mpc_predicted_states=info.get('predicted_states_car1'),
                            current_state_car2=info['state_car2_cartesian'],
                            reward_targets=info.get('reward_pts'),
                            mpc_predicted_states_car2=info.get('predicted_states_car2')
                        )
                        time.sleep(0.01) # Control the visualization speed
                    else:
                        print("Visualization window closed. Skipping updates.")
                        # If window is closed, no point in continuing the episode vis
                        break 

                except Exception as e:
                    print(f"Visualization failed during step {step_count}: {e}. Continuing run.")

            # --- End of Episode ---
            
            # --- Stop Video Recording ---
            if SAVE_VIDEO and vis.is_recording:
                print(f"Stopping recording for episode {i+1}...")
                vis.stop_recording()
            # ---

            total_scores.append(score)
            print(f'Episode {i+1}: Score {score:.2f}, Steps {step_count}')
            
    finally:
        if vis.is_open:
            vis.close()
            print("Visualization window closed.")

    # --- Final Results ---
    avg_score = np.mean(total_scores) if total_scores else 0
    print("\n--- Evaluation Finished ---")
    print(f"Total Episodes Run: {len(total_scores)}")
    print(f"Average Score over {len(total_scores)} episodes: {avg_score:.2f}")

if __name__ == '__main__':
    if not hasattr(config, 'RL_PARAMS'):
        print("ERROR: 'RL_PARAMS' dictionary not found in config.py. Please add it.")
    else:
        evaluate()
