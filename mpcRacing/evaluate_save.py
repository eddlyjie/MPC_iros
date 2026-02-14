import numpy as np
import time
import os
import re # Added for parsing filenames
import argparse # For command-line arguments
import scipy.io as sio # <-- ADDED: For saving .mat files

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
    SAVE_MATFILE = True # <-- ADDED: Whether to save episode data
    MATFILE_DIR = "eval_data" # <-- ADDED: Directory to save .mat files

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

    # --- 5. Create Output Directories ---
    if SAVE_VIDEO:
        os.makedirs(VIDEO_DIR, exist_ok=True)
        print(f"Videos will be saved to '{VIDEO_DIR}' directory.")
    if SAVE_MATFILE: # <-- ADDED
        os.makedirs(MATFILE_DIR, exist_ok=True)
        print(f"MAT-files will be saved to '{MATFILE_DIR}' directory.")


    total_scores = []
    
    print(f"\n--- Starting Evaluation for {N_EVAL_EPISODES} Episodes ---")
    try:
        for i in range(N_EVAL_EPISODES):
            observation, _ = env.reset()
            terminated = False
            truncated = False
            score = 0
            step_count = 0

            # <-- ADDED: Lists to store data for this episode
            episode_states_car1 = []
            episode_states_car2 = []
            episode_active_targets = []
            episode_solve_times_car1 = []
            episode_s_car1 = []
            episode_s_car2 = []
            episode_predictions_car1 = []
            
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
                
                # --- ADDED: Get data for storage and visualization ---
                current_state_car1 = info['state_car1_cartesian']
                current_state_car2 = info['state_car2_cartesian']
                all_reward_targets = info.get('reward_pts') # Get all targets for visualization
                solve_time_car1 = info.get('solve_time_car1', 0)
                s_car1 = info.get('s_car1', 0)
                s_car2 = info.get('s_car2', 0)
                current_prediction_car1 = info.get('predicted_states_car1')
                
                if SAVE_MATFILE:
                    # Store vehicle states
                    episode_states_car1.append(current_state_car1)
                    episode_states_car2.append(current_state_car2)
                    episode_solve_times_car1.append(solve_time_car1)
                    episode_s_car1.append(s_car1)
                    episode_s_car2.append(s_car2)
                    episode_predictions_car1.append(current_prediction_car1)
                    
                    # Process and store active reward targets
                    # We create a (4, 3) array for this step, filled with NaNs.
                    # This ensures a consistent shape (N_steps, 4, 3) for the final array.
                    step_targets = np.full((4, 3), np.nan) 
                    
                    if all_reward_targets is not None:
                        # Filter for points where weight > 0 (as inferred from snippet)
                        active_targets = all_reward_targets[all_reward_targets[:, 2] > 0]
                        num_active = active_targets.shape[0]
                        
                        if num_active > 0:
                            # Copy up to 4 active targets into our step_targets array
                            num_to_copy = min(num_active, 4)
                            step_targets[0:num_to_copy, :] = active_targets[0:num_to_copy, :]
                    
                    # Append the (4, 3) array to the episode list
                    episode_active_targets.append(step_targets)
                # ---
                
                # Update visualization
                try:
                    if vis.is_open: # Check if visualization window is still open
                        vis.update_plot(
                            current_state=current_state_car1, # Use variable
                            mpc_predicted_states=info.get('predicted_states_car1'),
                            current_state_car2=current_state_car2, # Use variable
                            reward_targets=all_reward_targets, # Pass all targets to viz
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

            # --- ADDED: Save Episode Data to MAT-file ---
            if SAVE_MATFILE:
                try:
                    mat_filename = os.path.join(MATFILE_DIR, f"model_{MODEL_SUFFIX}_episode_{i+1}_data.mat")
                    
                    # Convert lists to numpy arrays for robust saving
                    # vehicle_states will be (N_steps, state_dim)
                    # reward_targets_active will be (N_steps, 4, 3)
                    data_to_save = {
                        'vehicle1_states': np.array(episode_states_car1),
                        'vehicle2_states': np.array(episode_states_car2),
                        'reward_targets_active': np.array(episode_active_targets),
                        'solve_times_car1': np.array(episode_solve_times_car1),
                        's_car1': np.array(episode_s_car1),
                        's_car2': np.array(episode_s_car2),
                        'predicted_states_car1': np.array(episode_predictions_car1)
                    }
                    
                    sio.savemat(mat_filename, data_to_save)
                    print(f"Successfully saved episode data to {mat_filename}")
                
                except Exception as e:
                    print(f"Error saving MAT-file for episode {i+1}: {e}")
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

