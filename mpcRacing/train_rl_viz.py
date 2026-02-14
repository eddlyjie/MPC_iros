import numpy as np
import time
import matplotlib.pyplot as plt
import os
import re # Added for parsing filenames
import keyboard

from environment_rl import RLEnvironment
from sac_agent import SACAgent
from visualization_rl import Visualization
import config

def find_latest_model(chkpt_dir):
    """
    Finds the latest saved model in the checkpoint directory.
    Returns the suffix (e.g., "save10") and the next save number (11).
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
    next_save_num = max_num + 1
    return latest_suffix, next_save_num

def plot_and_save_scores(score_history, avg_history, filename='learning_curve.png'):
    """
    Plots the raw episode scores and the 100-episode moving average, saving to a file.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(1, len(score_history) + 1), score_history, label='Episode Score', alpha=0.6)
    plt.plot(np.arange(1, len(avg_history) + 1), avg_history, label='100-Ep Avg Score', color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Training Learning Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
    print(f"Learning curve saved to {filename}")

def main():
    """
    Main function to run RL training with periodic visualization for debugging.
    """
    print("--- Initializing RL Training Environment ---")

    # --- Configurable Parameters ---
    VISUALIZE_EVERY_N_EPISODES = 50
    N_GAMES = 10000
    SAVE_CHECKPOINT_EVERY_N = 100 # How often to check for a new best model

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
    
    best_score = -np.inf
    score_history = []
    avg_score_history = []
    
    # --- Load models block ---
    chkpt_dir = agent.actor.checkpoint_dir
    latest_suffix, next_save_num = find_latest_model(chkpt_dir)

    if latest_suffix:
        print(f"Found latest model: {latest_suffix}. Attempting to load...")
        try:
            agent.load_models(latest_suffix)
            agent.save_counter = next_save_num
            print(f"... models loaded successfully. Next save will be 'save{next_save_num}'.")
            
            # Try to load score history to resume best_score
            if os.path.exists('score_history.csv'):
                print("... loading score history.")
                # Use .tolist() to convert numpy array back to a standard list
                score_history = np.loadtxt('score_history.csv', delimiter=',').tolist()
                
                # Recalculate best_score and avg_score_history from loaded history
                if len(score_history) > 100:
                    # Recreate the average score history
                    avg_score_history = [np.mean(score_history[max(0, i-100):i]) for i in range(1, len(score_history)+1)]
                    best_score = np.max(avg_score_history)
                    print(f"... Resumed best score: {best_score:.2f} from {len(score_history)} episodes.")
                else:
                    # Not enough history to have a best score yet
                    avg_score_history = [np.mean(score_history[max(0, i-100):i]) for i in range(1, len(score_history)+1)]
                    print(f"... Resumed with {len(score_history)} episodes.")
            
        except Exception as e:
            print(f"... error loading models: {e}. Starting from scratch.")
            score_history = []
            avg_score_history = []
            best_score = -np.inf
            agent.save_counter = 0 # Reset counter
    else:
        print("... no existing models found, starting from scratch.")

    stop = False
    print("\n--- Starting Training Loop ---")
    try:
        for i in range(N_GAMES):
            if stop:
                break
            observation,_ = env.reset()
            terminated = False
            truncated = False
            score = 0
            step_count = 0
            
            visualize_this_episode = ((i) % VISUALIZE_EVERY_N_EPISODES == 0)
            if visualize_this_episode:
                print(f"--- Visualizing Episode {i} ---")

            while not terminated and not truncated:
                # when key ] is pressed, stop training
                if keyboard.is_pressed(']'):
                    print("']' key pressed. Stopping training after this episode.")
                    stop = True
                    break

                action = agent.choose_action(observation)
                
                observation_, reward, terminated, truncated, info = env.step(action)

                agent.remember(observation, action, reward, observation_, terminated)
                agent.learn()
                
                score += reward
                observation = observation_
                step_count += 1
                
                if visualize_this_episode:
                    try:
                        vis.update_plot(
                            current_state=info['state_car1_cartesian'],
                            mpc_predicted_states=info.get('predicted_states_car1'),
                            current_state_car2=info['state_car2_cartesian'],
                            reward_targets=info.get('reward_pts'),
                            mpc_predicted_states_car2=info.get('predicted_states_car2')
                        )
                        time.sleep(0.001)
                    except Exception as e:
                        print(f"Visualization failed: {e}")
                        print("Turning off visualization for this episode.")
                        visualize_this_episode = False
            
            # --- End of Episode ---
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])
            avg_score_history.append(avg_score)

            # --- Save logic: check every N episodes ---
            if (i + 1) % SAVE_CHECKPOINT_EVERY_N == 0 and i > 0:
                if avg_score > best_score:
                    best_score = avg_score
                    print(f"--- Episode {i+1}: New best avg score: {best_score:.2f}. Saving model... ---")
                    agent.save_models() # This will use agent.save_counter (e.g., save0, save1...)
                else:
                    print(f"--- Episode {i+1}: Avg score {avg_score:.2f} (Best: {best_score:.2f}). Not saving. ---")

            print(f'Episode {i}: Score {score:.2f}, Avg Score (100) {avg_score:.2f}, Steps {step_count}')
            
            if visualize_this_episode:
                print(f"--- Visualization for Episode {i} complete ---")

        print("\n--- Training Finished ---")
        vis.close() # Close the plot window at the end
    finally:
        agent.save_models()  # Ensure models are saved at the end
        print("Final models saved.")
        # if visualization is not closed yet
        if vis.is_open:
            vis.close()
            print("Visualization window closed.")


    # --- Save final logs and plot ---
    print("Saving final score history and plot.")
    try:
        # Save score history to CSV
        scores_data = np.array(score_history)
        np.savetxt('score_history.csv', scores_data, delimiter=',')
        
        # Save plot
        plot_and_save_scores(score_history, avg_score_history)
    except Exception as e:
        print(f"Error saving logs/plot: {e}")


if __name__ == '__main__':
    if not hasattr(config, 'RL_PARAMS'):
        print("ERROR: 'RL_PARAMS' dictionary not found in config.py. Please add it.")
    else:
        main()
