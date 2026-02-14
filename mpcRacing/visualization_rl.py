import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter # Added for video saving
from utils import debug

class Visualization:
    """
    Handles the real-time plotting of the vehicle, its predicted trajectory,
    the race track, and the RL agent's sparse reward targets.
    Can also save the animation to a video file.
    """

    def __init__(self, track_data):
        """
        Initializes the visualization plot.

        Args:
            track_data (dict): A dictionary containing the track's centerline
                               and boundaries.
        """
        self.is_open = True
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        
        # --- Video Recording Setup ---
        self.video_writer = None
        self.is_recording = False
        # --- End Video Recording Setup ---
        
        self.track_center = track_data['center_line']
        self.track_left = track_data['left_bound']
        self.track_right = track_data['right_bound']

        # --- Plot static track elements ---
        self.ax.plot(self.track_center[:, 0], self.track_center[:, 1], 'k--', label='Track Centerline')
        self.ax.plot(self.track_left[:, 0], self.track_left[:, 1], 'k-', label='Track Boundaries')
        self.ax.plot(self.track_right[:, 0], self.track_right[:, 1], 'k-')
        
        # --- Initialize dynamic plot elements ---
        
        # Car 1 (Ego Vehicle)
        self.vehicle_pos, = self.ax.plot([], [], 'bo', markersize=3, label="Ego Vehicle (RL)")
        self.mpc_prediction, = self.ax.plot([], [], 'b-', alpha=0.5, label="Ego Prediction")
        
        # Car 2 (Other Vehicle)
        self.vehicle_pos_car2, = self.ax.plot([], [], 'ro', markersize=3, label="Other Vehicle")
        self.mpc_prediction_car2, = self.ax.plot([], [], 'r-', alpha=0.5, label="Other Prediction")
        
        # --- NEW: RL Reward Targets ---
        self.rl_reward_points, = self.ax.plot([], [], 'g*', markersize=10, label='RL Reward Targets')

        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.grid(True)
        self.ax.legend()
        plt.show(block=False) # Show the plot without blocking

    def start_recording(self, filename='simulation_output.mp4', fps=30, dpi=150):
        """Initializes the video writer to start recording frames."""
        if self.is_recording:
            debug("Already recording. Please stop recording first.")
            return

        metadata = dict(title='Vehicle Simulation', artist='Matplotlib', comment='MPC Trajectory')
        self.video_writer = FFMpegWriter(fps=fps, metadata=metadata)
        try:
            # Set up the writer with the figure, filename, and DPI
            self.video_writer.setup(self.fig, filename, dpi=dpi)
            self.is_recording = True
            debug(f"Recording started. Output file: {filename}")
        except Exception as e:
            self.video_writer = None
            self.is_recording = False
            debug(f"Error initializing FFMpegWriter. Check if ffmpeg is installed. Error: {e}")

    def stop_recording(self):
        """Stops the video recording and finalizes the file."""
        if self.is_recording and self.video_writer:
            self.video_writer.finish()
            self.is_recording = False
            self.video_writer = None
            debug("Recording stopped and file finalized.")
        elif not self.is_recording:
            debug("Not currently recording.")


    def update_plot(self, current_state, mpc_predicted_states, current_state_car2=None, mpc_predicted_states_car2=None, reward_targets=None):
        """
        Updates the positions of all dynamic elements on the plot.
        """
        
        # --- 1. Update Car 1 (Ego Vehicle) ---
        current_state_flat = np.array(current_state).flatten()
        x, y = current_state_flat[0], current_state_flat[1]
        debug(f"Updating Car 1 Position to: x={x}, y={y}")
        self.vehicle_pos.set_data([x], [y])
        debug(f"mpc_predicted_states: {mpc_predicted_states}")
        if mpc_predicted_states is not None:
            pred_x = mpc_predicted_states[0, :]
            pred_y = mpc_predicted_states[1, :]
            self.mpc_prediction.set_data(pred_x, pred_y)
        else:
            self.mpc_prediction.set_data([], []) # Clear if no prediction

        # --- 2. Update Car 2 (Other Vehicle) ---
        x2, y2 = x, y # Default to car 1's position for view centering if car 2 is None
        if current_state_car2 is not None:
            current_state_car2_flat = np.array(current_state_car2).flatten()
            x2, y2 = current_state_car2_flat[0], current_state_car2_flat[1]
            self.vehicle_pos_car2.set_data([x2], [y2])

        if mpc_predicted_states_car2 is not None:
            pred_x2 = mpc_predicted_states_car2[0, :]
            pred_y2 = mpc_predicted_states_car2[1, :]
            self.mpc_prediction_car2.set_data(pred_x2, pred_y2)
        
        # --- 3. Update RL Reward Targets ---
        if reward_targets is not None:
            # Filter for points where weight > 0
            active_targets = reward_targets[reward_targets[:, 2] > 0]
            if active_targets.shape[0] > 0:
                self.rl_reward_points.set_data(active_targets[:, 0], active_targets[:, 1])
            else:
                self.rl_reward_points.set_data([], [])
        else:
            self.rl_reward_points.set_data([], [])

        # set xlim and ylim to keep the view consistent
        view_range = 200.0
        self.ax.set_xlim((x+x2)/2 - view_range/2, (x+x2)/2 + view_range/2)
        self.ax.set_ylim((y+y2)/2 - view_range/2, (y+y2)/2 + view_range/2)

        # --- 4. Force GUI update & 5. Video Recording ---
        try:
            self.fig.canvas.draw_idle()
            
            if self.is_recording and self.video_writer:
                self.video_writer.grab_frame()

            plt.pause(0.001)  # Small pause to allow GUI to update

        except Exception as e:
            # Catches errors when the Matplotlib GUI window is closed
            debug(f"Visualization error ({type(e).__name__}) detected.")
            if self.is_recording:
                debug("Stopping video recording gracefully.")
                self.stop_recording()
            self.is_open = False # Mark as closed

    def close(self):
        """Closes the plot window and finalizes video."""
        
        # --- NEW: Stop recording if active ---
        if self.is_recording:
            self.stop_recording()
            
        if self.is_open:
            plt.close(self.fig)
            self.is_open = False

