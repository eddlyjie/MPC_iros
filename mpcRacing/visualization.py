import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    """
    Handles the real-time plotting of the vehicle, its predicted trajectory,
    and the race track.
    """

    def __init__(self, track_data):
        """
        Initializes the visualization plot.

        Args:
            track_data (dict): A dictionary containing the track's centerline
                               and boundaries.
        """
        plt.ion()  # Turn on interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 9))
        
        self.track_center = track_data['center_line']
        self.track_left = track_data['left_bound']
        self.track_right = track_data['right_bound']

        # --- Plot static track elements ---
        self.ax.plot(self.track_center[:, 0], self.track_center[:, 1], 'k--', label='Track Centerline')
        self.ax.plot(self.track_left[:, 0], self.track_left[:, 1], 'k-', label='Track Boundaries')
        self.ax.plot(self.track_right[:, 0], self.track_right[:, 1], 'k-')
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.grid(True)

        # --- MODIFIED: Plot dynamic elements for BOTH cars ---
        # Car 1 (Ego Vehicle) - Blue
        self.vehicle_pos, = self.ax.plot([], [], 'bo', markersize=8, label='Car 1 (Ego)')
        self.mpc_prediction, = self.ax.plot([], [], 'b-', linewidth=1.5, label='Car 1 Prediction')

        # Car 2 (Delayed Vehicle) - Red
        self.vehicle_pos_car2, = self.ax.plot([], [], 'ro', markersize=8, label='Car 2 (Delayed)')
        # Note: We are not visualizing Car 2's prediction for simplicity, but the object is here if needed.
        self.mpc_prediction_car2, = self.ax.plot([], [], 'r-', linewidth=1.5, label='Car 2 Prediction')
        
        self.ax.legend()
        self.fig.canvas.draw()
        plt.show(block=False)

    def update_plot(self, current_state, mpc_predicted_states, current_state_car2=None, mpc_predicted_states_car2=None):
        """
        Updates the plot with the latest vehicle states and predictions.

        Args:
            current_state (np.ndarray): The current state of Car 1.
            mpc_predicted_states (np.ndarray): The predicted trajectory for Car 1.
            current_state_car2 (np.ndarray, optional): The current state of Car 2. Defaults to None.
            mpc_predicted_states_car2 (np.ndarray, optional): The predicted trajectory for Car 2. Defaults to None.
        """
        # --- 1. Update Car 1 (Ego Vehicle) ---
        current_state_flat = np.array(current_state).flatten()
        x, y = current_state_flat[0], current_state_flat[1]
        self.vehicle_pos.set_data([x], [y])

        if mpc_predicted_states is not None:
            pred_x = mpc_predicted_states[0, :]
            pred_y = mpc_predicted_states[1, :]
            self.mpc_prediction.set_data(pred_x, pred_y)

        # --- 2. MODIFIED: Update Car 2 (Delayed Vehicle) ---
        if current_state_car2 is not None:
            current_state_car2_flat = np.array(current_state_car2).flatten()
            x2, y2 = current_state_car2_flat[0], current_state_car2_flat[1]
            self.vehicle_pos_car2.set_data([x2], [y2])
        
        # We are not currently passing the prediction for car 2, but this is how you would plot it
        if mpc_predicted_states_car2 is not None:
            pred_x2 = mpc_predicted_states_car2[0, :]
            pred_y2 = mpc_predicted_states_car2[1, :]
            self.mpc_prediction_car2.set_data(pred_x2, pred_y2)
        else:
            # Clear the line if no prediction is available
            self.mpc_prediction_car2.set_data([], [])

        # --- 3. Update plot limits to follow the ego car (Car 1) ---
        view_range = 40.0
        self.ax.set_xlim(x - view_range, x + view_range)
        self.ax.set_ylim(y - view_range, y + view_range)
        
        # --- 4. Redraw the canvas ---
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
