"""
Utility functions for processing and handling race track data.
"""
import numpy as np
import scipy.io as sio

def load_track_data(mat_file):
    """Loads track data from a .mat file."""
    try:
        track_mat = sio.loadmat(mat_file)
        return {
            "block_info": track_mat["block_info"].T,
            "center_line": track_mat["fined_center_line"].T,
            "lane_yaw": track_mat["fined_yawang"].T,
            "left_bound": track_mat["fined_left_bound"].T,
            "right_bound": track_mat["fined_right_bound"].T
        }
    except FileNotFoundError:
        print(f"ERROR: Track data file not found at '{mat_file}'")
        return None

def find_closest_point(point, points_array):
    """Finds the index and distance of the closest point in points_array to point."""
    diff = points_array - point
    distances = np.linalg.norm(diff, axis=1)
    idx = np.argmin(distances)
    return idx, distances[idx]

def get_local_centerline(init_pos, center_line, lane_yaw):
    """Extracts a local segment of the track centerline near the vehicle."""
    cline_begin_idx, _ = find_closest_point(init_pos, center_line)
    num_points = 361
    loop_num = center_line.shape[0]
    
    indices = np.arange(cline_begin_idx, cline_begin_idx + num_points) % loop_num
    
    mpc_track_info = np.zeros((num_points, 4))
    mpc_track_info[:, 0:2] = center_line[indices, :]
    mpc_track_info[:, 2] = lane_yaw[indices, 0]
    
    # Calculate cumulative arc length 's'
    # FIX: Reshape the prepended point to be 2D to match the dimensions of mpc_track_info
    prepend_point = center_line[(cline_begin_idx - 1) % loop_num, :].reshape(1, -1)
    dist = np.linalg.norm(np.diff(mpc_track_info[:, 0:2], axis=0, prepend=prepend_point), axis=1)
    mpc_track_info[:, 3] = np.cumsum(dist)
    
    return mpc_track_info

def fit_progress_polynomial(mpc_track_info):
    """Fits a 3rd order polynomial to estimate track progress."""
    track_width_candidate = np.linspace(-4.5, 4.5, 25)
    num_points = len(track_width_candidate) * mpc_track_info.shape[0]
    
    x_fit, y_fit, s_fit = np.zeros(num_points), np.zeros(num_points), np.zeros(num_points)
    total_arc_length = mpc_track_info[-1, 3]
    
    count = 0
    for i in range(mpc_track_info.shape[0]):
        base_x, base_y, base_yaw, base_s = mpc_track_info[i, :]
        for width in track_width_candidate:
            x_fit[count] = base_x + width * np.cos(base_yaw + np.pi / 2)
            y_fit[count] = base_y + width * np.sin(base_yaw + np.pi / 2)
            s_fit[count] = total_arc_length - base_s
            count += 1
            
    A = np.vstack([np.ones_like(x_fit), x_fit, y_fit, x_fit**2, x_fit*y_fit, y_fit**2,
                   x_fit**3, x_fit**2*y_fit, x_fit*y_fit**2, y_fit**3]).T
    params, _, _, _ = np.linalg.lstsq(A, s_fit, rcond=None)
    return params

