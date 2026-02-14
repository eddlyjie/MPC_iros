"""
Utility functions for processing and handling race track data.
This version is enhanced with Frenet-Cartesian coordinate conversion
and handles the one-time processing of the track's arc length,
curvature, and boundary deviations.
"""
import numpy as np
import scipy.io as sio
import os
import casadi as ca  # --- ADDED IMPORT ---

def load_and_process_track(mat_file):
    """
    Loads track data, calculates the reference arc length, curvature, and
    boundary interpolators if not already done, and caches the processed
    data for future use.

    Args:
        mat_file (str): Path to the original .mat track file.

    Returns:
        dict: A dictionary containing the processed track data, including
              's_ref', 'curvature_interp', 'd_left_interp', 'd_right_interp',
              or None if the file cannot be loaded.
    """
    processed_track_path = mat_file.replace('.mat', '_processed.npy')

    if os.path.exists(processed_track_path):
        print(f"INFO: Loading pre-processed track data from {processed_track_path}")
        try:
            track_data = np.load(processed_track_path, allow_pickle=True).item()
            # --- NEW: Check if loaded data is complete ---
            required_keys = ['s_ref', 'curvature_interp', 'd_left_interp', 'd_right_interp']
            if all(key in track_data for key in required_keys):
                print("INFO: Pre-processed data is complete and valid.")
                return track_data
            else:
                print("WARNING: Pre-processed data is outdated or incomplete. Reprocessing...")
            # --- END OF NEW CHECK ---
        except Exception as e:
            print(f"WARNING: Could not load processed track file '{processed_track_path}'. Error: {e}. Reprocessing...")

    print(f"INFO: No valid pre-processed track data found. Processing from {mat_file}...")
    raw_track_data = _load_mat_track_data(mat_file)
    if raw_track_data is None:
        return None # Error message is handled in the loading function

    # --- 1. Calculate the cumulative arc length 's_ref' ---
    center_line = raw_track_data['center_line']
    # --- FIX: Changed axis=0 to axis=1 ---
    diffs = np.linalg.norm(np.diff(center_line, axis=0), axis=1)
    # --- END OF FIX ---
    s_ref = np.concatenate(([0], np.cumsum(diffs)))
    raw_track_data['s_ref'] = s_ref
    s = s_ref # Use 's' for clarity in calculations

    # --- Check for length consistency ---
    if len(s_ref) != len(raw_track_data['lane_yaw']):
         print(f"WARNING: Centerline length ({len(s_ref)}) does not match yaw length ({len(raw_track_data['lane_yaw'])}). This may cause issues.")
         # Attempt to fix if off-by-one, which can happen with 'fined' data
         min_len = min(len(s_ref), len(raw_track_data['lane_yaw']))
         s = s_ref[:min_len]
         raw_track_data['lane_yaw'] = raw_track_data['lane_yaw'][:min_len]
         raw_track_data['center_line'] = raw_track_data['center_line'][:min_len]
         raw_track_data['left_bound'] = raw_track_data['left_bound'][:min_len]
         raw_track_data['right_bound'] = raw_track_data['right_bound'][:min_len]
         print(f"INFO: Arrays truncated to minimum length: {min_len}")
    
    print("INFO: Calculating curvature and boundary interpolators...")

    # --- 2. Calculate Curvature (kappa) ---
    yaw = raw_track_data['lane_yaw'].flatten()
    # Unwrap angles to avoid jumps from +pi to -pi
    unwrapped_yaw = np.unwrap(yaw)
    # Calculate curvature kappa = d(yaw) / ds
    kappa = np.gradient(unwrapped_yaw, s)
    # Fix potential gradient artifact at the loop closure
    kappa[-1] = kappa[-2] 
    raw_track_data['kappa'] = kappa
    # Create CasADi interpolator
    raw_track_data['curvature_interp'] = ca.interpolant('curvature_interp', 'linear', [s], kappa)

    # --- 3. Calculate Lateral Deviations (d_left, d_right) ---
    # We assume 'left_bound[i]' and 'right_bound[i]' correspond to 'center_line[i]'
    cl = raw_track_data['center_line']
    lb = raw_track_data['left_bound']
    rb = raw_track_data['right_bound']
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)

    # Calculate lateral deviation 'd' by projecting the (dx, dy) vector
    # onto the centerline's normal vector: N = (-sin(yaw), cos(yaw))
    # d = (P_bound - P_center) . N
    
    # Left boundary
    dx_l = lb[:, 0] - cl[:, 0]
    dy_l = lb[:, 1] - cl[:, 1]
    d_left = dy_l * cos_y - dx_l * sin_y

    # Right boundary
    dx_r = rb[:, 0] - cl[:, 0]
    dy_r = rb[:, 1] - cl[:, 1]
    d_right = dy_r * cos_y - dx_r * sin_y

    raw_track_data['d_left'] = d_left
    raw_track_data['d_right'] = d_right
    
    # Create CasADi interpolators
    raw_track_data['d_left_interp'] = ca.interpolant('d_left_interp', 'linear', [s], d_left)
    raw_track_data['d_right_interp'] = ca.interpolant('d_right_interp', 'linear', [s], d_right)
    
    print("INFO: Track processing complete.")

    # --- 4. Save the newly processed data ---
    try:
        np.save(processed_track_path, raw_track_data)
        print(f"INFO: Saved pre-processed track data to {processed_track_path}")
    except Exception as e:
        print(f"WARNING: Could not save processed track data. Error: {e}")
        
    return raw_track_data

def _load_mat_track_data(mat_file):
    """Internal function to load track data from a .mat file."""
    try:
        track_mat = sio.loadmat(mat_file)
        # --- Ensure all data is .T ---
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
    except KeyError as e:
        print(f"ERROR: .mat file is missing a required key: {e}")
        return None
    except Exception as e:
        print(f"ERROR: An unexpected error occurred loading {mat_file}: {e}")
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
    num_points = 350
    loop_num = len(center_line)
    
    indices = np.arange(cline_begin_idx, cline_begin_idx + num_points) % loop_num
    
    mpc_track_info = np.zeros((num_points, 4))
    mpc_track_info[:, 0:2] = center_line[indices, :]
    mpc_track_info[:, 2] = lane_yaw[indices, 0]
    
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
        for d in track_width_candidate:
            x_fit[count] = base_x - d * np.sin(base_yaw)
            y_fit[count] = base_y + d * np.cos(base_yaw)
            s_fit[count] = base_s
            count += 1
            
    # Wrap s_fit for tracks that loop
    s_fit[s_fit > total_arc_length / 2] -= total_arc_length
    
    coeffs_x = np.polyfit(s_fit, x_fit, 3)
    coeffs_y = np.polyfit(s_fit, y_fint, 3) # <-- Potential typo, changing to s_fit
    coeffs_y = np.polyfit(s_fit, y_fit, 3)
    
    return np.concatenate((coeffs_x, coeffs_y))

# --- Functions Required for RL Environment ---

def cartesian_to_frenet(x, y, yaw, track_data):
    """
    Converts a single Cartesian state (x, y, yaw) to Frenet coordinates (s, d, phi_e).
    """
    centerline = track_data['center_line']
    s_ref = track_data['s_ref']
    lane_yaw = track_data['lane_yaw']
    
    # Find the closest point on the centerline
    idx, _ = find_closest_point(np.array([x, y]), centerline)
    
    # Get reference values from the centerline
    s_val = s_ref[idx]
    ref_x, ref_y = centerline[idx]
    ref_yaw = lane_yaw[idx]
    
    # Calculate lateral distance 'd'
    # Robust method using sign(dot_product) * magnitude
    dx = x - ref_x
    dy = y - ref_y
    d = np.sign(dy * np.cos(ref_yaw) - dx * np.sin(ref_yaw)) * np.sqrt(dx**2 + dy**2)
    
    # Calculate heading error 'phi_e'
    phi_e = yaw - ref_yaw
    # Normalize angle to [-pi, pi]
    phi_e = (phi_e + np.pi) % (2 * np.pi) - np.pi

    return s_val, d, phi_e

def frenet_to_cartesian(s, d, track_data):
    """
    Converts a single Frenet point (s, d) to Cartesian coordinates (x, y).
    """
    centerline = track_data['center_line']
    s_ref = track_data['s_ref']
    lane_yaw = track_data['lane_yaw']
    
    # Find the index corresponding to the given 's' value
    # np.interp requires monotonically increasing s_ref, which we have from cumsum
    
    # wrap around if s exceeds the track length
    s_total = s_ref[-1]
    s = s % s_total
    
    # Interpolate to find the index
    # We use s_ref and its corresponding indices (0 to len-1)
    indices_array = np.arange(len(s_ref))
    idx = np.interp(s, s_ref, indices_array)
    
    # Get the integer index and interpolation fraction
    idx_int = int(idx)
    frac = idx - idx_int
    
    # Ensure we wrap around correctly for the last point
    next_idx = (idx_int + 1) % len(centerline)
    
    # Interpolate to find the reference point on the centerline
    ref_x = (1 - frac) * centerline[idx_int, 0] + frac * centerline[next_idx, 0]
    ref_y = (1 - frac) * centerline[idx_int, 1] + frac * centerline[next_idx, 1]
    
    # Interpolate yaw (handling wraparound)
    yaw_start = lane_yaw[idx_int].item()
    yaw_end = lane_yaw[next_idx].item()
    delta_yaw = yaw_end - yaw_start
    if delta_yaw > np.pi:
        delta_yaw -= 2 * np.pi
    elif delta_yaw < -np.pi:
        delta_yaw += 2 * np.pi
    ref_yaw = yaw_start + frac * delta_yaw


    # Calculate Cartesian coordinates by moving 'd' distance along the normal vector
    x = ref_x - d * np.sin(ref_yaw)
    y = ref_y + d * np.cos(ref_yaw)
    
    return x, y

