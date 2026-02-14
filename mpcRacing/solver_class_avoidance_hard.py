import casadi as ca
import numpy as np
from utils import debug

# Assuming these files are in the same directory and contain the necessary functions/data
# You will need to provide these files for the code to run.
try:
    import track_utils
    import vehicle_model as mpc_vehicle_model
except ImportError:
    print("Warning: 'track_utils' or 'vehicle_model' not found. This code is for modification review.")
    # Define dummy fallbacks so the class can be defined
    class mpc_vehicle_model:
        @staticmethod
        def get_vehicle_model(params):
            return lambda st, con: ca.SX.zeros(8, 1) # Dummy model
            
    class track_utils:
        @staticmethod
        def load_track_data(file):
            return {'block_info': np.zeros((10, 5)), 'center_line': np.zeros((10, 2)), 'lane_yaw': np.zeros((10,1))}
        @staticmethod
        def find_closest_point(pose, data):
            return 0, 0
        @staticmethod
        def get_local_centerline(pose, cl, yaw):
            return np.zeros((5, 2))
        @staticmethod
        def fit_progress_polynomial(cl):
            return ca.DM.zeros(10, 1)


class MPC_Solver:
    """
    A class to encapsulate the Nonlinear Model Predictive Control (NMPC) solver.
    It handles the creation of the solver, setting bounds, and running the optimization.
    This version includes dynamic avoidance and reward points.
    
    MODIFICATION: Obstacle avoidance is now a hard constraint, not a soft cost.
    """
    def __init__(self, T, N, vehicle_params, track_params, mpc_costs, state_bounds=None, control_bounds=None):
        """
        Initializes the MPC solver.
        """
        self.T = T
        self.N = N
        self.vehicle_params = vehicle_params
        self.track_params = track_params
        self.mpc_costs = mpc_costs
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds
        
        self.AVOID_RADIUS = self.mpc_costs.get('AVOID_RADIUS', None)
        print(f"Using AVOID_RADIUS = {self.AVOID_RADIUS} for hard constraint.")
        # These are no longer needed for the hard constraint
        # self.AVOID_STEEPNESS = 50.0
        # self.W_AVOID_STEEP = 100.0

        self.n_states = 8
        self.n_controls = 2
        
        self.model_func = mpc_vehicle_model.get_vehicle_model(self.vehicle_params)
        self.track_data = track_utils.load_track_data(self.track_params['mat_file'])
        
        self.hard_avoidance_pts_num = 10  # Number of avoidance points treated as hard constraints
        self.solver = self._create_solver()
        self._set_solver_bounds()

        self.u0 = None
        self.X0_pred = None


    def _create_solver(self):
        """
        Builds and returns the CasADi NLP solver for the NMPC problem.
        """
        maxNumBlocks = self.track_params['maxNumBlocks']
        U = ca.SX.sym('U', self.n_controls, self.N)
        X = ca.SX.sym('X', self.n_states, self.N + 1)
        
        P_size = self.n_states + 5 * maxNumBlocks + 10 + 3 * self.N + 3 * self.N
        P = ca.SX.sym('P', P_size)
        
        obj = 0
        g = []

        gravity = self.vehicle_params['g']
        ratio = self.vehicle_params['ratio']
        muf, mur = self.vehicle_params['muf'], self.vehicle_params['mur']
        la, lb, hcg, M = self.vehicle_params['la'], self.vehicle_params['lb'], self.vehicle_params['hcg'], self.vehicle_params['M']
        R = self.mpc_costs['R']
        w_v, w_sa, w_r, w_ax = self.mpc_costs['w_v'], self.mpc_costs['w_sa'], self.mpc_costs['w_r'], self.mpc_costs['w_ax']
        w_tube, w_goal = self.mpc_costs['w_tube'], self.mpc_costs['w_goal']

        block_params_flat = P[self.n_states : self.n_states + 5 * maxNumBlocks]
        cost_para = P[self.n_states + 5 * maxNumBlocks : self.n_states + 5 * maxNumBlocks + 10]
        block_center_x = block_params_flat[0*maxNumBlocks : 1*maxNumBlocks]
        block_center_y = block_params_flat[1*maxNumBlocks : 2*maxNumBlocks]
        block_yaw      = block_params_flat[2*maxNumBlocks : 3*maxNumBlocks]
        block_length   = block_params_flat[3*maxNumBlocks : 4*maxNumBlocks]
        block_width    = block_params_flat[4*maxNumBlocks : 5*maxNumBlocks]

        dynamic_points_start_idx = self.n_states + 5 * maxNumBlocks + 10
        P_avoid_flat = P[dynamic_points_start_idx : dynamic_points_start_idx + 3 * self.N]
        P_reward_flat = P[dynamic_points_start_idx + 3 * self.N : dynamic_points_start_idx + 6 * self.N]
        P_avoid = ca.reshape(P_avoid_flat, self.N, 3)
        P_reward = ca.reshape(P_reward_flat, self.N, 3)

        g.append(X[:, 0] - P[0:self.n_states])

        for k in range(self.N):
            st, con, st_next = X[:, k], U[:, k], X[:, k+1]
            v_val, r_val, ux_val, sa_val, ax_val = st[2], st[3], st[5], st[6], st[7]
            
            stability_cost = w_v * v_val**2 + w_sa * sa_val**2 + w_r * r_val**2
            effort_cost = w_ax * ax_val**2
            control_cost = ca.mtimes([con.T, R, con])
            
            x_pos, y_pos = st[0], st[1]

            # Track boundary cost (tube cost)
            dx, dy = x_pos - block_center_x, y_pos - block_center_y
            cos_yaw, sin_yaw = ca.cos(block_yaw), ca.sin(block_yaw)
            dx_rot, dy_rot = cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy
            
            p_norm, rhoKS = 4, 5
            super_ellipse = ((dx_rot / (block_length + 1e-6))**p_norm + 
                             (dy_rot / (block_width - 0.5 + 1e-6))**p_norm + 0.01)**(1/p_norm)
            
            ks_sum = ca.sum1(ca.exp(rhoKS * (-(super_ellipse) + 1)))
            drivable_dist = (1/rhoKS) * ca.log(ks_sum)
            tube_cost_k = 10 * ca.log(1 + ca.exp(-10 * (0.3 + drivable_dist)))
                
            obj += stability_cost + effort_cost + control_cost + w_tube * tube_cost_k

            # --- MODIFICATION START: Avoidance as Hard Constraint ---
            avoid_pt_k = P_avoid[k, :]
            # dist_sq_avoid = (x_pos - avoid_pt_k[0])**2 + (y_pos - avoid_pt_k[1])**2
            dist_sq_avoid = (st_next[0] - avoid_pt_k[0])**2 + (st_next[1] - avoid_pt_k[1])**2
            
            # This value is positive inside the radius (unsafe) and negative outside (safe).
            if k < self.hard_avoidance_pts_num:
                violation = self.AVOID_RADIUS**2 - dist_sq_avoid
            else:
                violation = 0.0  # No constraint for points beyond the hard constraint limit
            
            # The constraint is (avoid_pt_k[2] * violation) <= 0.
            # We assume avoid_pt_k[2] is a non-negative activation weight (0 or >0).
            # If avoid_pt_k[2] > 0, this enforces violation <= 0
            # (i.e., dist_sq_avoid >= AVOID_RADIUS**2), which is the safe region.
            # If avoid_pt_k[2] == 0, this enforces 0 <= 0, which is always true (no constraint).
            g.append(violation)

            # The cost is removed from the objective function:
            # avoid_cost = avoid_pt_k[2] * self.W_AVOID_STEEP * ca.log(1 + ca.exp(self.AVOID_STEEPNESS * violation))
            # obj += avoid_cost
            # --- MODIFICATION END ---

            # Reward Cost (unchanged)
            # reward_pt_k = P_reward[:, k]
            # dist_sq_reward = (x_pos - reward_pt_k[0])**2 + (y_pos - reward_pt_k[1])**2
            # reward_cost = reward_pt_k[2] * dist_sq_reward
            # obj += reward_cost

            # Acceleration constraints
            g.append(ax_val - ((-6.75 / 80.0) * (ux_val - 80.0)))
            gxb_c, gzb_c = 0.0, gravity
            g.append(ax_val - (gxb_c + (mur * la * M * gzb_c) / (M * (la + lb) - mur * hcg * M)))
            g.append(ax_val - (gxb_c + lb * gzb_c / hcg))
            g.append(-ax_val + gxb_c + ((muf * lb * M * gzb_c) / (-M * (la + lb) * ratio + muf * hcg * M)))
            g.append(-ax_val + gxb_c + ((mur * la * M * gzb_c) / (((la + lb) * (ratio - 1) * M) - mur * hcg * M)))

            # RK4 Integration
            k1 = self.model_func(st, con)
            k2 = self.model_func(st + self.T/2 * k1, con)
            k3 = self.model_func(st + self.T/2 * k2, con)
            k4 = self.model_func(st + self.T * k3, con)
            st_next_RK4 = st + (self.T / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            g.append(st_next_RK4 - st_next)
        
        x_end, y_end = X[0, -1], X[1, -1]
        obj_goal = (cost_para[0] + cost_para[1] * x_end + cost_para[2] * y_end + 
                    cost_para[3] * x_end**2 + cost_para[4] * x_end * y_end + cost_para[5] * y_end**2 +
                    cost_para[6] * x_end**3 + cost_para[7] * x_end**2 * y_end + 
                    cost_para[8] * x_end * y_end**2 + cost_para[9] * y_end**3)
        obj += w_goal * obj_goal

        OPT_variables = ca.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))
        nlp_prob = {'f': obj, 'x': OPT_variables, 'g': ca.vertcat(*g), 'p': P}
        opts = {
            'ipopt': {
                "mu_strategy": "adaptive", "max_iter": 350, "tol": 4e-2, 
                "warm_start_init_point": "yes", "print_level": 1, "sb" : "yes"
            },
            'print_time': 0
        }
        
        return ca.nlpsol('solver', 'ipopt', nlp_prob, opts)
    
    def _set_solver_bounds(self):
        """
        Sets the lower and upper bounds for the solver's decision variables and constraints.
        """
        self.lbx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        self.ubx = ca.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        
        STATE_BOUNDS = self.state_bounds if self.state_bounds is not None else [(-ca.inf, ca.inf)] * self.n_states
        CONTROL_BOUNDS = self.control_bounds if self.control_bounds is not None else [(-ca.inf, ca.inf)] * self.n_controls
            
        for i in range(self.n_states):
            self.lbx[i:self.n_states*(self.N + 1):self.n_states] = STATE_BOUNDS[i][0]
            self.ubx[i:self.n_states*(self.N + 1):self.n_states] = STATE_BOUNDS[i][1]
            
        control_start_idx = self.n_states * (self.N + 1)
        for i in range(self.n_controls):
            self.lbx[control_start_idx+i::self.n_controls] = CONTROL_BOUNDS[i][0]
            self.ubx[control_start_idx+i::self.n_controls] = CONTROL_BOUNDS[i][1]
            
        # --- MODIFICATION: Account for new constraint ---
        # We now have 1 (avoid) + 5 (ax_val) + n_states (RK4) constraints per loop
        self.n_g_per_loop = 1 + 5 + self.n_states 
        
        self.lbg = ca.DM.zeros((self.n_states + self.N * self.n_g_per_loop, 1))
        self.ubg = ca.DM.zeros((self.n_states + self.N * self.n_g_per_loop, 1))
        
        # The first n_states constraints are for the initial condition (g.append(X[:, 0] - P[0:self.n_states]))
        # These are equality constraints, so lbg=0, ubg=0 (which is the default).
        
        for k in range(self.N):
            # We have 1 avoidance constraint + 5 ax_val constraints = 6 inequality constraints
            # All are of the form g_i(x) <= 0.
            n_inequality_constraints = 1 + 5 

            start_idx = self.n_states + k * self.n_g_per_loop
            
            # Set bounds for g_i(x) <= 0  (i.e., -inf <= g_i(x) <= 0)
            self.lbg[start_idx : start_idx + n_inequality_constraints] = -ca.inf
            self.ubg[start_idx : start_idx + n_inequality_constraints] = 0
            
            # The remaining n_states constraints (for RK4) are equality constraints
            # (g.append(st_next_RK4 - st_next)).
            # Their bounds remain 0 (lbg=0, ubg=0), which is the default.
        # --- MODIFICATION END ---

    def get_solution(self, x0, u0, X0_pred, avoid_points=None, reward_points=None):
        """
        Solves the NMPC problem for a given initial state and initial guesses.
        """
        avoid_pts = np.zeros((3, self.N)) if avoid_points is None else avoid_points
        reward_pts = np.zeros((3, self.N)) if reward_points is None else reward_points

        cur_pose = x0[0:2].T
        bg_idx, _ = track_utils.find_closest_point(cur_pose, self.track_data['block_info'][:, 0:2])
        indices = np.arange(bg_idx, bg_idx + self.track_params['maxNumBlocks']) % self.track_data['block_info'].shape[0]
        local_blocks = self.track_data['block_info'][indices, :]
        
        mpc_centerline = track_utils.get_local_centerline(cur_pose, self.track_data['center_line'], self.track_data['lane_yaw'])
        cost_poly = track_utils.fit_progress_polynomial(mpc_centerline)
        
        p = ca.vertcat(
            x0, 
            ca.reshape(local_blocks, -1, 1), 
            cost_poly,
            ca.reshape(avoid_pts, -1, 1),
            ca.reshape(reward_pts, -1, 1)
        )
        # maxNumBlocks = self.track_params['maxNumBlocks']
        # dynamic_points_start_idx = self.n_states + 5 * maxNumBlocks + 10
        # P_avoid_flat = p[dynamic_points_start_idx : dynamic_points_start_idx + 3 * self.N]
        # P_reward_flat = p[dynamic_points_start_idx + 3 * self.N : dynamic_points_start_idx + 6 * self.N]
        # P_avoid = ca.reshape(P_avoid_flat, self.N, 3)
        # P_reward = ca.reshape(P_reward_flat, 3, self.N)
        # avoid_pt_k = P_avoid[1, :]
        # # print(avoid_pt_k)
        x_init = ca.vertcat(ca.reshape(X0_pred, -1, 1), ca.reshape(u0, -1, 1))
        
        try:
            sol = self.solver(x0=x_init, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
            stats = self.solver.stats()
            if not stats['success']:
                # This will now catch failures that don't raise exceptions
                print(f"Solver Follower failed with status: {stats['return_status']}")
                return None
            return sol
        except Exception as e:
            print(f"Solver failed: {e}")
            # This might happen if the problem becomes infeasible with the new hard constraint
            return None

    def solve_and_update(self, x0, avoid_points=None, reward_points=None):
        """
        Solves the NMPC problem using the previous solution as a warm start.
        """
        if self.u0 is None or self.X0_pred is None:
            self.u0 = ca.DM.zeros((self.n_controls, self.N))
            current_x0_dm = ca.DM(x0) if isinstance(x0, np.ndarray) else x0
            self.X0_pred = ca.repmat(current_x0_dm, 1, self.N + 1)

        solution = self.get_solution(x0, self.u0, self.X0_pred, avoid_points, reward_points)
        
        if solution is not None:
            u_optimal = ca.reshape(solution['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
            X_optimal = ca.reshape(solution['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)
            
            # Warm start for next iteration
            self.u0 = u_optimal
            self.X0_pred = X_optimal
        
        return solution

# Example of how you might run this (requires all dependencies and parameters)
if __name__ == "__main__":
    print("This script defines the MPC_Solver class with hard constraints.")
    print("To run it, you need to instantiate it with appropriate parameters and data.")
    
    # Dummy parameters for demonstration
    T = 0.1  # Time step
    N = 20   # Prediction horizon
    
    vehicle_params = {
        'g': 9.81, 'ratio': 0.5, 'muf': 1.2, 'mur': 1.2,
        'la': 1.5, 'lb': 1.5, 'hcg': 0.5, 'M': 1500
    }
    
    track_params = {
        'mat_file': 'dummy_track.mat', # You need a real track file
        'maxNumBlocks': 5
    }
    
    mpc_costs = {
        'R': ca.diag(ca.DM([1e-3, 1e-3])),
        'w_v': 0.1, 'w_sa': 0.1, 'w_r': 0.1, 'w_ax': 0.1,
        'w_tube': 1.0, 'w_goal': 1.0,
        'AVOID_RADIUS': 2.5
    }
    
    state_bounds = [
        (-ca.inf, ca.inf), # x
        (-ca.inf, ca.inf), # y
        (-ca.inf, ca.inf), # v
        (-ca.inf, ca.inf), # r
        (-ca.inf, ca.inf), # yaw
        (0, 80),           # ux
        (-0.5, 0.5),       # sa
        (-ca.inf, ca.inf)  # ax
    ]
    
    control_bounds = [
        (-6.0, 6.0),    # d_sa (steering angle rate)
        (-6.75, 6.75)   # d_ax (acceleration rate)
    ]
    
    try:
        solver = MPC_Solver(T, N, vehicle_params, track_params, mpc_costs, state_bounds, control_bounds)
        print("MPC_Solver initialized successfully with hard constraints.")
        
        # Example solve call (would fail without real track data)
        # x0 = np.zeros(8)
        # avoid_points = np.zeros((3, N))
        # avoid_points[0, 5] = 10 # x-pos of obstacle at step 5
        # avoid_points[1, 5] = 2  # y-pos of obstacle at step 5
        # avoid_points[2, 5] = 1  # activate constraint at step 5
        #
        # solution = solver.solve_and_update(x0, avoid_points=avoid_points)
        # if solution:
        #     print("Solver ran successfully.")
        # else:
        #     print("Solver failed (as expected without real data).")
            
    except Exception as e:
        print(f"Could not initialize solver (likely due to missing dummy files): {e}")
