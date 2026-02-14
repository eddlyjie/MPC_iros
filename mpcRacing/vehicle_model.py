"""
Defines the dynamic model of the racing vehicle using CasADi.
"""
import casadi as ca

def get_vehicle_model(vehicle_params):
    """
    Creates and returns a CasADi function for the vehicle dynamics.

    Args:
        vehicle_params (dict): Dictionary of vehicle parameters.

    Returns:
        casadi.Function: A function 'f(states, controls) -> rhs' that computes
                         the state derivatives.
    """
    # Unpack vehicle parameters for clarity
    la, lb, hcg, Izz = vehicle_params['la'], vehicle_params['lb'], vehicle_params['hcg'], vehicle_params['Izz']
    muf, mur, M, gravity = vehicle_params['muf'], vehicle_params['mur'], vehicle_params['M'], vehicle_params['g']
    Caf, Car, ratio = vehicle_params['Caf'], vehicle_params['Car'], vehicle_params['ratio']

    # Define symbolic state and control variables
    x, y, v, r, psi, ux, sa, ax = [ca.SX.sym(s) for s in ['x', 'y', 'v', 'r', 'psi', 'ux', 'sa', 'ax']]
    states = ca.vertcat(x, y, v, r, psi, ux, sa, ax)
    
    sr, jx = ca.SX.sym('sr'), ca.SX.sym('jx')
    controls = ca.vertcat(sr, jx)

    # Simplified vehicle dynamics (assuming flat ground: phi=0, theta=0)
    gxb = 0.0
    gyb = 0.0
    gzb = gravity

    # Tire slip angles
    alphaf = ca.atan((v + la * r) / (ux + 1e-6)) - sa
    alphar = ca.atan((v - lb * r) / (ux + 1e-6))

    # Longitudinal force distribution
    axf = ax * ratio
    axr = ax - axf
    FXF = axf * M
    FXR = axr * M

    # Vertical forces (load transfer)
    FZR = (la * M * gzb + hcg * M * (ax - gxb) - M * hcg * v * r) / (la + lb)
    FZF = M * gzb - FZR

    # Tire friction model (simplified Pacejka-like)
    fyf_expr = 10.0 * (1 - (FXF / (muf * FZF + 1e-6))**2 - 0.03)
    fyr_expr = 10.0 * (1 - (FXR / (mur * FZR + 1e-6))**2 - 0.03)
    
    fyf_expr_clamped = ca.fmax(-50, ca.fmin(50, fyf_expr))
    fyr_expr_clamped = ca.fmax(-50, ca.fmin(50, fyr_expr))
    
    FYFMAX = ca.sqrt(ca.log(1 + ca.exp(fyf_expr_clamped)) / 10.0) * (muf * FZF) + 0.1
    FYRMAX = ca.sqrt(ca.log(1 + ca.exp(fyr_expr_clamped)) / 10.0) * (mur * FZR) + 0.1
    
    FYF = -2 * FYFMAX * (1 / (1 + ca.exp(-(2 * Caf / (FYFMAX + 1e-6)) * alphaf)) - 0.5)
    FYR = -2 * FYRMAX * (1 / (1 + ca.exp(-(2 * Car / (FYRMAX + 1e-6)) * alphar)) - 0.5)
    
    # Equations of motion
    x_dot = ux * ca.cos(psi) - v * ca.sin(psi)
    y_dot = ux * ca.sin(psi) + v * ca.cos(psi)
    v_dot = ((FYF * ca.cos(sa) + FYR + FXF * ca.sin(sa)) / M - r * ux - gyb)
    r_dot = ((la * (FYF * ca.cos(sa) + FXF * ca.sin(sa)) - lb * FYR) / Izz)
    psi_dot = r
    ux_dot = (ax + v * r - FYF * ca.sin(sa) / M)
    sa_dot = sr
    ax_dot = jx
    
    rhs = ca.vertcat(x_dot, y_dot, v_dot, r_dot, psi_dot, ux_dot, sa_dot, ax_dot)
    
    return ca.Function('f', [states, controls], [rhs])
