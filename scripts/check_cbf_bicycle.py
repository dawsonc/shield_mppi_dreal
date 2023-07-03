"""Verify a CBF with simplified kinematic bicycle car dynamics using dReal.

The state of the car is q = (x, y, theta, v), where (x, y) is the position,
theta is the heading angle, and v is the velocity. The control input is
the steering angle delta and the acceleration a. The dynamics are given by

    x' = v cos(theta)
    y' = v sin(theta)
    theta' = v tan(delta) / L
    v' = a

where L is the length of the car.

The CBF is given by h(q) = w^2 - y^2, where w is half the width of the track.

We define one discrete time step by integrating the continuous-time dynamics for
multiple substeps with a forward Euler integration scheme.
"""
import time
from itertools import product

import dreal as dr
import numpy as np
from tqdm import tqdm


def check_feasibility(state, tolerance=1e-3):
    # Define constants
    L = 1.0  # length of car
    w = 2.0  # half-width of track
    dt = 0.1  # substep size
    N_substeps = 4  # number of substeps per time step
    delta_limit = np.pi / 4  # maximum steering angle
    accel_limit = 3.0  # maximum acceleration

    # Define variables
    ## State variables (one initial state + N_substeps future states)
    xs = [dr.Variable(f"x_{i}") for i in range(N_substeps + 1)]
    ys = [dr.Variable(f"y_{i}") for i in range(N_substeps + 1)]
    thetas = [dr.Variable(f"theta_{i}") for i in range(N_substeps + 1)]
    vs = [dr.Variable(f"v_{i}") for i in range(N_substeps + 1)]
    ## Control variables (one control input for all substeps)
    # deltas = [dr.Variable(f"delta_{i}") for i in range(N_substeps)]
    # accels = [dr.Variable(f"accel_{i}") for i in range(N_substeps)]
    delta = dr.Variable("delta")
    accel = dr.Variable("accel")
    ## Initial and next CBF values
    h_now = dr.Variable("h_now")
    h_next = dr.Variable("h_next")
    ## CBF constant
    cbf_alpha = dr.Variable("cbf_alpha")

    # Define constraints
    constraints = []
    ## Initial state
    constraints.append(xs[0] == state[0])
    constraints.append(ys[0] == state[1])
    constraints.append(thetas[0] == state[2])
    constraints.append(vs[0] == state[3])

    ## Dynamics
    for i in range(N_substeps):
        constraints.append(xs[i + 1] == xs[i] + dt * vs[i] * dr.cos(thetas[i]))
        constraints.append(ys[i + 1] == ys[i] + dt * vs[i] * dr.sin(thetas[i]))
        constraints.append(thetas[i + 1] == thetas[i] + dt * vs[i] * dr.tan(delta) / L)
        constraints.append(vs[i + 1] == vs[i] + dt * accel)

    ## Control limits
    # for i in range(N_substeps):
    #     constraints.append(deltas[i] >= -delta_limit)
    #     constraints.append(deltas[i] <= delta_limit)
    #     constraints.append(accels[i] >= -accel_limit)
    #     constraints.append(accels[i] <= accel_limit)
    constraints.append(delta >= -delta_limit)
    constraints.append(delta <= delta_limit)
    constraints.append(accel >= -accel_limit)
    constraints.append(accel <= accel_limit)

    ## CBF
    ### Initial CBF value
    constraints.append(h_now == w**2 - ys[0] ** 2)
    ### Next CBF value
    constraints.append(h_next == w**2 - ys[-1] ** 2)
    ### CBF constant limits
    constraints.append(cbf_alpha >= 1e-3)
    constraints.append(cbf_alpha <= 1.0 - 1e-3)
    ### CBF constraint
    constraints.append(h_next - h_now >= -cbf_alpha * h_now)

    # Define the problem and solve
    problem = dr.And(*constraints)
    result = dr.CheckSatisfiability(problem, tolerance)

    return result


if __name__ == "__main__":
    # Define the domain to search over
    grid_spacing = 0.05
    y_min, y_max = 0.0, 2.0  # reflection symmetry about y = 0
    theta_min, theta_max = -np.pi / 4.0, np.pi / 4.0
    v_min, v_max = 5.0, 10.0
    y = np.arange(y_min, y_max + grid_spacing, grid_spacing)
    theta = np.arange(theta_min, theta_max + grid_spacing, grid_spacing)
    v = np.arange(v_min, v_max + grid_spacing, grid_spacing)
    x = [0.0]  # translation invariant in x, so only need to check one value

    # Check all of the states and compute the feasibility fraction
    num_states = 0
    num_feasible = 0
    expected_total = len(x) * len(y) * len(theta) * len(v)
    pbar = tqdm(product(x, y, theta, v), total=expected_total)
    for state in pbar:
        result = check_feasibility(state, tolerance=grid_spacing)
        num_states += 1
        if result:
            num_feasible += 1

        pbar.set_description(f"{num_feasible / num_states * 100.0:.2f}% feasible")

    print(
        f"Checked {num_states} states; {num_feasible / num_states * 100.0:.2f}% feasible"
    )
