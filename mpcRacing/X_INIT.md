# x0 = [x, y, v, r, psi, ux, sa, ax]

change the train_rl_viz.py config.py and sac_agent.py

for thunder:
X0_INIT = np.array([349.0, -577.0, 0.0, 0.0, -3.0227, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([355.0, -577.0, 0.0, 0.0, -3.0227, 15.0, 0.0, 0.0]).reshape(-1, 1)


track2:
X0_INIT = np.array([-15, -350, 0.0, 0.0, -2, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([-15, -342, 0.0, 0.0, -2, 15.0, 0.0, 0.0]).reshape(-1, 1)

track8:
X0_INIT = np.array([143, 370, 0.0, 0.0, 0.5, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([141, 365, 0.0, 0.0, 0.5, 15.0, 0.0, 0.0]).reshape(-1, 1)