# x0 = [x, y, v, r, psi, ux, sa, ax]

change the config.py for model map and initial state

for thunder:
X0_INIT = np.array([349.0, -577.0, 0.0, 0.0, -3.0227, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([355.0, -577.0, 0.0, 0.0, -3.0227, 15.0, 0.0, 0.0]).reshape(-1, 1)


track2:
X0_INIT = np.array([-15, -350, 0.0, 0.0, -2, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([-15, -342, 0.0, 0.0, -2, 15.0, 0.0, 0.0]).reshape(-1, 1)

track8:
X0_INIT = np.array([143, 370, 0.0, 0.0, 0.5, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([141, 365, 0.0, 0.0, 0.5, 15.0, 0.0, 0.0]).reshape(-1, 1)

track1:
#01
X0_INIT = np.array([174, 50, 0.0, 0.0, 0.75, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([170, 48, 0.0, 0.0, 0.75, 15.0, 0.0, 0.0]).reshape(-1, 1)


track3:
X0_INIT = np.array([8, 2, 0.0, 0.0, 0.1, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([1, 1, 0.0, 0.0, 0.1, 15.0, 0.0, 0.0]).reshape(-1, 1)


track6:
X0_INIT = np.array([-839, -32, 0.0, 0.0, -1.85, 15.0, 0.0, 0.0]).reshape(-1, 1)
X0_INIT_2 = np.array([-832, -24, 0.0, 0.0, -1.85, 15.0, 0.0, 0.0]).reshape(-1, 1)
