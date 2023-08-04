import numpy as np

trial = np.random.normal(loc=0, scale=1, size=(3,3))

print(trial)

trial[1][1] = 0
print(trial)