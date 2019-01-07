#%%
import numpy as np

timesteps = 100
input_features = 32
output_features = 64

input = np.random.random((timesteps, input_features))

state_t = np.zeros((output_features,))
