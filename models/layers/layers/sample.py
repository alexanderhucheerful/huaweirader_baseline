import torch
import numpy as np

random_flip = np.random.random_sample((3,2,1))
print(random_flip)
true_token = (random_flip < 1)
print(true_token)