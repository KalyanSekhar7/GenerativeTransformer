import torch

B,T,C = 4,8,2

# B: Batches
# T: Timesteps ( previous 8 timesteps, or 7 or like that)
# C: Channels of the embedding layers
