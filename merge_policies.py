import torch
import pickle
import numpy as np

her_policy = 'policy_best.pkl'
ball_policy = 'ball_policy.pkl'
output_policy = 'arm_policy.pkl'

cuda = torch.device('cuda')

w = pickle.load(open(her_policy, 'rb'))

def t(arr, swap=False):
    if swap:
        assert len(arr.shape) == 2
        arr = arr.T
    return torch.tensor(arr, dtype=torch.float32, device=cuda)

# Normalizers
obs_mean = t(w.main.o_stats.mean.eval())
obs_std = t(w.main.o_stats.std.eval())
goal_mean = t(w.main.g_stats.mean.eval())
goal_std = t(w.main.g_stats.std.eval())

# Main networks
weights = {v.name:v for v in w.main_vars if 'pi' in v.name}
linear1_w = t(weights['ddpg/main/pi/_0/kernel:0'].eval(), swap=True)
linear1_b = t(weights['ddpg/main/pi/_0/bias:0'].eval())
linear2_w = t(weights['ddpg/main/pi/_1/kernel:0'].eval(), swap=True)
linear2_b = t(weights['ddpg/main/pi/_1/bias:0'].eval())
linear3_w = t(weights['ddpg/main/pi/_2/kernel:0'].eval(), swap=True)
linear3_b = t(weights['ddpg/main/pi/_2/bias:0'].eval())
linear4_w = t(weights['ddpg/main/pi/_3/kernel:0'].eval(), swap=True)
linear4_b = t(weights['ddpg/main/pi/_3/bias:0'].eval())

new_weights = torch.load(ball_policy)
new_weights['layer_0_actor']['infer_net'] = {
    'linear1.weight' : linear1_w,
    'linear1.bias' : linear1_b,
    'linear2.weight' : linear2_w,
    'linear2.bias' : linear2_b,
    'linear3.weight' : linear3_w,
    'linear3.bias' : linear3_b,
    'linear4.weight' : linear4_w,
    'linear4.bias' : linear4_b,
    'obs_mean': obs_mean,
    'obs_std': obs_std,
    'goal_mean': goal_mean,
    'goal_std': goal_std,
}

torch.save(new_weights, output_policy)
