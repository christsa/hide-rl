import torch

point = torch.load('HAC.pkl')
ant = torch.load('HAC_Ant.pkl')

for key in ['layer_1_actor', 'layer_1_critic', 'layer_0_actor', 'layer_0_critic']:
    point[key] = ant[key]

torch.save(point, 'HAC.pkl')

