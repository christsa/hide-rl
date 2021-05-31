import torch

ant = torch.load('HAC.pkl')
point = torch.load('HAC_Point.pkl')

for key in ['layer_1_actor', 'layer_1_critic', 'layer_0_actor', 'layer_0_critic']:
    ant[key] = point[key]

torch.save(ant, 'HAC.pkl')

