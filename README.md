#  Hierarchical Decompositional Reinforcement Learning (HiDe)
This repository contains the code to implement the *Learning Functionally Decomposed Hierarchies for Continuous Navigation Tasks (HiDe)* algorithm. 
The paper is available at [IEEE RA-L](https://ieeexplore.ieee.org/document/9357915))

We are happy to answer any questions. Feel free to file an issue.

## Quick summary 
Quick summary and videos can be found on the project's website: https://sites.google.com/view/hide-rl

## Requirments
* Python 3.7
* PyTorch. 
* NumPy
* MuJoCo - to obtain a license please follow: https://www.roboti.us/license.html
* MuJoCo Py
* scitkit-video - for rendering videos
* cv2 - for rendering videos with sigma overlay

## Installation

Once MuJoCo is installed, the rest of the required packages can be installed by running the following pip command in your virtual environment:
```
pip install -r requirements.txt
```

## Citing
Please use the following citation:
```
@ARTICLE{christen2021hide,
  author={{Christen}, Sammy and {Jendele}, Lukas and {Aksan}, Emre and {Hilliges}, Otmar},
  journal={IEEE Robotics and Automation Letters},
  title={Learning Functionally Decomposed Hierarchies for Continuous Control Tasks With Path Planning},
  year={2021},
  volume={6},
  number={2},
  pages={3623-3630},
  doi={10.1109/LRA.2021.3060403}}
}
```

## Reproduce experiments 
Each experiment is defined by a variant file. The variant names follow this pattern: `{AgentName}{MazeName}`. In order to change a maze, just replace the maze name.
To visualize live, add `--show` flag. To render a video, add `--save_video` to the command line.

### Show pretrained models
This repository contains pretrained models for all our experiments.

#### Experiment 1 - Simple Maze:
HiDe:

```
# Training
python initialize_HAC.py --test --variant AntHiroMaze --exp_name HiDe_Simple_Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Backward
python initialize_HAC.py --test --variant AntHiroMazeBackwards --exp_name HiDe_Simple_Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show

# Flipped
python initialize_HAC.py --test --variant AntHiroMazeFlipped --exp_name HiDe_Simple_Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show

```

#### Experiment 2 - Complex Maze:
HiDe:

```
# Ant Forward
python initialize_HAC.py --test --variant AntWall --exp_name HiDe_Complex_Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show

# Ant Backward
python initialize_HAC.py --test --variant AntWallBackwards --exp_name HiDe_Complex_Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show

# Ant Flipped
python initialize_HAC.py --test --variant AntWallFlipped --exp_name HiDe_Complex_Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Ant Random
python initialize_HAC.py --test --variant AntMaze --exp_name HiDe_Complex_Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show

```

```
# Ball Forward
python initialize_HAC.py --test --variant DMPointWall --exp_name HiDe_Complex_Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Ball Backward
python initialize_HAC.py --test --variant DMPointWallBackwards --exp_name HiDe_Complex_Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Ball Flipped
python initialize_HAC.py --test --variant DMPointWallFlipped --exp_name HiDe_Complex_Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Ball Random
python initialize_HAC.py --test --variant DMPointMaze --exp_name HiDe_Complex_Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show

```

#### Experiment 3 - Transfer of agents for a complex environment:

```
# Forward A->B
python initialize_HAC.py --test --variant AntWallFlipped --exp_name HiDe_Complex_Ant2Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Backward A->B
python initialize_HAC.py --test --variant AntWallBackwards --exp_name HiDe_Complex_Ant2Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Flipped A->B
python initialize_HAC.py --test --variant AntWallFlipped --exp_name HiDe_Complex_Ant2Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Random A->B
python initialize_HAC.py --test --variant AntMaze --exp_name HiDe_Complex_Ant2Ball --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Forward B->A
python initialize_HAC.py --test --variant DMPointWall --exp_name HiDe_Complex_Ball2Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Backward B->A
python initialize_HAC.py --test --variant DMPointWallBackwards --exp_name HiDe_Complex_Ball2Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Flipped B->A
python initialize_HAC.py --test --variant DMPointWallFlipped --exp_name HiDe_Complex_Ball2Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --no_middle_level --torch --show

# Random B->A
python initialize_HAC.py --test --variant DMPointMaze --exp_name HiDe_Complex_Ball2Ant --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show
```

### Retrain a model
The commands are the same as above, just replace `--test` with `--retrain`. Changing the experiment name is recommended as the command would overwrite the pretrained experiments.

#### Experiment 1 - Simple Maze:
HiDe:

```
python initialize_HAC.py --retrain --variant AntHiroMaze --exp_name HiDe_Simple_Ant_new --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn  --no_middle_level --torch --show
```


#### Experiment 2 - Complex Maze:
HiDe:

```
# Ant
python initialize_HAC.py --retrain --variant AntWall --exp_name HiDe_complex_Ant_new --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --high_penalty  --no_middle_level --torch --show

# Ball
python initialize_HAC.py --retrain --variant DMPointWall --exp_name HiDe_complex_Ant_new --num_Qs 1 --exp_num 1 --relative_subgoals --mask_global_info --vpn --featurize_image --vpn_double_conv --Q_penalize --vpn_masking --no_attention --covariance --vpn_dqn --high_penalty --no_middle_level --torch --show
```

We found out that we get better generalization results when trained with `--high_penalty` flag.

#### Experiment 3 - Transfer of agents:
We provide the pretrained models both for Ant and Ball. Both were trained using RelHAC without the planning layer. 
Ant model has been taken from experiment 1, where as Point model was trained on Empty arena. You can find it in models/ directory together with scripts that transfer the weights.


## Credits
We would like to thank:
* [HAC](https://github.com/andrew-j-levy/Hierarchical-Actor-Critc-HAC-) Our codebase is based on theirs.
* [HIRO](https://github.com/tensorflow/models/tree/master/research/efficient-hrl) Our environments are based on their code.
* [DeepMind Control Suite](https://github.com/deepmind/dm_control/tree/master/dm_control/suite) Our Ball agent is inspired by their PointMass one.