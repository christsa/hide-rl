import argparse

"""
Below are training options user can specify in command line.

Options Include:

1. Retrain boolean ("--retrain")
- If included, actor and critic neural network parameters are reset

2. Testing boolean ("--test")
- If included, agent only uses greedy policy without noise.  No changes are made to policy and neural networks.
- If not included, periods of training are by default interleaved with periods of testing to evaluate progress.

3. Show boolean ("--show")
- If included, training will be visualized

4. Train Only boolean ("--train_only")
- If included, agent will be solely in training mode and will not interleave periods of training and testing

5. Verbosity boolean ("--verbose")
- If included, summary of each transition will be printed

6. All Trans boolean ("--all_trans")
- If included, all transitions including (i) hindsight action, (ii) subgoal penalty, (iii) preliminary HER, and (iv) final HER transitions will be printed.  Use below options to print out specific types of transitions.

7. Hindsight Action trans boolean ("hind_action")
- If included, prints hindsight actions transitions for each level

8. Subgoal Penalty trans ("penalty")
- If included, prints the subgoal penalty transitions

9. Preliminary HER trans ("prelim_HER")
-If included, prints the preliminary HER transitions (i.e., with TBD reward and goal components)

10.  HER trans ("HER")
- If included, prints the final HER transitions for each level

11. Show Q-values ("--Q_values")
- Show Q-values for each action by each level

"""

def parse_options():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Include to reset policy'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Include to fix current policy'
    )

    parser.add_argument(
        '--show',
        action='store_true',
        help='Include to visualize training'
    )

    parser.add_argument(
        '--train_only',
        action='store_true',
        help='Include to use training mode only'
    )

    parser.add_argument(
        '--no_target_net',
        action='store_true',
        help='Does not use target networks.'
    )

    parser.add_argument(
        '--exp_name',
        type=str,
        default="",
        help='Experiment name.'
    )

    parser.add_argument(
        '--variant',
        type=str,
        default="",
        help='Variant name.'
    )

    parser.add_argument(
        '--exp_num',
        type=int,
        default=0,
        help='Experiment name.'
    )

    parser.add_argument(
        '--num_Qs',
        type=int,
        default=2,
        help='Number of critic networks.'
    )

    parser.add_argument(
        '--oracle',
        action='store_true',
        help='Use oracle instead of the first layer.'
    )

    parser.add_argument(
        '--relative_subgoals',
        action='store_true',
        help='Instead of absolute goals use relative subgoals.'
    )

    parser.add_argument(
        '--no_middle_level',
        action='store_true',
        help='No middle level.'
    )

    parser.add_argument(
        '--mask_global_info',
        action='store_true',
        help='Mask unnecessary observations for the middle layer.'
    )

    parser.add_argument(
        '--radam',
        action='store_true',
        help='Use Radam instead of Adam.'
    )

    parser.add_argument(
        '--vpn',
        action='store_true',
        help='Use Value Propagation Network.'
    )

    parser.add_argument(
        '--no_vpn_weights',
        action='store_true',
        help='Use deterministic vpn module.'
    )

    parser.add_argument(
        "--save_video", 
        action='store_true', 
        help="Saves video. Can't be used with show at the same time."
    )

    parser.add_argument(
        "--featurize_image", 
        action='store_true', 
        help="Uses simple image of the environement instead of the rendered scene."
    )

    parser.add_argument(
        "--always_penalize", 
        action='store_true', 
        help="On the highest level always penalize for not completed goal."
    )

    parser.add_argument(
        "--Q_penalize", 
        action='store_true', 
        help="On the highest level always penalize when the lower level's Q function think the goal is reachable."
    )

    parser.add_argument(
        "--vpn_double_conv", 
        action='store_true', 
        help="VPN layer has double convolution."
    )

    parser.add_argument(
        "--vpn_dqn", 
        action='store_true', 
        help="VPN's V function is multiplied by pooled wall probs."
    )

    parser.add_argument(
        '--vpn_masking',
        action='store_true',
        help='Destill sigma from the images and apply appropriate Gaussian mask.'
    )

    parser.add_argument(
        '--covariance',
        action='store_true',
        help='Apply extra conv layer after the VPN propagation.'
    )

    parser.add_argument(
        "--gaussian_attention", 
        action='store_true', 
        help="Use Gaussian kernel instead of fixed attention."
    )

    parser.add_argument(
        "--no_attention", 
        action='store_true', 
        help="Don't use attention."
    )

    parser.add_argument(
        "--sigma_overlay", 
        action='store_true', 
        help="When saving video, show attention."
    )

    parser.add_argument(
        "--noisy", 
        action='store_true', 
        help="Add noise to the V map before predicting covariance matrix."
    )

    parser.add_argument(
        "--reconstruction", 
        action='store_true', 
        help="Add reconstruction loss to the planning layer."
    )

    parser.add_argument(
        "--high_penalty", 
        action='store_true', 
        help="Threshold wall probabilities."
    )

    FLAGS, unparsed = parser.parse_known_args()


    return FLAGS
