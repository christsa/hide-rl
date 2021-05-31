import csv,json
from collections import defaultdict
import os

destination = 'Directory with results.'

def filter_duplicates(data):
    new_data = []
    for i, (steps, rate) in enumerate(data):
        if i == 0 or i == len(data)-1:
            new_data.append((steps, rate))
        else:
            if data[i-1][1] == rate:
                continue
            new_data.append((steps, rate))
    return new_data


def process(exp_name="PAPERAntMazeRandom", real_exp_name="RelHAC", seed=1, exp_num=1):
    source = os.path.join('models/', exp_name, '1', 'progress.csv')
    data = []
    with open(source, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = -1
        for row in csv_reader:
            line_count +=1
            if line_count == 0:
                continue
            else:
                data.append((row['total_steps_taken'], row['test/success_rate']))
    # data = filter_duplicates(data)

    target_dir = os.path.join(destination, real_exp_name, 'seed%d'%seed)
    os.makedirs(target_dir, exist_ok=True)
    json.dump({
        'Seed': seed,
        'Alg': real_exp_name,
        'Experiment': exp_name,
    }, open(os.path.join(target_dir, "params.json"), 'w+'), sort_keys=True, indent=4)
    with open(os.path.join(target_dir, "progress.csv"), 'w+') as f:
        print("Total steps,Success Rate", file=f)
        for step, rate in data:
            print("{},{}".format(step, rate), file=f)

if __name__ == "__main__":
    process("PAPERAntMazeRandom", real_exp_name="RelHAC", seed=1, exp_num=1)
    process("PAPERAntMazeRandom2", real_exp_name="RelHAC", seed=2, exp_num=1)
    process("PAPERAntMazeRandom3", real_exp_name="RelHAC", seed=3, exp_num=1)
    process("PAPERAntMazeRandom4", real_exp_name="RelHAC", seed=4, exp_num=1)
    process("PAPERAntMazeRandom5", real_exp_name="RelHAC", seed=5, exp_num=1)

    process("PAPERAntMazeEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN", real_exp_name="HiDe", seed=1, exp_num=1)
    process("PAPERAntMazeEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN2", real_exp_name="HiDe", seed=2, exp_num=1)
    process("PAPERAntMazeEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN3", real_exp_name="HiDe", seed=3, exp_num=1)
    process("PAPERAntMazeEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN4", real_exp_name="HiDe", seed=4, exp_num=1)
    process("PAPERAntMazeEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN5", real_exp_name="HiDe", seed=5, exp_num=1)

    process("PAPERAntMazeRandomEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN", real_exp_name="HiDe-R", seed=1, exp_num=1)
    process("PAPERAntMazeRandomEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN2", real_exp_name="HiDe-R", seed=2, exp_num=1)
    process("PAPERAntMazeRandomEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN3", real_exp_name="HiDe-R", seed=3, exp_num=1)
    process("PAPERAntMazeRandomEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN4", real_exp_name="HiDe-R", seed=4, exp_num=1)
    process("PAPERAntMazeRandomEasyMVProp2Conv3Her2BlurMoreNoiseQPenalizeMaskingNoAttentionCovFasterDQN5", real_exp_name="HiDe-R", seed=5, exp_num=1)