import argparse
import os
import re
from itertools import product

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=list, default=[42])
    parser.add_argument('--lr', type=list, default=[1e-5])
    parser.add_argument('--bsz', type=list, default=[12])
    parser.add_argument('--device', type=int, default=0)
    # parser.add_argument('-subtask', type=int, default=2)
    # parser.add_argument('--wd', type=list, default=[0.01], help='weight decay')
    args = parser.parse_args()
    seed_list, lr_list, bsz_list, device = args.seed, args.lr, args.bsz, args.device
    # subtask = args.subtask
    f1 = {}
    for seed, lr, batch_size in product(seed_list, lr_list, bsz_list):
        command = 'CUDA_VISIBLE_DEVICES={} python run.py --seed {} --batch_size {} --lr {}'.format(device, seed, batch_size, lr)
        output = os.popen(command)
        for line in output:
            r = re.findall(r'Best f1: \d.+\d', line)
            if len(r) == 0:
                continue
            else:
                key = 'seed: ' + str(seed) + ' lr: ' + str(lr) + ' bsz: ' + str(batch_size)
                value = float(r[0].split(':')[1]) * 100
                f1[key] = value

    print(f1)
    with open('./tune-result', 'w') as f:
        f.write(str(f1))








