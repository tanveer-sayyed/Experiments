"""

    data parallelism: models are replicated into different devices (GPU)
                      and trained on batches of data.    
    for master node:
        python torch/main.py --nodes 4 --gpus 8 --rank 0
    
    for subsequent node(s):
        python torch/main.py --nodes 4 --gpus 8 --rank 1
        python torch/main.py --nodes 4 --gpus 8 --rank 2
        python torch/main.py --nodes 4 --gpus 8 --rank 3
        
"""

import os
import torch.multiprocessing as mp
from argparse import ArgumentParser

from train import train

def main():
    parser = ArgumentParser()
    parser.add_argument('-n', '--nodes',
                        default=1, type=int)
    parser.add_argument('-g', '--gpus',
                        default=1, type=int, help='number of gpus per node')
    parser.add_argument('-r', '--rank',
                        default=0, type=int, help='ranking within the nodes')
    parser.add_argument('-e','--epochs',
                        default=25, type=int, help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = "123.123.123.123"
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(train, nprocs=args.gpus, args=(args,))
