"""Match coefficient 100 without global mining to separate strength from search."""
import argparse
import torch
from run import fit

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,default=0);parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args();torch.set_num_threads(1)
    for i,seed in enumerate(range(130,140)):
        if i%args.shards==args.shard:fit(seed,'sigreg',100.)
