import os
import argparse
parser = argparse.ArgumentParser(description='ProGAN')
parser.add_argument('--config', '-c', metavar='CONFIG', default=None, help='The yaml config file path')
parser.add_argument('--batch', '-b', metavar='BATCH', default=1024, help='The maximum batch size')
parser.add_argument('--gpu', type=str, default='0', metavar='N', help='the gpu used for training, separated by comma and no space left(default: 0)')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.gpu

from trainer import Trainer
from model import *
print(args)
trainer = Trainer(args.config, max_batch_size=int(args.batch))
trainer.train()

