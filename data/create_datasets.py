""" File to create training and testing datasets
    Author: Zander Blasingame"""

import argparse
from random import shuffle

parser = argparse.ArgumentParser()

parser.add_argument('file',
                    type=str,
                    help='The location of the input file')

parser.add_argument('train_file',
                    type=str,
                    help='Training file location')

parser.add_argument('test_file',
                    type=str,
                    help='Test file location')

parser.add_argument('--p',
                    type=float,
                    help='Percentage of data to be training (0.0 - 1.0)')

args = parser.parse_args()

lines = []

with open(args.file, 'r') as f:
    for line in f.readlines():
        lines.append(line)

shuffle(lines)

stop_index = int(args.p * len(lines))

with open(args.train_file, 'w') as f:
    f.writelines(lines[:-stop_index])

with open(args.test_file, 'w') as f:
    f.writelines(lines[-stop_index:])
