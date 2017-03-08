"""Creates training and testing datasets out of single csv dataset

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import argparse
import pandas as pd
import csv
from random import shuffle

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('file',
                    type=str,
                    help='Location of input file (csv formatted)')
parser.add_argument('train',
                    type=str,
                    help='Location of output training file (csv formatted)')
parser.add_argument('test',
                    type=str,
                    help='Location of output testing file (csv formatted)')
parser.add_argument('--percentage',
                    type=float,
                    default=0.8,
                    help='Percentage of data to be training between 0 and 1')

args = parser.parse_args()

# Load csv
data = pd.read_csv(args.file)

values = data.values

# randomize and split dataset
shuffle(values)

p = args.percentage
stop_index = int(p * len(values))
training = values[:stop_index]
testing = values[stop_index:]

with open(args.train, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(data.columns)
    writer.writerows(training)

with open(args.test, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(data.columns)
    writer.writerows(testing)
