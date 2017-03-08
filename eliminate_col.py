"""Eliminates columns not in the whitelist of a file

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import argparse
import pandas as pd
import csv

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('file',
                    type=str,
                    help='Location of input file (csv formatted)')
parser.add_argument('whitelist',
                    type=str,
                    help='Location of whitelist file (csv formatted)')
parser.add_argument('out',
                    type=str,
                    help='Location of output file (csv formatted)')

args = parser.parse_args()

assert args.out != args.file, 'Input file is output file!'

# whitelist
whitelist = []
with open(args.whitelist, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        whitelist = row

# Load csv
data = pd.read_csv(args.file)
names = data.columns[1:]

# Remove everything not in whitelist
for name in names:
    if name not in whitelist:
        print(name)
        data = data.drop(name, 1)

data.to_csv(args.out)
