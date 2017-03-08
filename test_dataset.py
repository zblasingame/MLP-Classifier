"""Tests all subsets of dataset and each permutation of features

Author:         Zander Blasingame
Instiution:     Clarkson University
Lab:            CAMEL
"""

import argparse
import csv
import classifier
import itertools
import json
import os

# Parser arguments
parser = argparse.ArgumentParser()
parser.add_argument('dir',
                    type=str,
                    help=('Location of subset directories containing '
                          'training and testing csv files '
                          '(must be labeled train.csv and test.csv)'))
parser.add_argument('whitelist',
                    type=str,
                    help='Location of the whitelist file (csv formatted)')
parser.add_argument('out',
                    type=str,
                    help='Location of output file (json formatted)')

args = parser.parse_args()

# whitelist
whitelist = []
with open(args.whitelist, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        whitelist = row

hpc_combinations = list(itertools.combinations(whitelist, 4))

dataset_data = [{'combination': entry, 'accuracy': 0}
                for i, entry in enumerate(hpc_combinations)]
num_subset = 0

# Iterate through each subloop
for path, dirs, files in os.walk(args.dir):
    if 'subset' in path:
        train = '{}/train.csv'.format(path)
        test = '{}/test.csv'.format(path)

        # Iterate through each combinations of HPCs
        for i, entry in enumerate(hpc_combinations):
            # generate blacklist from whitelist
            blacklist = [el for el in whitelist if el not in entry]

            num_input, num_classes = classifier.get_dimensions(train,
                                                               blacklist)

            mlp = classifier.Classifier(num_input, 15, num_classes,
                                        batch_size=200, num_epochs=100,
                                        blacklist=blacklist)

            # train and test neural net
            mlp.train(train)
            acc = mlp.test(test)
            print(i)
            print(acc)

            dataset_data[i]['accuracy'] += acc

        print('data')
        print(dataset_data)
        num_subset += 1

for i in range(len(hpc_combinations)):
    dataset_data[i]['accuracy'] /= num_subset

with open(args.out, 'w') as f:
    json.dump(dataset_data, f, indent=2)
