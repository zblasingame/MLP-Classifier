""" Collection of helpful utilities for deep learning
    Author: Zander Blasingame """

import numpy as np


# function parses csv and returns a list of input matrices and output labels
def parse_csv(filename, normalize=True):
    input_matricies = []
    output_labels = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            if line[0] == '#':
                continue

            line = line[:-1]  # remove newline for string
            entries = line.split(',')

            output_labels.append(entries[0])
            entries = entries[1:]

            in_mat = [float(entry) for entry in entries]

            # normalize matrix
            if normalize:
                input_matricies.append(in_mat/np.linalg.norm(in_mat))
            else:
                input_matricies.append(in_mat)

    return np.array(input_matricies), np.array(output_labels)


# Returns one_hot labels and one_hot mapping
def one_hot_encoding(labels):
    # Convert labels from string to int
    uniq_labels = np.unique(labels)

    one_hot_matrix = np.eye(uniq_labels.size)

    one_hot_map = {label: one_hot_matrix[i]
                   for i, label in enumerate(uniq_labels)}

    encoding = np.vectorize(lambda x: one_hot_map[x])

    return encoding(labels), one_hot_map
