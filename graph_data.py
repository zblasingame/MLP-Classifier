"""Script to graph the data of the accuracy in relation to HPC events

Author:         Zander Blasingame
Institution:    Clarkson University
Lab:            CAMEL
"""

import argparse
import json
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('file',
                    type=str,
                    help='Location of input file (json formatted)')
parser.add_argument('--online',
                    action='store_true',
                    help='Flag to publish graphs online')

args = parser.parse_args()

ONLINE = args.online


# Functions
def make_graph(data, filename, layout=go.Layout()):
    fig = go.Figure(data=data, layout=layout)

    if ONLINE:
        return py.iplot(fig, filename=filename)
    else:
        filename = './graphs/{}.png'.format(filename)
        py.image.save_as(fig, filename=filename)
        return None

dataset_stats = []

with open(args.file, 'r') as f:
    dataset_stats = json.load(f)

x = [','.join(entry['combination']) for entry in dataset_stats]
y = [entry['accuracy'] for entry in dataset_stats]


data = [go.Bar(x=x, y=y)]

name = args.file.split('.')[0].split('/')[-1]

layout = go.Layout(title='Accuracy for Combination in {}'.format(name),
                   yaxis=dict(title='Accuracy (%)'),
                   annotations=[dict(x=xi, y=yi,
                                     text='{:.2f}'.format(yi),
                                     xanchor='center',
                                     yanchor='bottom',
                                     showarrow=False)
                                for xi, yi, in zip(x, y)])

plot_url = make_graph(data=data, layout=layout,
                      filename='{}-freak-classification-bar'.format(name))
