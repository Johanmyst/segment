#!/usr/bin/env python3

"""
Author: Blaser, Johannes (11044527)

Description: A simple testfile for creating and testing simple topologies.
"""

import networkx as nx
import matplotlib.pyplot as plt
import sys
import getopt

def print_graph(G):
    nx.draw(G)
    plt.draw()
    plt.show()

def create_graph(information):
    num_edges = information['num_nodes'] * information['avg_degree']
    num_edges = round(num_edges)
    return nx.gnm_random_graph(information['num_nodes'], num_edges)

def main():
    information = {}
    set_nodes = False
    set_degree = False
    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hn:d:',
                                       ['help',
                                        'nodes=',
                                        'degree='])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print("I'll help you.")
            exit(0)
        elif opt in ('-n', '--nodes'):
            set_nodes = True
            information['num_nodes'] = int(arg)
        elif opt in ('-d', '--degree'):
            set_degree = True
            information['avg_degree'] = int(arg)

    if not set_nodes:
        information['num_nodes'] = 6
    if not set_degree:
        information['avg_degree'] = 3

    G = create_graph(information)
    print_graph(G)


if __name__ == "__main__":
    main()