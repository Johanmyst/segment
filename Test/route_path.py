#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import random

"""
Author: Blaser, Johannes (11044527)

Description: A simple testfile for creating a graph and
    calculating the path through said graph visiting
    the desired nodes on the way.
"""

help_message = """Graph and Path generator.

Usage: python3 route_path.py [arguments]    -   Generate a graph
    draw a path in it.

Arguments:
    -h or --help          : Print this messagge.
    -n or --nodes  <num>  : Give the number of nodes in the network.
    -d or --degree <num>  : Give the average degree of the network.
    -s or --style <style> : Give the style topology of the network.

Style:
    "simple"    : Defines a simple graph topology.
"""


def get_paths(G, information):
    start = information['initial']
    paths = []

    for node in information['function_nodes']:
        paths.append(nx.shortest_path(G, start, node))
        start = node

    paths.append(nx.shortest_path(G, start, information['target']))

    return paths


def get_edges(G, paths):
    edges = []
    for route in paths:
        edges.append([(route[n], route[n+1]) for n in range(len(route)-1)])
    return edges


def print_graph(G, information):
    if G:
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=information['function_nodes'],
                               node_color='purple',
                               label='VNF')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=set([information['initial']]),
                               node_color='blue',
                               label='Ingress node')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=set([information['target']]),
                               node_color='red',
                               label='Egress node')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=G.nodes() -
                               set(information['function_nodes']) -
                               set([information['initial']]) -
                               set([information['target']]),
                               node_color='grey')
        nx.draw_networkx_labels(G, pos)
        paths = get_paths(G, information)
        edges = get_edges(G, paths)

        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.25)
        for route in edges:
            nx.draw_networkx_edges(G, pos, edgelist=route, width=3.0)

        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.show()
    else:
        print("Print called with a NULL graph. Aborting...")
        exit(1)


def create_graph(topology, info):
    if topology == 'simple':
        return nx.barabasi_albert_graph(info['num_nodes'],
                                        info['avg_degree'])
    elif topology == 'star':
        return nx.star_graph(info['num_nodes'])
    elif topology == 'tree':
        return nx.random_tree(info['num_nodes'])
    elif topology == 'ladder':
        return nx.ladder_graph(round(info['num_nodes'] / 2))
    else:
        print("Invalid network style received. Aborting...")
        exit(1)


def get_function_nodes(information):
    function_nodes = []

    for i in range(information['functions']):
        function_nodes.append(random.randint(0, information['num_nodes'] + 1))

    return function_nodes


def main():
    information = {
        'functions': 1,
        'avg_degree': 3,
        'num_nodes': 20,
        'style': 'simple',
        'initial': 2,
        'target': 19,
        'prob1': 0.5,
        'prob2': 0.5
    }

    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hn:d:s:f:i:t:',
                                       ['help',
                                        'nodes=',
                                        'degree=',
                                        'style=',
                                        'functions=',
                                        'initial=',
                                        'target='])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print(help_message)
            exit(0)
        elif opt in ('-n', '--nodes'):
            information['num_nodes'] = int(arg)
        elif opt in ('-d', '--degree'):
            information['avg_degree'] = int(arg)
        elif opt in ('-f', '--functions'):
            information['functions'] = int(arg)
        elif opt in ('-s', '--style'):
            information['style'] = str(arg)
        elif opt in ('-i', '--initial'):
            information['initial'] = int(arg)
        elif opt in ('-t', '--target'):
            information['target'] = int(arg)

    G = create_graph(information['style'], information)
    information['function_nodes'] = get_function_nodes(information)
    print_graph(G, information)


if __name__ == "__main__":
    main()
