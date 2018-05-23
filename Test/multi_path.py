#!/usr/bin/env python3

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

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import random

def get_closest(G, information):
    distances = {}

    for n in information['function_nodes']:
        if nx.has_path(G, information['source'], n):
            distances[n] = nx.shortest_path_length(G, information['source'], n)

    return min(distances, key=distances.get)

def get_shortest_length(G, closest, information):
    lengths = {}

    length = nx.shortest_path_length(G, information['source'], closest)
    length += nx.shortest_path_length(G, closest, information['target'])

    for n in information['function_nodes']:
        if nx.has_path(G, information['source'], n):
            lengths[n] = nx.shortest_path_length(G, information['source'], n)

    for n in information['function_nodes']:
        if nx.has_path(G, n, information['target']):
            lengths[n] += nx.shortest_path_length(G, n, information['target'])

    if min(lengths, key=lengths.get) is closest:
        return 1
    else:
        return 0

def get_edges(G, closest, information):
    edges = []
    paths = []

    paths.append(nx.shortest_path(G, information['source'], closest))
    paths.append(nx.shortest_path(G, closest, information['target']))

    for route in paths:
        edges.append([(route[n], route[n+1]) for n in range(len(route)-1)])
    return edges

def create_graph(information):
    if information['style'] == 'simple':
        return nx.barabasi_albert_graph(information['num_nodes'],
                                        information['avg_degree'])
    elif information['style'] == 'star':
        return nx.star_graph(information['num_nodes'])
    elif information['style'] == 'wheel':
        return nx.wheel_graph(information['num_nodes'])
    elif information['style'] == 'lobster':
        return nx.random_lobster(information['num_nodes'],
                                 information['prob1'],
                                 information['prob2'])
    else:
        print("Invalid network style received. Aborting...")
        exit(1)

def print_graph(G, information):
    if G:
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G, pos,
                               nodelist=information['function_nodes'],
                               node_color='blue')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=set([information['source']]),
                               node_color='green')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=set([information['target']]),
                               node_color='purple')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=G.nodes() - set(information['function_nodes']) - set([information['source']]) - set([information['target']]),
                               node_color='red')
        nx.draw_networkx_labels(G, pos)
        closest = get_closest(G, information)
        edges = get_edges(G, closest, information)

        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.25)
        for route in edges:
            nx.draw_networkx_edges(G, pos, edgelist=route, width=3.0)

        plt.show()
    else:
        print("Print called with a NULL graph. Aborting...")
        exit(1)

def get_random_function_nodes(information):
    function_nodes = []

    for i in range(information['functions']):
        function_nodes.append(random.randint(0, information['num_nodes'] - 1))

    return function_nodes

def get_random_source(information):
    return random.randint(0, information['num_nodes'] - 1)

def get_random_target(information):
    number = random.randint(0, information['num_nodes'] - 1)
    while number is information['source']:
        number = random.randint(0, information['num_nodes'] - 1)
    return number

def main():
    information = {
        'functions'  : 5,
        'avg_degree' : 3,
        'num_nodes'  : 50,
        'style'      : 'simple',
        'source'     : 0,
        'target'     : 49,
        'prob1'      : 0.5,
        'prob2'      : 0.5,
        'cycles'     : 100000
    }

    topologies = ['simple', 'star', 'wheel', 'lobster']

    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hn:d:l:f:s:t:p1:p2:c:',
                                       ['help',
                                        'nodes=',
                                        'degree=',
                                        'topology=',
                                        'functions=',
                                        'source=',
                                        'target=',
                                        'probability1=',
                                        'probability2=',
                                        'cycles='])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print(help_message)
            exit(0)
        elif opt in ('-n', '--nodes'):
            information['num_nodes']  = int(arg)
        elif opt in ('-d', '--degree'):
            information['avg_degree'] = int(arg)
        elif opt in ('-f', '--functions'):
            information['functions']  = int(arg)
        elif opt in ('-l', '--topology'):
            information['style']      = str(arg)
        elif opt in ('-s', '--source'):
            information['source']    = int(arg)
        elif opt in ('-t', '--target'):
            information['target']     = int(arg)
        elif opt in ('-c', '--cycles'):
            information['cycles']     = int(arg)
        elif opt in ('-p1', '--probability1'):
            information['prob1']     = float(arg)
        elif opt in ('-p2', '--probability2'):
            information['prob2']     = float(arg)

    G = create_graph(information)

    took_shortest = 0
    for i in range(information['cycles']):
        information['function_nodes'] = get_random_function_nodes(information)
        information['source']         = get_random_source(information)
        information['target']         = get_random_target(information)
        closest = get_closest(G, information)
        took_shortest += get_shortest_length(G, closest, information)

    print(took_shortest)

    print_graph(G, information)

if __name__ == "__main__":
    main()
