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

def get_paths(G, information):
    start = information['initial']
    paths = []

    for node in information['function_nodes']:
        if nx.has_path(G, start, node):
            paths.append(nx.shortest_path(G, start, node))
            start = node

    if nx.has_path(G, start, information['target']):
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
                               node_color='blue')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=set([information['initial']]),
                               node_color='green')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=set([information['target']]),
                               node_color='purple')
        nx.draw_networkx_nodes(G, pos,
                               nodelist=G.nodes() - set(information['function_nodes']) - set([information['initial']]) - set([information['target']]),
                               node_color='red')
        nx.draw_networkx_labels(G, pos)
        paths = get_paths(G, information)
        edges = get_edges(G, paths)

        nx.draw_networkx_edges(G, pos, edgelist=G.edges(), alpha=0.25)
        for route in edges:
            nx.draw_networkx_edges(G, pos, edgelist=route, width=3.0)

        plt.show()
    else:
        print("Print called with a NULL graph. Aborting...")
        exit(1)

def print_plots(topologies, information, random_trues, non_random_trues):
    fig, ax = plt.subplots()
    index = np.arange(len(topologies))
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, random_trues, bar_width,
                     alpha=opacity,
                     color='blue',
                     label='Random VNF allocation')

    rects2 = plt.bar(index + bar_width, non_random_trues, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Non-random VNF allocation')

    plt.xlabel('Topology')
    plt.ylabel('Probability of traversing VNF')
    plt.title('Probability of VNF traversal by topology')
    plt.xticks(index + bar_width, topologies)
    plt.legend()

    plt.tight_layout()
    plt.show()


def hit_service_node(G, information):
    if information['initial'] in G.nodes() and information['target'] in G.nodes():
        if nx.has_path(G, information['initial'], information['target']):
            path = nx.shortest_path(G, information['initial'], information['target'])
        else:
            return False
    else:
        return False

    hit_all = True
    for element in information['function_nodes']:
        if element not in path:
            hit_all = False
    return hit_all

def create_graph(information):
    if information['style'] == 'simple':
        return nx.barabasi_albert_graph(information['num_nodes'],
                                        information['avg_degree'])
    elif information['style'] == 'star':
        return nx.star_graph(information['num_nodes'])
    elif information['style'] == 'tree':
        return nx.random_tree(information['num_nodes'])
    elif information['style'] == 'lobster':
        return nx.random_lobster(information['num_nodes'],
                                 information['prob1'],
                                 information['prob2'])
    else:
        print("Invalid network style received. Aborting...")
        exit(1)

def get_random_function_nodes(information):
    function_nodes = []

    for i in range(information['functions']):
        function_nodes.append(random.randint(0, information['num_nodes'] - 1))

    return function_nodes

def get_non_random_function_nodes(G, information):
    function_nodes = []

    degrees = list(G.degree())
    degree_sequence = sorted(degrees, key=lambda tup: tup[1], reverse=True)

    if information['style'] == 'simple' or \
        information['style'] == 'star' or \
        information['style'] == 'tree':
        for i in range(information['functions']):
            function_nodes.append(degree_sequence[i][0])

    elif information['style'] == 'lobster':
        for i in range(information['functions']):
            num = round(random.gauss(int(information['num_nodes'] / 2), 1))
            if num not in function_nodes:
                function_nodes.append(num)

    return function_nodes

def get_random_initial(information):
    return random.randint(0, information['num_nodes'] - 1)

def get_random_target(information):
    return random.randint(0, information['num_nodes'] - 1)
    while number is information['initial']:
        number = random.randint(0, information['num_nodes'] - 1)
    return number

def run_random_cycles(information):
    print("Running random " + str(information['style'] + "..."))
    num_true = 0
    G = create_graph(information)
    information['function_nodes'] = get_random_function_nodes(information)
    for i in range(information['cycles']):
        information['initial']        = get_random_initial(information)
        information['target']         = get_random_target(information)
        if hit_service_node(G, information):
            num_true += 1
    return num_true

def run_non_random_cycles(information):
    print("Running non-random " + str(information['style'] + "..."))
    num_true = 0
    G = create_graph(information)
    information['function_nodes'] = get_non_random_function_nodes(G, information)
    for i in range(information['cycles']):
        information['initial']        = get_random_initial(information)
        information['target']         = get_random_target(information)
        if hit_service_node(G, information):
            num_true += 1

    return num_true

def main():
    information = {
        'functions'  : 1,
        'avg_degree' : 3,
        'num_nodes'  : 20,
        'style'      : 'simple',
        'initial'    : 0,
        'target'     : 19,
        'prob1'      : 0.5,
        'prob2'      : 0.5,
        'cycles'     : 100000
    }

    topologies = ['simple', 'star', 'tree', 'lobster']

    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hn:d:s:f:i:t:p1:p2:c:',
                                       ['help',
                                        'nodes=',
                                        'degree=',
                                        'style=',
                                        'functions=',
                                        'initial=',
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
        elif opt in ('-s', '--style'):
            information['style']      = str(arg)
        elif opt in ('-i', '--initial'):
            information['initial']    = int(arg)
        elif opt in ('-t', '--target'):
            information['target']     = int(arg)
        elif opt in ('-c', '--cycles'):
            information['cycles']     = int(arg)
        elif opt in ('-p1', '--probability1'):
            information['prob1']     = float(arg)
        elif opt in ('-p2', '--probability2'):
            information['prob2']     = float(arg)

    random_trues = []
    for topology in topologies:
        information['style'] = topology
        random_trues.append(run_random_cycles(information))
    print("Random: " + str(random_trues))

    non_random_trues = []
    for topology in topologies:
        information['style'] = topology
        non_random_trues.append(run_non_random_cycles(information))
    print("Non-random: " + str(non_random_trues))

    new_random_trues = [x / information['cycles'] for x in random_trues]
    new_non_random_trues = [x / information['cycles'] for x in non_random_trues]

    print_plots(topologies, information, new_random_trues, new_non_random_trues)

if __name__ == "__main__":
    main()
