#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import random

"""
Author: Blaser, Johannes (11044527)

Description: A simple program to analyse the probability a
    given VNF is traversed using traditional routing
    paradigms. This is analysed for a spectrum of different
    topologies. The difference between random VNF placement
    and clever topology-aware placement is visualised.
"""


def create_graph(topology, info):
    """Create a graph depending on the information in the
    info struct provided.
    """
    if topology == "NORDUnet":
        G = nx.Graph()
        G.add_nodes_from(np.arange(15))
        G.add_edges_from([(0, 1), (0, 2), (1, 5), (2, 3), (2, 4), (2, 5),
                          (3, 4), (4, 6), (5, 6), (5, 7), (6, 8), (7, 8),
                          (7, 9), (8, 11), (9, 10), (9, 13), (10, 12),
                          (10, 14), (11, 12), (13, 14)])
        return G
    elif topology == "GEANT":
        G = nx.Graph()
        G.add_nodes_from(np.arange(45))
        G.add_edges_from([(0, 23), (1, 10), (2, 10), (2, 33), (2, 42),
                          (2, 35), (2, 5), (2, 18), (2, 17), (2, 38),
                          (2, 23), (3, 10), (4, 31), (4, 41), (5, 19),
                          (5, 35), (5, 29), (6, 33), (7, 10), (7, 23),
                          (7, 15), (7, 43), (8, 10), (8, 41), (9, 10),
                          (9, 19), (10, 31), (10, 44), (10, 33), (10, 16),
                          (10, 19), (10, 40), (10, 21), (10, 25), (11, 32),
                          (11, 37), (11, 44), (11, 31), (11, 22), (12, 44),
                          (12, 26), (13, 15), (13, 43), (13, 34), (14, 37),
                          (15, 41), (16, 19), (17, 23), (18, 19), (18, 38),
                          (19, 39), (19, 35), (19, 40), (19, 36), (19, 28),
                          (20, 41), (21, 41), (22, 41), (23, 30), (23, 43),
                          (24, 26), (24, 33), (25, 31), (27, 35), (29, 35),
                          (31, 44), (32, 37), (34, 41)])
        return G
    elif topology == "SURFnet":
        G = nx.Graph()
        G.add_nodes_from(np.arange(50))
        G.add_edges_from([(0, 1), (0, 6), (1, 2), (2, 3), (3, 4), (4, 5),
                          (5, 7), (6, 7), (6, 8), (8, 9), (9, 10), (10, 7),
                          (6, 11), (11, 14), (6, 14), (6, 33), (6, 31),
                          (14, 13), (13, 17), (17, 20), (20, 24), (24, 31),
                          (14, 31), (14, 7), (8, 12), (12, 18), (18, 21),
                          (21, 33), (7, 15), (15, 19), (19, 22), (22, 25),
                          (25, 34), (7, 34), (7, 16), (16, 33), (33, 34),
                          (33, 34), (33, 26), (26, 27), (27, 28), (28, 29),
                          (29, 30), (30, 34), (32, 33), (31, 32), (31, 35),
                          (35, 32), (31, 36), (36, 38), (38, 40), (40, 42),
                          (42, 44), (31, 44), (32, 45), (32, 37), (37, 39),
                          (39, 41), (41, 45), (44, 45), (44, 47), (47, 48),
                          (48, 49), (49, 45), (45, 46), (46, 43), (43, 34),
                          (45, 34)])
        return G
    else:
        print("Invalid network style received. Aborting...")
        exit(1)


def generate_function_nodes(G, topology, info, is_rand):
    """Generate a VNF's location based on the provided
    parameters in the info object. Randomly placed or not.
    """
    function_nodes = []

    if is_rand:
        # VNF is randomly placed. Pick any unique place.
        for i in range(info['VNFs']):
            tmp = np.random.choice(list(G.nodes()))

            while tmp in function_nodes:
                tmp = np.random.choice(list(G.nodes()))

            function_nodes.append(tmp)
    else:
        # Calculate most connected nodes.
        degrees = list(G.degree())
        degree_sequence = sorted(degrees, key=lambda tup: tup[1], reverse=True)
        if len(degree_sequence) is 0:
            return []

        # Caluclate most "middle" nodes in network and place VNF there.
        if topology == 'lobster':
            for i in range(info['VNFs']):
                num = int(round(info['num_nodes'] / 2))

                j, tmp = (0, num)

                while tmp in function_nodes:
                    if j % 2 is 0:
                        tmp = num + j
                    else:
                        tmp = num - j
                    j += 1

                if num not in G.nodes():
                    num = np.random.choice(list(G.nodes()))

                function_nodes.append(num)

        # Else just pick the most connected nodes.
        else:
            for i in range(info['VNFs']):
                function_nodes.append(degree_sequence[i][0])

    return function_nodes


def generate_source(G, info):
    """Generate a random source node. Can't be the VNF.
    """
    tmp = np.random.choice(list(G.nodes()))

    while tmp in info['curr_VNFs']:
        tmp = np.random.choice(list(G.nodes()))

    return tmp


def generate_target(G, info, source):
    """Generate a random target node. Can't be the VNF nor the source.
    """
    tmp = np.random.choice(list(G.nodes()))

    while tmp in info['curr_VNFs'] or tmp is source:
        tmp = np.random.choice(list(G.nodes()))

    return tmp


def hit_VNF(G, info, source, target):
    """Determine whether a given path from source to target
    traverses the VNFs defined in the info struct.
    """
    paths = []
    any_hit = False

    # Calulcate shortest path from source to target.
    if source in G.nodes() and target in G.nodes():
        if nx.has_path(G, source, target):
            paths = nx.all_shortest_paths(G, source, target)
        else:
            return False
    else:
        return False

    # See if all VNFs are on above path.
    for path in paths:
        hit_all = True
        for element in info['curr_VNFs']:
            if element not in path:
                hit_all = False
        if hit_all:
            any_hit = True

    return any_hit


def run_cycles(G, info):
    """Run a given topology cycle number of times. Return how often
    a VNF was encounted during the cycles.
    """
    count = 0
    pairs = []

    for n in range(info['cycles']):
        # Generate new random source and target.
        source = generate_source(G, info)
        target = generate_target(G, info, source)

        # Store the pair to make sure it's not re-used.
        pairs.append((source, target))

        # While the pair is non-unique, generate a new pair.
        while (source, target) in pairs:
            source = generate_source(G, info)
            target = generate_target(G, info, source)

        # Check if VNF is on path from source to target.
        if hit_VNF(G, info, source, target):
            count += 1

    return count


def run_topology(topology, info, is_rand):
    """Run a given topology either with a randomly placed VNF
    or a non-randomly placed VNF.
    """
    count = []

    for n in range(info['topologies']):
        # Create a new graph.
        G = create_graph(topology, info)
        info['curr_VNFs'] = generate_function_nodes(G, topology, info, is_rand)

        count.append(run_cycles(G, info))

    return np.sum(count), np.std(count)


def run_experiments(topologies, info):
    """Run the experiments on the parameters defined in the
    information struct.
    """
    random_prob = {}
    random_std = {}
    non_random_prob = {}
    non_random_std = {}

    # Saves some computation.
    divisor = (info['topologies'] * info['cycles'])

    # Run each topology.
    for topology in topologies:
        # Update user on progress.
        print("Running topology: " + str(topology) + "...")

        # Calculate probability and standard deviation for random.
        random_prob[topology], random_std[topology] = \
            run_topology(topology, info, True)
        random_prob[topology] /= divisor
        random_std[topology] /= info['cycles']

        # Calcluate probability and standard devication for non-random.
        non_random_prob[topology], non_random_std[topology] = \
            run_topology(topology, info, False)
        non_random_prob[topology] /= divisor
        non_random_std[topology] /= info['cycles']

    return (random_prob, random_std,
            non_random_prob, non_random_std)


def visualise_probabilities(topologies, info, probs):
    """Visualise the probs provide in a bar chart.
    """
    fig, ax = plt.subplots()
    index = np.arange(len(topologies))
    bar_width = 0.35
    opacity = 0.8

    # Plot the first plot (random and non-random, small sized topology).
    rects1 = ax.bar(index, list(probs[0].values()), bar_width,
                    alpha=opacity,
                    color='blue',
                    label='Random VNF placement',
                    yerr=list(probs[1].values()))

    rects2 = ax.bar(index + bar_width, list(probs[2].values()), bar_width,
                    alpha=opacity,
                    color='g',
                    label='Topology-aware VNF placement',
                    yerr=list(probs[3].values()))

    # Set the title data.
    plt.title('Probability of VNF traversal over {} VNF(s)'.
              format(info['VNFs']))
    plt.ylabel('Probability of traversing VNF')
    plt.xlabel('Topology')

    plt.legend(loc=0)

    plt.ylim(0, 1)
    plt.xticks(index + bar_width / 2, topologies)

    plt.tight_layout()
    plt.show()


def main():
    """Runs the main function.
    Reads the command line information provided and calls the experiments.
    """

    # Define the default data.
    information = {
        'VNF':         0,
        'VNFs':        1,
        'topologies':  1000,
        'cycles':      1000
    }

    # Define the topologies experimented upon.
    topologies = ['NORDUnet', 'GEANT', 'SURFnet']

    # Read data from command line.
    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hs:m:l:V:t:c:',
                                       ['help',
                                        'VNFs=',
                                        'topologies=',
                                        'cycles='])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print(help_message)
            exit(0)
        elif opt in ('-V', '--VNFs'):
            information['VNFs'] = int(arg)
        elif opt in ('-t', '--topologies'):
            information['topologies'] = int(arg)
        elif opt in ('-c', '--cycles'):
            information['cycles'] = int(arg)

    # Relay parameters to user.
    print("Running " + str(information['cycles']) + " cycles on " +
          str(information['topologies']) + " topologies...")

    # Perform experiments.
    probabilities = run_experiments(topologies, information)
    visualise_probabilities(topologies, information, probabilities)


if __name__ == "__main__":
    main()
