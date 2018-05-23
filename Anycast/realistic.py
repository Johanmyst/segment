#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import math
import getopt
import itertools

"""
Author: Blaser, Johannes (11044527)

Description: A simple testfile to analyse the effectiveness of
    using multiple identical VNFs that perform an equivalent
    function -- an anycast. This produces a plot that shows
    the performance impact when using n identical VNFs in a
    given topology. The result will be a graph showing the
    effectiveness of using n VNFs in a given topology in
    terms of performance benefit. Uses realistic topologies.
"""


def create_graph(topology):
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
                          (45, 34), (23, 0)])
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
        for i in range(info['num_VNFs']):
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

        for i in range(info['num_VNFs']):
            function_nodes.append(degree_sequence[i][0])

    return function_nodes


def generate_source(G, VNFs):
    """Generate a random source node. Can't be the VNF.
    """
    tmp = np.random.choice(list(G.nodes()))

    while tmp in VNFs:
        tmp = np.random.choice(list(G.nodes()))

    return tmp


def generate_target(G, VNFs, source):
    """Generate a random target node. Can't be the VNF nor the source.
    """
    tmp = np.random.choice(list(G.nodes()))

    while tmp in VNFs or tmp is source:
        tmp = np.random.choice(list(G.nodes()))

    return tmp


def get_path_length(G, source, target, VNF):
    """Determine whether a given path from source to target
    traverses the VNFs defined in the info struct.
    """
    path = [source]

    if nx.has_path(G, path[-1], VNF):
        for hop in nx.shortest_path(G, path[-1], VNF)[1:]:
            path.append(hop)

    if nx.has_path(G, path[-1], target):
        for hop in nx.shortest_path(G, path[-1], target)[1:]:
            path.append(hop)

    return len(path)


def find_closest(G, source, VNFs):
    """Generate the optimal path from source to target
    over the given VNFs where the VNFs can be traversed
    in any order.
    """
    minimum_cost = math.inf
    closest = -1

    if VNFs:
        if nx.has_path(G, source, VNFs[0]):
            minimum_cost = nx.shortest_path_length(G, source, VNFs[0])
            closest = VNFs[0]

        for VNF in VNFs:
            if nx.has_path(G, source, VNF):
                tmp = nx.shortest_path_length(G, source, VNF)
                if tmp < minimum_cost:
                    minimum_cost = tmp
                    closest = VNF

    return closest


def run_cycles(G, info, VNFs):
    """Run a given number of cycles of the experiment. Create
    a random source and target pair each time but keep the
    VNF the same.
    """
    lengths = []
    pairs = []
    random_order = VNFs

    for i in range(info['cycles']):
        source = generate_source(G, VNFs)
        target = generate_target(G, VNFs, source)

        while (source, target) in pairs:
            source = generate_source(G, VNFs)
            target = generate_target(G, VNFs, source)

        pairs.append((source, target))

        # Generate same cost for optimal ordering.
        if VNFs:
            closest = find_closest(G, source, VNFs)
            if closest is -1:
                continue
            lengths.append(get_path_length(G, source, target, closest))
        else:
            if nx.has_path(G, source, target):
                lengths.append(nx.shortest_path_length(G, source, target))

    return np.average(lengths), np.std(lengths)


def run_experiments(topologies, info):
    """Run the experiments on the parameters defined in the
    information struct.
    """
    total_avg, total_std = {}, {}

    for topology in topologies:
        print("Running topology: " + topology + "...")

        avg1, std1 = [], []

        G = create_graph(topology)

        for i in range(info['topologies']):
            avg2, std2 = [], []

            VNFs = []
            avg3, std3 = run_cycles(G, info, VNFs)
            avg2.append(avg3)
            std2.append(std3)

            for j in range(1, len(G.nodes()) - info['free_nodes']):
                info['num_VNFs'] = j
                VNFs = generate_function_nodes(G, topology, info,
                                               info['random_VNFs'])

                avg3, std3 = run_cycles(G, info, VNFs)

                avg2.append(avg3)
                std2.append(std3)

            avg1.append(avg2)
            std1.append(std2)

        total_avg[topology] = np.average(avg1, axis=0)
        total_std[topology] = np.average(std1, axis=0)

    return total_avg, total_std


def visualise_probabilities(topologies, info, results):
    """Visualise the results provided in a bar chart.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    total_avg = list(results[0].values())
    total_std = list(results[1].values())

    x_vals0 = np.arange(1, len(total_avg[0]))
    x_vals1 = np.arange(1, len(total_avg[1]))
    x_vals2 = np.arange(1, len(total_avg[2]))

    base0 = list(total_avg[0])[0]
    base1 = list(total_avg[1])[0]
    base2 = list(total_avg[2])[0]

    y_vals0 = list(total_avg[0])[1:]
    y_vals1 = list(total_avg[1])[1:]
    y_vals2 = list(total_avg[2])[1:]

    y_err0 = list(total_std[0])[1:]
    y_err1 = list(total_std[1])[1:]
    y_err2 = list(total_std[2])[1:]

    y_max = max(y_vals0)
    if max(y_vals1) > y_max:
        y_max = max(y_vals1)
    if max(y_vals2) > y_max:
        y_max = max(y_vals2)

    y_min = min(y_vals0)
    if min(y_vals1) < y_min:
        y_min = min(y_vals1)
    if min(y_vals2) < y_min:
        y_min = min(y_vals2)

    y_incr = max(y_err0)
    if max(y_err1) > y_incr:
        y_incr = max(y_err1)
    if max(y_err2) > y_incr:
        y_incr = max(y_err2)

    y_max += y_incr + 0.5
    y_min -= y_incr - 0.5

    # Plot the first plot (random and non-random, small sized topology).
    plot0 = ax1.errorbar(x_vals0, y_vals0,
                         yerr=y_err0,
                         label='NORDUnet',
                         color='red',
                         fmt='-o',
                         capthick=1,
                         capsize=2,
                         elinewidth=1,
                         markeredgewidth=1)
    plot1 = ax2.errorbar(x_vals1, y_vals1,
                         yerr=y_err1,
                         label='GEANT',
                         color='blue',
                         fmt='-o',
                         capthick=1,
                         capsize=2,
                         elinewidth=1,
                         markeredgewidth=1)
    plot2 = ax3.errorbar(x_vals2, y_vals2,
                         yerr=y_err2,
                         label='SURFnet',
                         color='green',
                         fmt='-o',
                         capthick=1,
                         capsize=2,
                         elinewidth=1,
                         markeredgewidth=1)

    # Set the title data.
    ax1.set(ylabel='Path length')
    ax1.set(xlabel='Number of VNFs')
    ax1.set(ylim=[y_min, y_max])
    ax1.legend(loc=0)

    ax2.set(ylabel='Path length')
    ax2.set(xlabel='Number of VNFs')
    ax2.set(ylim=[y_min, y_max])
    ax2.legend(loc=0)

    ax3.set(ylabel='Path length')
    ax3.set(xlabel='Number of VNFs')
    ax3.set(ylim=[y_min, y_max])
    ax3.legend(loc=0)

    # Set the title data.
    if info['random_VNFs']:
        ax1.set(title='Path lengths from source to target over randomly ' +
                'placed VNF(s)')
    else:
        ax1.set(title='Path lengths from source to target over ' +
                'topology-aware placed VNF(s)')

    plt.tight_layout()
    plt.show()


def main():
    """Runs the main function.
    Reads the command line information provided and calls the experiments.
    """

    # Define the default data.
    information = {
        'VNF':         0,
        'num_VNFs':    0,
        'max_VNFs':    10,
        'topologies':  1000,
        'cycles':      1000,
        'random_VNFs': False,
        'free_nodes':  0
    }

    # Define the number of nodes needed to be left free.
    free_nodes = 0

    # Define the topologies experimented upon.
    topologies = ['NORDUnet', 'GEANT', 'SURFnet']

    # Read data from command line.
    options, remainder = getopt.getopt(sys.argv[1:],
                                       'ht:c:r:m:',
                                       ['help',
                                        'topologies=',
                                        'cycles=',
                                        'random_VNFs=',
                                        'max_VNFs='])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print(help_message)
            exit(0)
        elif opt in ('-t', '--topologies'):
            information['topologies'] = int(arg)
        elif opt in ('-c', '--cycles'):
            information['cycles'] = int(arg)
        elif opt in ('-r', '--random_VNFs'):
            information['random_VNFs'] = bool(int(arg))
        elif opt in ('-m', '--max_VNFs'):
            information['max_VNFs'] = int(arg)

    # Relay parameters to user.
    if information['random_VNFs']:
        print("Running " + str(information['cycles']) + " cycles on " +
              str(information['topologies']) +
              " topologies using random VNF placement...")
    else:
        print("Running " + str(information['cycles']) + " cycles on " +
              str(information['topologies']) +
              " topologies using non-random VNF placement...")

    while (free_nodes * (free_nodes - 1)) < information['cycles']:
        free_nodes += 1

    information['free_nodes'] = free_nodes

    print("Number of free nodes: {}".format(information['free_nodes']))

    # Perform experiments.
    results = run_experiments(topologies, information)
    visualise_probabilities(topologies, information, results)


if __name__ == "__main__":
    main()
