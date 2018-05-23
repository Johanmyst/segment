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

Description: This file analyses the performance impact of different
    orders of segments in the segment list. This will analyse what
    the benefit (if any) is of allowing the ingress node to re-order
    the segments in the segment list. The assumption is made that
    the segments are commutable and are order-independent. This
    file tests on realistic topologies.
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
        for i in range(info['num_VNFs']):
            tmp = np.random.choice(G.nodes())

            while tmp in function_nodes:
                tmp = np.random.choice(G.nodes())

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
    tmp = np.random.choice(G.nodes())

    while tmp in VNFs:
        tmp = np.random.choice(G.nodes())

    return tmp


def generate_target(G, VNFs, source):
    """Generate a random target node. Can't be the VNF nor the source.
    """
    tmp = np.random.choice(G.nodes())

    while tmp in VNFs or tmp is source:
        tmp = np.random.choice(G.nodes())

    return tmp


def get_path_length(G, source, target, VNFs):
    """Determine whether a given path from source to target
    traverses the VNFs defined in the info struct.
    """
    path = [source]

    for VNF in VNFs:
        if nx.has_path(G, path[-1], VNF):
            for hop in nx.shortest_path(G, path[-1], VNF)[1:]:
                path.append(hop)

    if nx.has_path(G, path[-1], target):
        for hop in nx.shortest_path(G, path[-1], target)[1:]:
            path.append(hop)

    return len(path)


def get_optimal_order(G, source, target, VNFs):
    """Generate the optimal path from source to target
    over the given VNFs where the VNFs can be traversed
    in any order.
    """
    all_orders = itertools.permutations(VNFs)
    minimum_cost = math.inf
    optimal = VNFs

    for order in all_orders:
        tmp = get_path_length(G, source, target, order)
        if tmp < minimum_cost:
            optimal = order
            minimum_cost = tmp

    return optimal


def run_cycles(G, info, VNFs):
    """Run a given number of cycles of the experiment. Create
    a random source and target pair each time but keep the
    VNF the same.
    """
    ordered_lengths, random_lengths, optimal_lengths = ([], [], [])
    random_order = VNFs

    for i in range(info['cycles']):
        source = generate_source(G, VNFs)
        target = generate_target(G, VNFs, source)

        # Generate cost for ordered list of segments.
        ordered_lengths.append(get_path_length(G, source, target,
                                               VNFs))

        # Generate same cost for randomly shuffled list.
        np.random.shuffle(random_order)
        random_lengths.append(get_path_length(G, source, target,
                                              random_order))

        # Generate same cost for optimal ordering.
        optimal_order = get_optimal_order(G, source, target, VNFs)
        optimal_lengths.append(get_path_length(G, source, target,
                                               optimal_order))

    return (np.average(ordered_lengths), np.std(ordered_lengths),
            np.average(random_lengths), np.std(random_lengths),
            np.average(optimal_lengths), np.std(optimal_lengths))


def run_experiments(topologies, info):
    """Run the experiments on the parameters defined in the
    information struct.
    """
    avg_ordered, avg_random, avg_optimal = ({}, {}, {})
    std_ordered, std_random, std_optimal = ({}, {}, {})

    for topology in topologies:
        print("Running topology: " + topology + "...")
        ordered_avg, random_avg, optimal_avg = ([], [], [])
        ordered_std, random_std, optimal_std = ([], [], [])

        G = create_graph(topology)

        for i in range(info['topologies']):
            VNFs = generate_function_nodes(G, topology, info,
                                           info['random_VNFs'])

            (tmp_avg1, tmp_std1,
             tmp_avg2, tmp_std2,
             tmp_avg3, tmp_std3) = run_cycles(G, info, VNFs)

            ordered_avg.append(tmp_avg1)
            ordered_std.append(tmp_std1)
            random_avg.append(tmp_avg2)
            random_std.append(tmp_std2)
            optimal_avg.append(tmp_avg3)
            optimal_std.append(tmp_std3)

        avg_ordered[topology] = np.average(ordered_avg)
        avg_random[topology] = np.average(random_avg)
        avg_optimal[topology] = np.average(optimal_avg)

        std_ordered[topology] = np.average(ordered_std)
        std_random[topology] = np.average(random_std)
        std_optimal[topology] = np.average(optimal_std)

    return (avg_ordered, avg_random, avg_optimal,
            std_ordered, std_random, std_optimal)


def visualise_probabilities(topologies, info, results):
    """Visualise the results provided in a bar chart.
    """
    fig, ax = plt.subplots()
    index = np.arange(len(topologies))
    bar_width = 0.25
    opacity = 0.8

    (avg_ordered, avg_random, avg_optimal,
     std_ordered, std_random, std_optimal) = \
        (list(results[0].values()), list(results[1].values()),
         list(results[2].values()), list(results[3].values()),
         list(results[4].values()), list(results[5].values()))

    # Plot the first plot (random and non-random, small sized topology).
    rects1 = ax.bar(index + (bar_width * 0),
                    avg_ordered,
                    bar_width,
                    alpha=opacity,
                    color='red',
                    label='No reordering',
                    yerr=std_ordered)

    rects2 = ax.bar(index + (bar_width * 1),
                    avg_random,
                    bar_width,
                    alpha=opacity,
                    color='blue',
                    label='Random reordering',
                    yerr=std_random)

    rects3 = ax.bar(index + (bar_width * 2),
                    avg_optimal,
                    bar_width,
                    alpha=opacity,
                    color='green',
                    label='Optimal reordering',
                    yerr=std_optimal)

    # Set the title data.
    if info['random_VNFs']:
        plt.title('Path lengths from source to target over ' +
                  '{} randomly '.format(info['num_VNFs']) +
                  'placed VNF(s)')
    else:
        plt.title('Path lengths from source to target over ' +
                  '{} topology-aware '.format(info['num_VNFs']) +
                  'placed VNF(s)')
    plt.ylabel('Path length in number of hops')
    plt.xlabel('Topology')

    plt.legend()

    plt.xticks(index + bar_width, topologies)

    plt.tight_layout()
    plt.show()


def main():
    """Runs the main function.
    Reads the command line information provided and calls the experiments.
    """

    # Define the default data.
    information = {
        'VNF':         0,
        'num_VNFs':    1,
        'topologies':  1000,
        'cycles':      1000,
        'random_VNFs': False
    }

    # Define the topologies experimented upon.
    topologies = ['NORDUnet', 'GEANT', 'SURFnet']

    # Read data from command line.
    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hs:m:l:V:t:c:r:',
                                       ['help',
                                        'num_VNFs=',
                                        'topologies=',
                                        'cycles=',
                                        'random_VNFs='])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print(help_message)
            exit(0)
        elif opt in ('-V', '--num_VNFs'):
            information['num_VNFs'] = int(arg)
        elif opt in ('-t', '--topologies'):
            information['topologies'] = int(arg)
        elif opt in ('-c', '--cycles'):
            information['cycles'] = int(arg)
        elif opt in ('-r', '--random_VNFs'):
            information['random_VNFs'] = bool(int(arg))

    # Relay parameters to user.
    if information['random_VNFs']:
        print("Running " + str(information['cycles']) + " cycles on " +
              str(information['topologies']) +
              " topologies using random VNF placement...")
    else:
        print("Running " + str(information['cycles']) + " cycles on " +
              str(information['topologies']) +
              " topologies using non-random VNF placement...")

    # Perform experiments.
    results = run_experiments(topologies, information)
    visualise_probabilities(topologies, information, results)


if __name__ == "__main__":
    main()
