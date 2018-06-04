#!/usr/bin/env python3

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import sys
import getopt
import random

"""
Author: Blaser, Johannes (11044527)

Description: A simple program to analyse the performance
    impact upon using Segment Routing and having to traverse
    a given VNF. Both done for random VNF placemenet and for
    topology-aware VNF placement.
"""


def create_graph(topology):
    """Create a graph depending on the information in the
    info struct provided.
    """
    if topology == "NORDUnet":
        G = nx.Graph()
        G.add_nodes_from(np.arange(24))
        G.add_edges_from([(0, 2), (1, 5), (2, 4), (2, 5), (2, 6), (3, 4),
                          (4, 6), (5, 6), (4, 8), (5, 11), (6, 8), (6, 9),
                          (7, 8), (7, 15), (8, 9), (8, 10), (8, 11),
                          (10, 11), (10, 12), (11, 13), (11, 14), (12, 13),
                          (13, 14), (14, 15), (14, 17), (14, 22), (14, 23),
                          (15, 16), (15, 18), (16, 18), (18, 19), (18, 23),
                          (19, 21), (21, 20)])
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


def run_cycles(G, info, VNFs):
    """Run a given number of cycles of the experiment. Create
    a random source and target pair each time but keep the
    VNF the same.
    """
    lengths = []
    pairs = []

    for i in range(info['cycles']):
        source = generate_source(G, VNFs)
        target = generate_target(G, VNFs, source)

        while (source, target) in pairs:
            source = generate_source(G, VNFs)
            target = generate_target(G, VNFs, source)

        pairs.append((source, target))

        lengths.append(get_path_length(G, source, target, VNFs))

    return np.average(lengths), np.std(lengths)


def run_experiments(topologies, info):
    """Run the experiments on the parameters defined in the
    information struct.
    """
    total_avg1, total_avg2, total_avg3 = ({}, {}, {})
    total_std1, total_std2, total_std3 = ({}, {}, {})

    for topology in topologies:
        print("Running topology: " + topology + "...")
        avg1, avg2, avg3 = ([], [], [])
        std1, std2, std3 = ([], [], [])

        G = create_graph(topology)

        for i in range(info['topologies']):
            VNFs = []
            avg, std = run_cycles(G, info, VNFs)
            avg1.append(avg)
            std1.append(std)

            VNFs = generate_function_nodes(G, topology, info, True)
            avg, std = run_cycles(G, info, VNFs)
            avg2.append(avg)
            std2.append(std)

            VNFs = generate_function_nodes(G, topology, info, False)
            avg, std = run_cycles(G, info, VNFs)
            avg3.append(avg)
            std3.append(std)

        total_avg1[topology] = np.average(avg1)
        total_avg2[topology] = np.average(avg2)
        total_avg3[topology] = np.average(avg3)

        total_std1[topology] = np.average(std1)
        total_std2[topology] = np.average(std2)
        total_std3[topology] = np.average(std3)

    return (total_avg1, total_std1,
            total_avg2, total_std2,
            total_avg3, total_std3)


def visualise_probabilities(topologies, info, results):
    """Visualise the results provided in a bar chart.
    """
    fig, ax = plt.subplots()
    index = np.arange(len(topologies))
    bar_width = 0.25
    opacity = 0.8

    avgs_0 = np.array(list(results[0].values()))
    tmp_avgs_1 = np.array(list(results[2].values()))
    tmp_avgs_2 = np.array(list(results[4].values()))

    stds_0 = np.array(list(results[1].values()))
    stds_1 = np.array(list(results[3].values()))
    stds_2 = np.array(list(results[5].values()))

    avgs_1 = tmp_avgs_1 - avgs_0
    avgs_2 = tmp_avgs_2 - avgs_0

    avgs_1 /= avgs_0
    avgs_2 /= avgs_0

    scale_1 = avgs_1 / tmp_avgs_1
    stds_1 *= scale_1
    scale_2 = avgs_2 / tmp_avgs_2
    stds_2 *= scale_2

    y_max = 1
    if np.amax(avgs_1) > y_max:
        y_max = np.amax(avgs_1)
    if np.amax(avgs_2) > y_max:
        y_max = np.amax(avgs_2)

    y_min = 0
    if np.amin(avgs_1) < y_min:
        y_min = np.amin(avgs_1)
    if np.amin(avgs_2) < y_min:
        y_min = np.amin(avgs_2)

    y_max += 0.25
    if y_min is not 0:
        y_min -= 0.25

    # Plot the first plot (random and non-random, small sized topology).
    rects1 = ax.bar(index + (bar_width * 1),
                    avgs_1,
                    bar_width,
                    alpha=opacity,
                    color='blue',
                    label='Random VNF allocation',
                    yerr=stds_1)

    rects2 = ax.bar(index + (bar_width * 2),
                    avgs_2,
                    bar_width,
                    alpha=opacity,
                    color='green',
                    label='Topology-aware VNF allocation',
                    yerr=stds_2)

    # Set the title data.
    plt.title('Relative increase in path length over ' +
              '{} VNF(s)'.format(info['num_VNFs']))
    plt.ylabel('Performance impact relative to the shortest path')
    plt.xlabel('Topology')
    plt.ylim([y_min, y_max])

    plt.legend()

    plt.xticks(index + (bar_width * 1.5), topologies)

    # plt.tight_layout()
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
        'cycles':      1000
    }

    # Define the topologies experimented upon.
    topologies = ['NORDUnet', 'GEANT', 'SURFnet']

    # Read data from command line.
    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hs:m:l:V:t:c:',
                                       ['help',
                                        'num_VNFs=',
                                        'topologies=',
                                        'cycles='])

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

    # Relay parameters to user.
    print("Running " + str(information['cycles']) + " cycles on " +
          str(information['topologies']) + " topologies...")

    # Perform experiments.
    results = run_experiments(topologies, information)
    visualise_probabilities(topologies, information, results)


if __name__ == "__main__":
    main()
