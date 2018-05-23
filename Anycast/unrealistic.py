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


def create_graph(topology, info):
    """Create a graph depending on the information in the
    info struct provided.
    """
    if topology == 'simple':
        return nx.barabasi_albert_graph(info['num_nodes'],
                                        info['avg_degree'])
    elif topology == 'star':
        return nx.star_graph(info['num_nodes'])
    elif topology == 'tree':
        return nx.random_tree(info['num_nodes'])
    elif topology == 'ladder':
        return nx.ladder_graph(int(round(info['num_nodes'] / 2)))
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

        # Caluclate most "middle" nodes in network and place VNF there.
        if topology == 'ladder':
            for i in range(info['num_VNFs']):
                num = int(round(info['num_nodes'] / 4))

                j, tmp = (0, num)

                while tmp in function_nodes:
                    if j % 2 is 0:
                        tmp = num + j
                    else:
                        tmp = num - j
                    j += 1

                if num not in G.nodes():
                    num = np.random.choice(list(G.nodes()))

                    while tmp in function_nodes:
                        num = np.random.choice(list(G.nodes()))

                function_nodes.append(num)

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
        closest = find_closest(G, source, VNFs)
        if closest is -1:
            continue
        lengths.append(get_path_length(G, source, target, closest))

    return np.average(lengths), np.std(lengths)


def run_experiments(topologies, info):
    """Run the experiments on the parameters defined in the
    information struct.
    """
    total_avg, total_std = {}, {}

    for topology in topologies:
        print("Running topology: " + topology + "...")

        avg1, std1 = [], []

        for i in range(info['topologies']):
            G = create_graph(topology, info)
            avg2, std2 = [], []

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
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    total_avg = list(results[0].values())
    total_std = list(results[1].values())

    x_vals0 = np.arange(1, len(total_avg[0]) + 1)
    x_vals1 = np.arange(1, len(total_avg[1]) + 1)
    x_vals2 = np.arange(1, len(total_avg[2]) + 1)
    x_vals3 = np.arange(1, len(total_avg[3]) + 1)

    y_vals0 = list(total_avg[0])
    y_vals1 = list(total_avg[1])
    y_vals2 = list(total_avg[2])
    y_vals3 = list(total_avg[3])

    y_err0 = list(total_std[0])
    y_err1 = list(total_std[1])
    y_err2 = list(total_std[2])
    y_err3 = list(total_std[3])

    y_max = max(y_vals0)
    if max(y_vals1) > y_max:
        y_max = max(y_vals1)
    if max(y_vals2) > y_max:
        y_max = max(y_vals2)
    if max(y_vals3) > y_max:
        y_max = max(y_vals3)

    y_min = min(y_vals0)
    if min(y_vals1) < y_min:
        y_min = min(y_vals1)
    if min(y_vals2) < y_min:
        y_min = min(y_vals2)
    if min(y_vals3) < y_min:
        y_min = min(y_vals3)

    y_incr = max(y_err0)
    if max(y_err1) > y_incr:
        y_incr = max(y_err1)
    if max(y_err2) > y_incr:
        y_incr = max(y_err2)
    if max(y_err3) > y_incr:
        y_incr = max(y_err3)

    y_max += y_incr + 0.5
    y_min -= y_incr - 0.5

    # Plot the first plot (random and non-random, small sized topology).
    plot0 = ax1.errorbar(x_vals0, y_vals0,
                         yerr=y_err0,
                         label='Simple',
                         color='red',
                         fmt='-o',
                         capthick=1,
                         capsize=2,
                         elinewidth=1,
                         markeredgewidth=1)
    plot1 = ax2.errorbar(x_vals1, y_vals1,
                         yerr=y_err1,
                         label='Star',
                         color='blue',
                         fmt='-o',
                         capthick=1,
                         capsize=2,
                         elinewidth=1,
                         markeredgewidth=1)
    plot2 = ax3.errorbar(x_vals2, y_vals2,
                         yerr=y_err2,
                         label='Tree',
                         color='green',
                         fmt='-o',
                         capthick=1,
                         capsize=2,
                         elinewidth=1,
                         markeredgewidth=1)
    plot2 = ax4.errorbar(x_vals3, y_vals3,
                         yerr=y_err3,
                         label='ladder',
                         color='purple',
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

    ax4.set(ylabel='Path length')
    ax4.set(xlabel='Number of VNFs')
    ax4.set(ylim=[y_min, y_max])
    ax4.legend(loc=0)

    # Set the title data.
    if info['random_VNFs']:
        fig.suptitle('Path lengths from source to target over randomly ' +
                     'placed VNF(s)')
    else:
        fig.suptitle('Path lengths from source to target over ' +
                     'topology-aware placed VNF(s)')

    # plt.tight_layout()
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
        'free_nodes':  0,
        'prob1':       0.5,
        'prob2':       0.5,
        'avg_degree':  3,
        'num_nodes':   30
    }

    # Define the number of nodes needed to be left free.
    free_nodes = 0

    # Define the topologies experimented upon.
    topologies = ['simple', 'star', 'tree', 'ladder']

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
