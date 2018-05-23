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
    elif topology == 'circular ladder':
        return nx.circular_ladder_graph(info['num_nodes'])
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

        # Caluclate most "middle" nodes in network and place VNF there.
        if topology == 'circular ladder':
            for i in range(info['num_VNFs']):
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


def run_experiments(topologies, sizes, info):
    """Run the experiments on the parameters defined in the
    information struct.
    """
    final_avg1, final_avg2, final_avg3 = ({}, {}, {})
    final_std1, final_std2, final_std3 = ({}, {}, {})

    for size in sizes:
        total_avg1, total_avg2, total_avg3 = ({}, {}, {})
        total_std1, total_std2, total_std3 = ({}, {}, {})

        info['num_nodes'] = size

        for topology in topologies:
            print("Running topology: " + topology + "...")
            avg1, avg2, avg3 = ([], [], [])
            std1, std2, std3 = ([], [], [])

            G = create_graph(topology, info)

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

        final_avg1[size] = total_avg1
        final_avg2[size] = total_avg2
        final_avg3[size] = total_avg3

        final_std1[size] = total_std1
        final_std2[size] = total_std2
        final_std3[size] = total_std3

    return (final_avg1, final_std1,
            final_avg2, final_std2,
            final_avg3, final_std3)


def visualise_probabilities(topologies, info, results):
    """Visualise the results provided in a bar chart.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    index = np.arange(len(topologies))
    bar_width = 0.25
    opacity = 0.8

    final_avg1, final_std1 = results[0], results[1]
    final_avg2, final_std2 = results[2], results[3]
    final_avg3, final_std3 = results[4], results[5]

    # Plot the first plot (random and non-random, small sized topology).
    rects1 = ax1.bar(index + (bar_width * 0),
                     list(final_avg1[info['small_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='red',
                     label='No VNF allocation',
                     yerr=list(final_std1[info['small_size']].values()))

    rects2 = ax1.bar(index + (bar_width * 1),
                     list(final_avg2[info['small_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='blue',
                     label='Random VNF allocation',
                     yerr=list(final_std2[info['small_size']].values()))

    rects3 = ax1.bar(index + (bar_width * 2),
                     list(final_avg3[info['small_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='green',
                     label='Topology-aware VNF allocation',
                     yerr=list(final_std3[info['small_size']].values()))

    # Plot the first plot (random and non-random, small sized topology).
    rects4 = ax2.bar(index + (bar_width * 0),
                     list(final_avg1[info['medium_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='red',
                     label='No VNF allocation',
                     yerr=list(final_std1[info['medium_size']].values()))

    rects5 = ax2.bar(index + (bar_width * 1),
                     list(final_avg2[info['medium_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='blue',
                     label='Random VNF allocation',
                     yerr=list(final_std2[info['medium_size']].values()))

    rects6 = ax2.bar(index + (bar_width * 2),
                     list(final_avg3[info['medium_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='green',
                     label='Topology-aware VNF allocation',
                     yerr=list(final_std3[info['medium_size']].values()))

    # Plot the first plot (random and non-random, small sized topology).
    rects7 = ax3.bar(index + (bar_width * 0),
                     list(final_avg1[info['large_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='red',
                     label='No VNF allocation',
                     yerr=list(final_std1[info['large_size']].values()))

    rects8 = ax3.bar(index + (bar_width * 1),
                     list(final_avg2[info['large_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='blue',
                     label='Random VNF allocation',
                     yerr=list(final_std2[info['large_size']].values()))

    rects9 = ax3.bar(index + (bar_width * 2),
                     list(final_avg3[info['large_size']].values()),
                     bar_width,
                     alpha=opacity,
                     color='green',
                     label='Topology-aware VNF allocation',
                     yerr=list(final_std3[info['large_size']].values()))

    # Set the title data.
    ax1.set(title='Path lengths from source to target over {} VNF(s)'.
            format(info['num_VNFs']))
    ax1.set(ylabel='Path length')
    ax1.set(xlabel='{} node topology'.format(info['small_size']))
    ax1.set(xticks=index + bar_width)
    ax1.set(xticklabels=topologies)
    ax1.legend(loc=0)

    ax2.set(ylabel='Path length')
    ax2.set(xlabel='{} node topology'.format(info['medium_size']))
    ax2.set(xticks=index + bar_width)
    ax2.set(xticklabels=topologies)
    ax2.legend(loc=0)

    ax3.set(ylabel='Path length')
    ax3.set(xlabel='{} node topology'.format(info['large_size']))
    ax3.set(xticks=index + bar_width)
    ax3.set(xticklabels=topologies)
    ax3.legend(loc=0)

    plt.tight_layout()
    plt.show()


def main():
    """Runs the main function.
    Reads the command line information provided and calls the experiments.
    """

    # Define the default data.
    information = {
        'small_size':  10,
        'medium_size': 25,
        'large_size':  50,
        'VNF':         0,
        'num_VNFs':        1,
        'topologies':  1000,
        'cycles':      1000,
        'prob1':       0.5,
        'prob2':       0.5,
        'avg_degree':  3
    }

    # Define the topologies experimented upon.
    topologies = ['simple', 'star', 'tree', 'circular ladder']

    # Read data from command line.
    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hs:m:l:V:t:c:',
                                       ['help',
                                        'small_size=',
                                        'medium_size=',
                                        'large_size=',
                                        'num_VNFs=',
                                        'topologies=',
                                        'cycles='])

    for opt, arg in options:
        if opt in ('-h', '--help'):
            print(help_message)
            exit(0)
        elif opt in ('-s', '--small_size'):
            information['small_size'] = int(arg)
        elif opt in ('-m', '--medium_size'):
            information['medium_size'] = int(arg)
        elif opt in ('-l', '--large_size'):
            information['large_size'] = int(arg)
        elif opt in ('-V', '--num_VNFs'):
            information['num_VNFs'] = int(arg)
        elif opt in ('-t', '--topologies'):
            information['topologies'] = int(arg)
        elif opt in ('-c', '--cycles'):
            information['cycles'] = int(arg)

    # Relay parameters to user.
    print("Running " + str(information['cycles']) + " cycles on " +
          str(information['topologies']) + " topologies of sizes: " +
          str(information['small_size']) + " nodes, " +
          str(information['medium_size']) + " nodes, and " +
          str(information['large_size']) + " nodes...")

    # Combine sizes.
    sizes = [information['small_size'],
             information['medium_size'],
             information['large_size']]

    # Perform experiments.
    results = run_experiments(topologies, sizes, information)
    visualise_probabilities(topologies, information, results)


if __name__ == "__main__":
    main()
