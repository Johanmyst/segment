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
        for i in range(info['VNFs']):
            tmp = random.randint(0, info['num_nodes'] - 1)

            while tmp in function_nodes:
                tmp = random.randint(0, info['num_nodes'] - 1)

            function_nodes.append(tmp)
    else:
        # Calculate most connected nodes.
        degrees = list(G.degree())
        degree_sequence = sorted(degrees, key=lambda tup: tup[1], reverse=True)
        if len(degree_sequence) is 0:
            return []

        # Caluclate most "middle" nodes in network and place VNF there.
        if topology == 'ladder':
            for i in range(info['VNFs']):
                num = int(round(info['num_nodes'] / 4))

                j, tmp = (0, num)
                while tmp in function_nodes:
                    if j % 2 is 0:
                        tmp = num + j
                    else:
                        tmp = num - j
                    j += 1

                num = tmp

                while num not in G.nodes():
                    num = random.randint(0, info['num_nodes'] - 1)

                function_nodes.append(num)

        # Else just pick the most connected nodes.
        else:
            for i in range(info['VNFs']):
                function_nodes.append(degree_sequence[i][0])

    return function_nodes


def generate_source(info):
    """Generate a random source node. Can't be the VNF.
    """
    tmp = random.randint(0, info['num_nodes'] - 1)

    while tmp in info['curr_VNFs']:
        tmp = random.randint(0, info['num_nodes'] - 1)

    return tmp


def generate_target(info, source):
    """Generate a random target node. Can't be the VNF nor the source.
    """
    tmp = random.randint(0, info['num_nodes'] - 1)

    while tmp in info['curr_VNFs'] or tmp is source:
        tmp = random.randint(0, info['num_nodes'] - 1)

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
        source = generate_source(info)
        target = generate_target(info, source)

        # Store the pair to make sure it's not re-used.
        pairs.append((source, target))

        # While the pair is non-unique, generate a new pair.
        while (source, target) in pairs:
            source = generate_source(info)
            target = generate_target(info, source)

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
    total_random_prob = {}
    total_random_std = {}
    total_non_random_prob = {}
    total_non_random_std = {}

    # Saves some computation.
    divisor = (info['topologies'] * info['cycles'])

    # Run the three sizes.
    for size in info['sizes']:
        info['num_nodes'] = size

        # Initialise the data structs.
        random_prob = {}
        random_std = {}
        non_random_prob = {}
        non_random_std = {}

        # Run each topology.
        for topology in topologies:
            # Update user on progress.
            print("Running topology: " + str(topology)
                  + " on " + str(size) + " nodes...")

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

        # Store data.
        total_random_prob[size] = random_prob
        total_random_std[size] = random_std

        total_non_random_prob[size] = non_random_prob
        total_non_random_std[size] = non_random_std

    return (total_random_prob, total_random_std,
            total_non_random_prob, total_non_random_std)


def visualise_probabilities(topologies, info, probs):
    """Visualise the probs provide in a bar chart.
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    index = np.arange(len(topologies))
    bar_width = 0.35
    opacity = 0.8

    # Cast the random data into lists.
    random_small = list(probs[0][info['small_size']].values())
    random_medium = list(probs[0][info['medium_size']].values())
    random_large = list(probs[0][info['large_size']].values())

    # Calculate the standard deviation (used for error bars).
    random_small_std = list(probs[1][info['small_size']].values())
    random_medium_std = list(probs[1][info['medium_size']].values())
    random_large_std = list(probs[1][info['large_size']].values())

    # Cast the non-random data into lists.
    non_random_small = list(probs[2][info['small_size']].values())
    non_random_medium = list(probs[2][info['medium_size']].values())
    non_random_large = list(probs[2][info['large_size']].values())

    # Calculate the standard deviation (used for error bars).
    non_random_small_std = list(probs[3][info['small_size']].values())
    non_random_medium_std = list(probs[3][info['medium_size']].values())
    non_random_large_std = list(probs[3][info['large_size']].values())

    # Plot the first plot (random and non-random, small sized topology).
    rects1 = ax1.bar(index, random_small, bar_width,
                     alpha=opacity,
                     color='blue',
                     label='Random VNF Placement',
                     yerr=random_small_std)

    rects2 = ax1.bar(index + bar_width, non_random_small, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Topology-aware VNF placement',
                     yerr=non_random_small_std)

    # Plot the second plot (random and non-random, medium sized topology).
    rects1 = ax2.bar(index, random_medium, bar_width,
                     alpha=opacity,
                     color='blue',
                     label='Random VNF Placement',
                     yerr=random_medium_std)

    rects2 = ax2.bar(index + bar_width, non_random_medium, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Topology-aware VNF placement',
                     yerr=non_random_medium_std)

    # Plot the third plot (random and non-random, large sized topology).
    rects1 = ax3.bar(index, random_large, bar_width,
                     alpha=opacity,
                     color='blue',
                     label='Random VNF Placement',
                     yerr=random_large_std)

    rects2 = ax3.bar(index + bar_width, non_random_large, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Topology-aware VNF placement',
                     yerr=non_random_large_std)

    # Set the title data.
    ax1.set(title='Probability of VNF traversal over {} VNF(s)'.
            format(info['VNFs']))
    ax1.set(ylabel='Probability')
    ax1.set(xlabel='{} node topology'.format(info['small_size']))
    ax1.set(xticks=index + (bar_width / 2))
    ax1.set(xticklabels=topologies)
    ax1.legend(loc=0)
    ax1.set(ylim=(0, 1))

    ax2.set(ylabel='Probability')
    ax2.set(xlabel='{} node topology'.format(info['medium_size']))
    ax2.set(xticks=index + (bar_width / 2))
    ax2.set(xticklabels=topologies)
    ax2.legend(loc=0)
    ax2.set(ylim=(0, 1))

    ax3.set(ylabel='Probability')
    ax3.set(xlabel='{} node topology'.format(info['large_size']))
    ax3.set(xticks=index + (bar_width / 2))
    ax3.set(xticklabels=topologies)
    ax3.legend(loc=0)
    ax3.set(ylim=(0, 1))

    plt.tight_layout()
    plt.show()


def main():
    """Runs the main function.
    Reads the command line information provided and calls the experiments.
    """

    # Define the default data.
    information = {
        'small_size':  15,
        'medium_size': 30,
        'large_size':  60,
        'VNF':         0,
        'VNFs':        1,
        'topologies':  1000,
        'cycles':      1000,
        'prob1':       0.5,
        'prob2':       0.5,
        'avg_degree':  3
    }

    # Define the topologies experimented upon.
    topologies = ['simple', 'star', 'tree', 'ladder']

    # Read data from command line.
    options, remainder = getopt.getopt(sys.argv[1:],
                                       'hs:m:l:V:t:c:',
                                       ['help',
                                        'small_size=',
                                        'medium_size=',
                                        'large_size=',
                                        'VNFs=',
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
        elif opt in ('-V', '--VNFs'):
            information['VNFs'] = int(arg)
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
    information['sizes'] = [information['small_size'],
                            information['medium_size'],
                            information['large_size']]

    # Perform experiments.
    probabilities = run_experiments(topologies, information)
    visualise_probabilities(topologies, information, probabilities)


if __name__ == "__main__":
    main()
