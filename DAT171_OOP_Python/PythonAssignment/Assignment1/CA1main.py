import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from matplotlib.collections import LineCollection
import scipy.sparse.csgraph as cs
from scipy.spatial import cKDTree
import os

filepath = os.path.dirname(os.path.abspath(__file__))

# Choose .txt-file, radius and start/end-city here
# ---------------------------------------------------------------------------
# FILENAME, RADIUS, STARTCITY, ENDCITY = cd + '/SampleCoordinates.txt', 0.08, 0, 5
# FILENAME, RADIUS, STARTCITY, ENDCITY =  cd + '/HungaryCities.txt', 0.005, 311, 702
FILENAME, RADIUS, STARTCITY, ENDCITY = filepath + '/GermanyCities.txt', 0.0025, 1573, 10584

def read_coordinate_file(filename):
    """
    Task 1: Function that reads the given coordinate file and parses the results into an array of coordinates

    :param filename: .txt file with coordinates in (long, lat) separated by ',' and contained by '{}'
    :type filename: str

    :return: numpy array with coordinates in cartesian
    """

    file = open(filename, 'r')                                          # Opens the .txt-file in read-mode

    coord = []

    for line in file:
        line_strip = line.strip(' {}\n')                        # Strips the .txt-file of the str-objects we're
        line_split = line_strip.split(',')
        x = np.pi * float(line_split[1]) / 180                          # Calculates coordinates from lat/long.
        y = np.log(np.tan(np.pi/4 + np.pi*float(line_split[0])/360))    # to xy-plane
        coord.append([x, y])
    file.close()

    coord_array = np.array(coord)
    return coord_array


def plot_points(coord_list):
    """
    Task 2: Function that plots the data points read from the file

    :param coord_list: numpy array of coordinates
    :type coord_list: numpy.array

    :return: scatter plot of data points
    """

    x = coord_list[:, 0]    # Separates the x and y coordinates
    y = coord_list[:, 1]

    plt.scatter(x, y)   # Plot data points in scatter
    plt.show()


def construct_graph_connections(coord_list, radius):
    """
    Task 3: Function that computes all the connections between all the points in the
        coordinates list that are within the radius given"
    
    :param coord_list: numpy array of coordinates
    :param radius: boundary condition for max distance for single travel
    :type coord_list: numpy.array 
    
    :return cost_array, indices: numpy array of cost for each travel possible and the indices for the travels
    """""

    cost_array = []  # Pre-define cost list
    indices = []    # and indice list

    for indice_start_city, coord_start_city in enumerate(coord_list):
        for indice_dest_city, coord_dest_city in enumerate(coord_list[indice_start_city+1:], start=indice_start_city+1):
            dist = math.sqrt(((coord_start_city[0] - coord_dest_city[0])**2 +
                              (coord_start_city[1] - coord_dest_city[1])**2))  # Calculates the distance between the points
            if dist < radius:
                cost = dist**(9/10)            # Calculate the cost of the trip
                cost_array.append(cost)        # add the cost and indices to list
                indices.append([indice_start_city, indice_dest_city])

    return np.array(cost_array), np.array(indices)


def construct_graph(indices, costs, N):
    """
    Task 4: Function that constructs a sparse graph. The graph should be represented
        as a compressed sparse row matrix, where N is number of travels

    :param indices: Numpy array of indices for the travels possible
    :param costs: Numpy array of cost for each travel
    :param N: Wanted shape of the sparse matrix

    :type N: int
    :type costs: list
    :type indices: list

    :return: Compressed sparse row matrix of dimension (N, N)
    """

    csr_cost = csr_matrix((costs, (indices[:, 0], indices[:, 1])), shape=(N, N))  # Creates a sparse-matrix

    return csr_cost


def plot_points5(coord_list, indices):
    """
    Task 5: Extend ​plot_points(coord_list​, indices​) ​to also include the graph connections from task 3

    :param coord_list: numpy array of coordinates
    :param indices: Numpy array of indices for the travels possible

    :type coord_list: numpy.array
    :type indices: numpy.array

    :return: Scatter plot of data points with possible travels between data points included
    """

    x = coord_list[:, 0]
    y = coord_list[:, 1]
    fig, ax = plt.subplots()
    plt.scatter(x, y, color='red', linewidth=0.1)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect('equal')

    lines = coord_list[indices]

    line_segments = LineCollection(lines, linewidth=0.1, color='black')  # Plot all lines in lines
    ax.add_collection(line_segments)

    plt.show()


def compute_path(sparse_graph, startpoint, endpoint):
    """
    Task 6/7: Uses dijkstra-algorithm to calculate the cheapest path from startPoint to endPoint. The dijkstra-func
    gives a distance vector and a predecessor vector. The predecessor vector shows the point we came from.

    :param sparse_graph: Sparse graph of possible travels
    :param startpoint: Start point of the travel
    :param endpoint: Where the travel should end

    :return path, costpath: list of indices for the cheapest path and the cost of the travel (float)
    """

    # Use dijkstra to get predecessor/cost vector.
    # sparse_graph is a parameter with all possible travels between points
    # indices sets stating city
    # return predecessors to True because we want calculate the cheapest path
    # "Directed" not necessary in this case, if false it finds the shortest path where the algorithm can
    # progress from point i to j along sparse_graph[i, j] or sparse_graph[j, i]

    cost_vector, predecessors = cs.dijkstra(sparse_graph, indices=startpoint, return_predecessors=True, directed=False)
    path = [endpoint]                   # Adds end point to path-list, would not be appended in loop
    cost_path = cost_vector[endpoint]
    node = endpoint

    while node != startpoint:           # Loops through predecessor vector "backwards" to find the cheapest path until
        node = predecessors[node]       # we arrive at the start city
        path.append(node)               # append previous visited node to path

    return list(reversed(path)), cost_path


def plot_points8(coord_list, indices, path):
    """
    Task 8: Extend ​plot_points(coord_list​, indices​) ​to also include the cheapest path from task 7

    :param coord_list: numpy array of coordinates
    :param indices: Numpy array of indices for the travels possible
    :param path: list of indices for the cheapest path possible

    :return: Scatter plot with all possible travels in gray, and the cheapest path available between start and end point
    """

    x = coord_list[:, 0]  # Separates the x and y coordinates
    y = coord_list[:, 1]
    fig, ax = plt.subplots()
    plt.scatter(x, y, color='red', s=0.3)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.set_aspect('equal')

    lines = coord_list[indices]
    linespath = coord_list[path]

    line_segments = LineCollection(lines, linewidth=0.1, color='black')
    ax.add_collection(line_segments)                                    # Plot lines in LineCollection
    ax.plot(linespath[:, 0], linespath[:, 1], color='blue')             # Plot cheapest path

    plt.show()


def construct_fast_graph_connections(coord_list, radius):
    """
    Task 10: Faster 'construct_graph_connections' function

    :param coord_list: numpy.array of coordinates
    :param radius: maximum distance for one individual travel
    :return cost_list, indices: numpy arrays of cost for each travel and indices for all the possible travels
    """

    x = coord_list[:, 0]
    y = coord_list[:, 1]

    tree = cKDTree(coord_list)                              # Create tree to make coord_list compatible with ball_point
    travels = tree.query_ball_point(coord_list, radius)     # Creates lists in list of all possible travels (indices)
    indices = []
    cost_list = []

    for indice_start_city, travel_list in enumerate(travels):  # Loops through travels to create list of ind, and calc
        for indice_dest_city in travel_list:                                     # cost, for possible travels
            if indice_start_city != indice_dest_city:
                dist = math.sqrt((x[indice_start_city] - x[indice_dest_city])**2
                                 + (y[indice_start_city] - y[indice_dest_city])**2)
                cost = dist**0.9
                indices.append([indice_start_city, indice_dest_city])
                cost_list.append(cost)

    return np.array(cost_list), np.array(indices)


pre_loop_time = time.time()
coordinates = read_coordinate_file(FILENAME)
readfileTime = time.time()

while True:
    answer = input('Fast or slow calculation of graph connections? ')
    if answer == ('fast' or 'FAST' or 'Fast'):
        loop_time = time.time()
        cost, indices = construct_fast_graph_connections(coordinates, RADIUS)
        break
    elif answer == ('slow' or 'SLOW' or 'Slow'):
        loop_time = time.time()
        cost, indices = construct_graph_connections(coordinates, RADIUS)
        break
    else:
        print('Invalid input, try again')


construct_graph_connectionsTime = time.time()

sparse_matrix = construct_graph(indices, cost, len(coordinates))

construct_graphTime = time.time()

cheapestPath, costPath = compute_path(sparse_matrix, STARTCITY, ENDCITY)

compute_pathTime = time.time()

plot_points8(coordinates, indices, cheapestPath)

plot_pointsTime = time.time()


print('read_coordinate_file: {} s' .format(readfileTime - pre_loop_time))
print('construct_graph_connections: {} s' .format(construct_graph_connectionsTime - loop_time))
print('construct_graph: {} s' .format(construct_graphTime - construct_graph_connectionsTime))
print('compute_path: {} s' .format(compute_pathTime - construct_graphTime))
print('plot_pointsTime: {} s' .format(plot_pointsTime - compute_pathTime))
print('Total run time: {} s' .format(plot_pointsTime - loop_time + readfileTime - pre_loop_time))
print()
print('Cheapest path: {} \ntotal cost {}' .format(cheapestPath, costPath))


