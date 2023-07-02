# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:00:08 2020

@author: Tanzeela Javid
"""
import copy
import sys
import time

from sklearn.decomposition import PCA
from decimal import Decimal
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from rss_algo import rss_rmd_sum

from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()
plot_points = False


def calculate_distance(const_x, const_y, const_z, x, y, z):
    dist = (const_x - x) * (const_x - x) + (const_y - y) * (const_y - y) + (const_z - z) * (const_z - z)
    return math.sqrt(dist)


def get_distance(p1, p2):
    dist = 0
    for a, b in zip(p1, p2):
        a, b = Decimal(float(a)), Decimal(float(b))
        dist += ((a - b) ** 2)

    return math.sqrt(dist)


# Checks if all centroids are identical
def is_identical(all_cluster1, all_cluster2):
    try:
        assert len(all_cluster1) == len(all_cluster2), "client_cluster len mismatch"
        for cluster1, cluster2 in zip(all_cluster1, all_cluster2):
            msg = "Cluster1: {}, Cluster2: {}".format(cluster1, cluster2)
            assert len(cluster1) == len(cluster2), msg + "cluster len mismatch"
            for point1, point2 in zip(cluster1, cluster2):
                assert point1 == point2, "Point1: {}, Point2: {} mismatch".format(point1, point2)
    except AssertionError as e:
        # print("Error: {}".format(e))  # This is for debugging
        return False
    return True


# Make sure that number of clusters doesn't exceed the total number of points
# of all clients
# Note clients_data should be list of list
def k_means_algorithm(k, clients_data, sum_method):
    point_dimension = len(clients_data[0][0])
    num_clients = len(clients_data)
    centroids = []
    # Taking random centroids
    for i in range(k):
        centroids.append(clients_data[i % num_clients][int(i / num_clients)])

    num_loops = 0  # Can be used to check the number of loops it ran
    prev_all_clusters = None
    while True:
        num_loops += 1

        # Dividing all the points into clusters
        client_clusters = [[list() for _ in range(k)] for _ in range(num_clients)]
        for i, client_i in enumerate(clients_data):
            for point in client_i:
                distance_from_centroids = list(map(lambda x: get_distance(point, x), centroids))
                nearest_centroid = min(enumerate(distance_from_centroids), key=lambda x: x[1])
                nearest_centroid_index = nearest_centroid[0]
                client_clusters[i][nearest_centroid_index].append(point)

        # Merge all client's clusters
        all_clusters = []
        for i in range(k):
            cluster_i_all_points = []
            for j in range(num_clients):
                cluster_i_all_points += client_clusters[j][i]
            all_clusters.append(cluster_i_all_points)

        # Check if any cluster is changed
        if prev_all_clusters and is_identical(all_clusters, prev_all_clusters):
            # Nothing got changed we should stop now.
            print("Loops ran for reaching to final centroids: {}".format(num_loops))
            return centroids, all_clusters

        # Recalculate the centroids
        new_centroids = []
        for i in range(k):
            cluster_i_all_points = all_clusters[i]

            total_points = len(cluster_i_all_points)
            if total_points == 0:
                continue

            # RSS technique for calculating ratio (average)
            one_points = [1] * total_points
            rss_sum_for_ones = sum_method(one_points)

            new_centroid_i = []
            for dim_k in range(point_dimension):
                dim_points = list(map(lambda x: x[dim_k], cluster_i_all_points))
                rss_sum_for_dim_k = sum_method(dim_points)
                rss_ratio = rss_sum_for_dim_k / rss_sum_for_ones
                # simple_ratio = sum(dim_points) / total_points

                new_centroid_i.append(rss_ratio)

                # print("RSS Ratio: {}".format(rss_ratio))
                # print("Simple Ratio: {}".format(simple_ratio))
                # print("--------")

            new_centroids.append(new_centroid_i)

        # following commented lines are for debugging
        # plt.clf()
        # all_points = []
        # markers = ['g', 'r']
        # for i, cluster in enumerate(all_clusters):
        #     plt.scatter(list(map(lambda x: x[0], cluster)),
        #                 list(map(lambda x: x[1], cluster)), color=markers[i])
        #     plt.scatter([centroids[i][0]], [centroids[i][1]],
        #                 marker='x', color=markers[i])
        #     plt.scatter([new_centroids[i][0]], [new_centroids[i][1]],
        #                 marker='v', color=markers[i])

        # centers = np.array(centroids)
        # plt.scatter(centers[:, 0], centers[:, 1], marker="x", color='r')
        #
        # centers = np.array(new_centroids)
        # plt.scatter(centers[:, 0], centers[:, 1], marker="v", color='r')

        # plt.show()

        # Update clients data
        new_clients_data = []
        for i in range(num_clients):
            client_i_clusters = client_clusters[i]
            client_i_points = []
            for cluster_points in client_i_clusters:
                client_i_points += cluster_points
            new_clients_data.append(client_i_points)

        centroids = copy.deepcopy(new_centroids)
        clients_data = copy.deepcopy(new_clients_data)
        # using this we will check if any cluster is changed
        prev_all_clusters = copy.deepcopy(all_clusters)
        print("Loop Count: {}".format(num_loops))


# This function takes an array and divides it into 3 equal arrays
def split_into_n(arr, parts):
    arr_len = len(arr)
    each_part_size = int(arr_len / parts)
    start = 0
    parts_arr = []
    for part_i in range(parts):
        parts_arr.append(arr[start:start + each_part_size])
        start += each_part_size
    return parts_arr


def get_data(filename, num_points):
    df = pd.read_csv(filename, delimiter=',')

    arr = df.select_dtypes(['number']).to_numpy()
    arr = list(map(tuple, arr))
    if num_points:
        arr = arr[:num_points]
    pca.fit(arr)

    clients = 3  # Update this variable to test for any number of clients
    all_clients_data = split_into_n(arr, clients)
    return all_clients_data


def main():
    ##########------------Reading columns from csv--------------

    #################### I AM TAKING THREE COLUMNS ONLYYYYY AGE, JOB, MARITAL######################

    sum_methods = {
        'rss': rss_rmd_sum,
        'simple': sum
    }

    file_name = input('Enter input csv filename:\n')
    num_points = input('Enter number of points to run, (enter 0 for using complete file):\n').strip()
    num_points = int(num_points)
    num_clusters = int(input("Enter the number of clusters you want to create:\n").strip())

    all_clients_data = get_data(file_name, num_points)
    print("Starting K-MEANS with RSS Method \n")
    rss_start_time = time.time()
    final_centroids1, final_clusters1 = k_means_algorithm(num_clusters, copy.deepcopy(all_clients_data), rss_rmd_sum)
    rss_end_time = time.time()
    rss_time = rss_end_time - rss_start_time
    print("RSS Method completed in {} seconds \n".format(rss_time))

    print("Starting K-MEANS with Simple Method \n")
    simple_start_time = time.time()
    final_centroids2, final_clusters2 = k_means_algorithm(num_clusters, copy.deepcopy(all_clients_data), sum)
    simple_end_time = time.time()
    simple_time = simple_end_time - simple_start_time
    print("Simple Method completed in {} seconds \n".format(simple_time))

    print("Difference in execution time of simple k-means algorithm and k-means algorithm withh RSS : {} seconds".format(rss_time - simple_time))

    error = []
    accuracy = []
    total_points = 0
    for cluster_rss, cluster_simple in zip(final_clusters1, final_clusters2):
        cluster_simple = set(cluster_simple)
        cluster_rss = set(cluster_rss)

        error.append(len(cluster_simple - cluster_rss))
        accuracy.append(len(cluster_simple & cluster_rss))
        total_points += len(cluster_simple)

    print("Accuracy of the K-Means Algorithm : {} \n".format((float(sum(accuracy)) / total_points) * 100))
    print("Error : {} \n".format(sum(error) / len(error)))

    plot_multiple(final_centroids1, final_clusters1, final_centroids2, final_clusters2)


# this is not working now
# def test():
#     import os
#     # change the directory name here
#     if not os.path.exists('test_result'):
#         os.mkdir('test_result')
#     all_clients_data = get_data()
#
#     # update these two variables it will put the results in test_results directory
#     min_k = 3
#     max_k = 6
#
#     for k in range(min_k, max_k + 1):
#         print("Running for k: {}".format(k))
#         final_centroids, final_clusters = k_means_algorithm(k, all_clients_data, rss_rmd_sum)
#         plot_figure(final_centroids, final_clusters, 'test_result/fig_method_rss_k_{}.png'
#                     .format(k))
#         final_centroids, final_clusters = k_means_algorithm(k, all_clients_data, sum)
#         plot_figure(final_centroids, final_clusters, 'test_result/fig_method_simple_k_{}.png'
#                     .format(k))
#         print("Done with k: {}".format(k))

def plot_multiple(final_centroids_rss, final_clusters_rss, final_centroids_simple,
                  final_clusters_simple):
    fig = plt.figure(figsize=(12,5))
    ax0 = fig.add_subplot(1, 2, 1, projection='3d')
    plot_figure(final_centroids_rss, final_clusters_rss, fig=ax0, show=False)
    ax0.set_title('RSS Method')
    ax1 = fig.add_subplot(1, 2, 2, projection='3d')
    plot_figure(final_centroids_simple, final_clusters_simple, fig=ax1, show=False)
    ax1.set_title('Simple Method')
    plt.show()


def plot_figure(final_centroids, final_clusters, file_name=None, fig=None, show=True):
    # getting the colors
    colors = list(cm.colors.BASE_COLORS.values())
    fig = fig or plt.figure()

    centers = np.array(final_centroids)
    centroid_index = 0
    for i, cluster in enumerate(final_clusters):
        if not cluster:
            continue
        transformed = np.array(pca.transform(np.array(cluster)))

        fig.scatter(transformed[:, 0],
                    transformed[:, 1],
                    transformed[:, 2],
                    marker='o', color=colors[i])

        center = centers[centroid_index]
        center = pca.transform(np.array([center]))

        fig.scatter(float(center[0][0]),
                    float(center[0][1]),
                    float(center[0][2]),
                    marker="x",
                    color=colors[i],
                    s=80)
        centroid_index += 1

    # plt.legend()

    # print("Final clusters: {}".format(centroid_index))

    # axes_3d.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker="x", color='r')

    # Either save to file or show to user
    if file_name:
        plt.savefig(file_name)
    if show:
        plt.show()


if __name__ == '__main__':
    pca = PCA(n_components=3)
    np.set_printoptions(formatter={'float_kind': lambda x: "%.9f" % x})
    np.seterr(divide='ignore', invalid='ignore')
    main()
    # test() # Check this function for testing the code for different values of k
