# This file contains the functions developed for single linkage clustering algorithm
import sys
import math
import random
import time
import numpy as np
import heapq as hp

def load_file():

    #load_file function reads the input file to a list and determines the number of clusters(k) based on the number of classes in the target attribute.
    #This function takes the first argument in the command line as an input and returns the dataset(list) and the number of clusters as the output.

    dataset = list()
    target = list()
    f = open(sys.argv[1],"r")
    for line in f:
    #each element of the list is a datapoint
        line = line.strip('\n')
        row = str(line)
        row = row.split(",")
        data = row[:-1]
        output = row[-1]
        dataset.append(map(float, data))
        target.append(output)
    
    #set function removes the duplicate values of the target class
    distinct_set = set(target)
    distinct = list(distinct_set)
    k = len(distinct)
    
    #creating the target_truth cluster from the last column of the file.
    target_truth = [distinct.index(x) for x in target]

    return dataset,k,target_truth

def euclidean_distance(datapoint1, datapoint2):

    #This function returns the pairwise distance between any 2 points in the Euclidean space.
    #It takes two data points as the input and returns the distance between them.
    #If (x1,y1) and (x2,y2) are two points, then the function returns sqrt((x2-x1)^2 + (y2-y1)^2)
    
    distance = 0.0
    dimensions = len(datapoint1)
    for i in range(dimensions):
        x = float(datapoint2[i]) - float(datapoint1[i])
        distance += float(x ** 2)

    distance = math.sqrt(distance)

    return distance

def pairwise_distance(X,Y):
    
    #This function calculates the pair wise distance between all the point in the dataset. It uses the euclidean_distance function for calculating the distance.
    # It returns the list of distances and the corresponding pair of datapoints
    
    #if Y is none, then the functions calculates pairwise_distance(X,X)
    if Y is None:
        n = len(X)
        data = np.array(X)
        data_norm = np.array([np.sum(data*data, axis=1),]*n)
        data_data = np.dot(data, np.transpose(data))

        distances = data_norm.transpose()+data_norm-2*data_data

    else: #if 2 datasets are mentioned, it calculates pair wise distances between the points across 2 arrays
        n1 = len(X)
        n2 = len(Y)
        data1 = np.array(X)
        data2 = np.array(Y)
        data_norm1 = np.array([np.sum(data1*data1, axis=1),]*n2)
        data_norm2 = np.array([np.sum(data2*data2, axis=1),]*n1)
        data1_data2 = np.dot(data1, np.transpose(data2))

        distances = data_norm1.transpose()+data_norm2-2*data1_data2

    distances = np.sqrt(np.absolute(distances))
    return distances

def single_linkage(cluster1,cluster2,distances):
    
    #This function returns the distance between two clusters using single linkage algorithm.
    #Single linkage algorithm picks the smallest distance between all the datapoints in two clusters
    #Inputs are the two clusters and the list of pairwise distances. Returns the single linkage distance as the output.
    
    min = sys.maxint
    #calculates all possible pairwise distances between the 2 clusters and picks the minimum value
    for i in range(len(cluster1)):
        point1 = cluster1[i]
        for j in range(len(cluster2)):
            point2 = cluster2[j]
            value = distances[point1][point2]
            if value < min:
                min = value
    
    return min


def complete_linkage(cluster1,cluster2,distances):
    
    #This function returns the distance between two clusters using complete linkage algorithm.
    #complete linkage algorithm picks the maximum distance between all pairs of datapoints in two clusters
    #Inputs are the two clusters and the list of pairwise distances. Returns the complete linkage distance as the output.
    
    max = 0
    #calculates all possible pairwise distances between the 2 clusters and picks the maximum value
    for i in range(len(cluster1)):
        point1 = cluster1[i]
        for j in range(len(cluster2)):
            point2 = cluster2[j]
            value = distances[point2][point1]
            if value > max:
                max = value

    return max


def average_linkage(cluster1,cluster2,distances):
    
    #This function returns the distance between two clusters using average linkage algorithm.
    #Average linkage algorithm calculates the average of all the pair wise distances between all the points in two clusters.
    #Inputs are the two clusters and the list of pairwise distances. Returns the average linkage distance as the output.
    
    total_sum = 0.0
    total_counts = 0
    #calculates all possible pairwise distances between the 2 clusters and calculates the average
    for i in range(len(cluster1)):
        point1 = cluster1[i]
        for j in range(len(cluster2)):
            point2 = cluster2[j]
            value = distances[point2][point1]
            total_sum += float(value)
            total_counts += 1

    average = float(total_sum)/total_counts
    return average

def print_cluster(cluster):
    
    #the function prints the number of clusters and the elements in each cluster
    #It takes the cluster as input and displays it. No value is returned
    
    k = len(cluster)
    print "Number of clusters = %d\n" %(k)
    for i in range(k):
        cluster[i].sort()
        print "cluster %d =" %(i)
        print cluster[i]
        print "\n"

def linkage_clustering_c(algorithm,k,dataset):
    
    #This function takes algorithm name,k(number of clusters) and the dataset as the input and prints the clusters created.
    #At every step, the closest clusters are merged until only k clusters are left
    #calculates the distance between clusters based on the algorithm argument provided(single/average/complete linkage)
    
    n = len(dataset)
    distances = pairwise_distance(dataset,None)
    old_clusters= []
    current = [[x] for x in range(n)]
    heap = []
    for i in range(n-1):  #creating a list in which each element is of the form [distance,[tuple]]
        for j in range(i+1,n):
            heap.append([distances[i][j],[[i],[j]]])
    #creating a heap
    hp.heapify(heap)

    while len(current) > k:
        min = hp.heappop(heap)
        pair_dist = min[0]
        pair_items = min[1]
        a= 0
        for old in old_clusters:
        #checks if the tuple oppped from the heap is already processed
            if old in pair_items:
                a =1
        if a == 1:
            continue
        #maintains a list of clusters already merged with others
        for pair in pair_items:
            old_clusters.append(pair)
            current.remove(pair)
        
        new_cluster = sum(pair_items,[])
        for i in range(len(current)):
            #calculating the distance appropriately for each algorithm
            if (algorithm == "single_linkage"):
                distance = single_linkage(current[i],new_cluster,distances)
            elif (algorithm == "complete_linkage"):
                distance = complete_linkage(current[i],new_cluster,distances)
            elif (algorithm == "average_linkage"):
                distance = average_linkage(current[i],new_cluster,distances)

            heap_entry = (distance, [current[i],new_cluster])
            hp.heappush(heap,heap_entry)

        current.append(new_cluster)

    return current


def hamming_distance(target,cluster_output,length):
    
    #This function compares the target truth and the clustering algorithm we developed using hamming distance
    #It takes target truth, cluster output and length of the dataset as inputs and returns the hamming distance as output
    
    output = [0 for x in range(length)]
    for i in range(len(cluster_output)): #assigning a tag to each datapoint mentioning the cluster number it belongs to
        for j in range(len(cluster_output[i])):
            output[cluster_output[i][j]] = i
    a=0
    b=0
    for i in range(length-1):
        for j in range(i+1,length):
            if (target[i] == target[j]) and (output[i] != output[j]): #in-cluster in target,between cluster in output
                a += 1
            if (target[i] != target[j]) and (output[i] == output[j]): #in-cluster in output,between cluster in target
                b += 1

    total_pairs = float(length * (length -1))/2 #C(n 2) total number of edge pairs
    hamming_dist = float(a+b)/total_pairs

    return hamming_dist



