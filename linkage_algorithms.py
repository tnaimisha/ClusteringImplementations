# This file contains the functions developed for single linkage clustering algorithm
import sys
import math
import random
import time
import numpy as np

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
    
    #This function calculates the pair wise distance between all the point in the dataset. It uses numpy arrays and other matrix operations for calculating the distance
    # It returns the list of distances and the corresponding pair of datapoints
    
    #if Y is none, then the functions calculates pairwise_distance(X,X)
    if Y is None:
        n = len(X)
        index = np.array([np.array(range(n)),]*n)
        data = np.array(X)
        data_norm = np.array([np.sum(data*data, axis=1),]*n)
        data_data = np.dot(data, np.transpose(data))

        distances = data_norm.transpose()+data_norm-2*data_data
    
    else:   #if 2 datasets are mentioned, it calculates pair wise distances between the points across 2 arrays
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

def linkage_clustering_c(algorithm,k,dataset):
    
    #This function takes algorithm name,k(number of clusters) and the dataset as the input and prints the clusters created.
    #At every step, the closest clusters are merged until only k clusters are left
    #calculates the distance between clusters based on the algorithm argument provided(single/average/complete linkage)
    
    n = len(dataset)
    distances = pairwise_distance(dataset,None)
    np.fill_diagonal(distances, sys.maxint)     #fills the diagonals with max values
    current = [[x] for x in range(n)]
    num_clusters = len(current)
    
    while len(current) > k:

        #Finding minimum from the pairwise_distances array
        (minx, miny) = np.unravel_index(np.argmin(distances), distances.shape)
        
        # Sorting the tuple (minx, miny)
        (minx, miny) = (minx, miny) if (minx < miny) else (miny, minx)

        # Getting the rows that need to be merged. X and Y are the 2 rows in the distance matrix that should be merged.
        X = distances[minx,:]
        Y = distances[miny,:]
        
        # Merging the two rows into one based on the algorithm
        if algorithm == "single_linkage":
            XY = np.minimum(X,Y)
        elif algorithm == "complete_linkage":
            XY = np.maximum(X,Y)
        elif algorithm == "average_linkage":
            XY = np.zeros(len(current))
            for i in range(len(current)):
                XY[i] = (len(current[i])*len(current[minx])*X[i] + len(current[i])*len(current[miny])*Y[i])/(len(current[i])*(len(current[minx])+len(current[miny])))

        # Removing the old rows from the distance matrix
        distances = np.delete(np.delete(distances, (minx, miny), axis = 0), (minx, miny), axis = 1)

        # Creating the row that needs to be inserted
        xy = np.delete(XY, (minx,miny))
        distances = np.insert(distances, minx, xy, axis = 0) # Inserting as a row
        # Creating the column that needs to be inserted
        xy = np.insert(xy, minx, sys.maxint, axis = 0)
        distances = np.insert(distances, minx, xy, axis = 1) #Inserting as a column

        # Updating the clusters
        current[minx] = sum([current[minx], current[miny]], [])
        del (current[miny])
        
    return current


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

def hamming_distance(target,cluster_output,length):

    #This function compares the target truth and the clustering algorithm we developed using clustering algorithm
    #It takes target truth, cluster output and length of the dataset as inputs and returns the hamming distance as output
    
    output = [0 for x in range(length)]
    for i in range(len(cluster_output)):  #assigning a tag to each datapoint mentioning the cluster number it belongs to
        for j in range(len(cluster_output[i])):
            output[cluster_output[i][j]] = i
    a=0
    b=0
    for i in range(length-1): #finding the number of wrong edges
        for j in range(i+1,length):
            if (target[i] == target[j]) and (output[i] != output[j]): #in-cluster in target,between cluster in output
                a += 1
            if (target[i] != target[j]) and (output[i] == output[j]): #in-cluster in output,between cluster in target
                b += 1

    total_pairs = float(length * (length -1))/2 #C(n 2) total number of edge pairs
    hamming_dist = float(a+b)/total_pairs

    return hamming_dist



