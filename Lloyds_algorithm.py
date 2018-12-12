#importing all the functions from linkage_algorithms.py
from linkage_algorithms import *
import numpy as np

def kmeans_centers(dataset,k):
    
    #This function randomly picks k initial centers
    #It takes dataset and k(number of clusters) as inputs and returns the initial k centers as output
    
    length = len(dataset)
    centers = list()
    for i in range(k):
        c = random.randint(0,length-1)
        centers.append(dataset[c])

    return centers

def kmeansplus_initialise_c(dataset,k):
    
    #selects n cluster centers needed for kmeans implementation by sampling with probability proportional to D(x)^2 instead of uniform random sampling. D(x) is the distance of a datapoint to it's closest center. The first center is chosen randomly
    #This function accepts dataset and k(num of clusters) as the input and returns k cluster centers selected using K-means++ initialization.
    
    length = len(dataset)
    centers = list()
    #selecting the first center randomly
    c = random.randint(0,length-1)
    centers.append(dataset[c])
    
    n_local_trials = 2 + int(np.log(k))
    clst_sq_dist = [sys.maxint for x in range(length)]
    
    for n in range(1,k):
        
        new_sq_dist = [[] for x in range(length)]
        for i in range(length):
            new_sq_dist[i] = (euclidean_distance(dataset[i],centers[n-1]) ** 2)
        #np.minimum gives the closest distance between the previous closest dist and the distance to the new center
        clst_sq_dist = np.minimum(clst_sq_dist,new_sq_dist)
        sum_sq_dist = sum(clst_sq_dist)
        
        prob = [float(x)/sum_sq_dist for x in clst_sq_dist]
        #calculating cummulative probability for sampling with probability values
        cumm_prob = np.cumsum(np.array(prob))
        n_trail_dist = []
        samples = []
        for i in range(n_local_trials):
            #sampling with probability is made few times and the best choice is picked up
            p = random.uniform(0,1)
            samples.append(np.searchsorted(cumm_prob, p)) #returns the index of the smallest value >= p
            for j in range(length):
                new_sq_dist[j] = (euclidean_distance(dataset[j],dataset[samples[i]]) ** 2)
            n_trail_dist.append(np.sum(np.minimum(clst_sq_dist, new_sq_dist)))
            
        centers.append(dataset[samples[n_trail_dist.index(min(n_trail_dist))]])
    
    return centers

def assign_to_centers(centers,dataset):
    
    #This function computes the distance of all the datapoints from the centers given and assigns them to the closest center
    #Inputs are centers and dataset and returns the cluster with all the datapoints assigned to the k closest centers
    
    cluster = [[] for x in range(len(centers))]
    distance = pairwise_distance(dataset,centers)
    #finds distance of each datapoint to the k centers and joins the nearest center
    for i in range(len(dataset)):
        ind = np.where(distance[i] == min(distance[i]))[0][0]
        cluster[ind].append(i)

    return cluster,centers

def centroid(cluster,dataset):
    
    #This function computes the centroid(vector mean) of the cluster
    #Takes the cluster and the dataset as the inputs and returns the centroid of the cluster
    
    return list(np.sum(np.array([dataset[i] for i in cluster]), axis = 0)/len(cluster))

def Lloyds_clustering(centers,dataset,k):
    
    #This function implements the actual Llyod's algorithm(k-means). New centers are repeatedly calculated and clusters are assigned untile the centers don't change anymore.
    #Inputs are initial centers, dataset and the number of clusters. Returns the final cluster as the output.

    current_cluster, current_centers = assign_to_centers(centers,dataset)
    
    new_centers = list()
    for i in range(k):
        new_center = centroid(current_cluster[i],dataset)
        new_centers.append(new_center)

    while (new_centers != current_centers):
    
        current_cluster, current_centers = assign_to_centers(new_centers,dataset)
        new_centers = list()
        for i in range(k):
            new_center = centroid(current_cluster[i],dataset)
            new_centers.append(new_center)

    return current_cluster

def main():

    #Based on the argument provided,the centers are initialised by random initialization or kmeans++
    #performs the Lloyds clustering and calculates the hamming distance for the output cluster
    
    dataset,k,target_truth = load_file()
    length = len(dataset)
    algorithm = sys.argv[2]
    start_time = time.time()
    if (algorithm == "Lloyds"):
        centers = kmeans_centers(dataset,k)
    elif (algorithm == "kmeansplus"):
        centers = kmeansplus_initialise_c(dataset,k)
    
    final_cluster = Lloyds_clustering(centers,dataset,k)

    print "K means implemented using %s:\n" %(algorithm)

    hamming_dist = hamming_distance(target_truth,final_cluster,length)
    print "Hamming distance for Lloyd's Algorithm = %f\n" %(hamming_dist)
    print("Time elapsed = %s seconds" % (time.time() - start_time))
    print "\n"


main()

