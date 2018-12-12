#importing all the functions from linkage_algorithms.py
from linkage_algorithms import *
import time

def main():
    
    #this function executes the three types of linkage algorithms and prints the hamming distance and time taken for each of them. It uses the functions imported from linkage_algorithms.py

    dataset,k,target_truth = load_file()
    st = time.time()
    length = len(dataset)
    print "Number of data points = %d\n" %(length)
    dimensions = len(dataset[0])
    print "Number of dimensions = %d\n" %(dimensions)
    print "Number of clusters = %d\n" %(k)
    
    print "*************SINGLE LINKAGE***************\n"
    #records the start time of the algorithm
    st = time.time()
    cluster = linkage_clustering_c("single_linkage",k,dataset)
    hamming_dist = hamming_distance(target_truth,cluster,length)
    print "Hamming distance for %s = %f" %("single_linkage",hamming_dist)
    #prints the total time elapsed
    print("Time elapsed = %s seconds" % (time.time() - st))

    print "\n*************COMPLETE LINKAGE***************\n"
    st = time.time()
    cluster = linkage_clustering_c("complete_linkage",k,dataset)
    hamming_dist = hamming_distance(target_truth,cluster,length)
    print "Hamming distance for %s = %f" %("complete_linkage",hamming_dist)
    print("Time elapsed = %s seconds" % (time.time() - st))

    print "\n*************AVERAGE LINKAGE***************\n"
    st = time.time()
    cluster = linkage_clustering_c("average_linkage",k,dataset)
    hamming_dist = hamming_distance(target_truth,cluster,length)
    print "Hamming distance for %s = %f" %("average_linkage",hamming_dist)
    print("Time elapsed = %s seconds" % (time.time() - st))
    print "\n"

main()
