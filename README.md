# ClusteringImplementations

This Assignment implements the below algorithms-

1. Single linkage Clustering

2. Average linkage Clustering

3. Complete linkage Clustering

4. Lloyds Algorithm

5. Lloyds Algorithm using k-means++ initialization.

The linkage algorithms were first implemented without using any data structures and numpy arrays. While it was working fine for smaller datasets, it wasn't able to converge for huge datasets. Implementation using heaps improved the performance over the initial approach and implementation using numpy elements provided the best timing complexity.

I am attaching the heap implementation as well as numpy implementation. linkage_algorithms_heap.py --> heap implementation
linkage_algorithms.py --> implementation using numpy

Lloyds Algorithm has two functions for initializing centers.  kmeans_centers() function --> for random initialization
kmeansplus_initialise() function --> for kmeans++ initialization

pca_plot.py is a script using principle component analysis technique that projects all the dimensions into 2 or 3 dimensional space as required. This is to get an idea about how the cluster looks like.


