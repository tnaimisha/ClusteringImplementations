This Assignment implements the below algorithms-
Single linkage Clustering
Average linkage Clustering
Complete linkage Clustering
Lloyds Algorithm
Lloyds Algorithm using k-means++ initialization.

The linkage algorithms were first implemented without using any data structures and numpy arrays. While it was working fine for smaller datasets, it wasn't able to converge for huge datasets. Implementation using heaps improved the performance over the initial approach and implementation using numpy elements provided the best timing complexity.

I am attaching the heap implementation as well as numpy implementation. 
linkage_algorithms_heap --> heap implementation
linkage_algorithms --> implementation using numpy

Lloyds Algorithm has two functions for initializing centers.
kmeans_centers() --> for random initialization
kmeansplus_initialise --> for kmeans++ initialization

pca_plot.py is a script using principle component analysis technique that projects all the dimensions into 2 or 3 dimensional space as required. This is to get an idea about how the cluster looks like.

Testing and other calculations are done using numpy implementation which are provided in the PPT

