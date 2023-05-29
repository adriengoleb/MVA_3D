#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Third script of the practical session. Neighborhoods in a point cloud
#
# ------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 13/12/2017
#


# ------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#


def brute_force_spherical(queries, supports, radius):

    neighborhoods = []
    for query in queries: 
        query = query[None, :]
        distance = cdist(query, supports,'sqeuclidean').reshape(-1) #Compute distance between each pair of our collections
        indices = np.where(distance < radius)[0]
        neighborhoods.append(supports[indices])
    return neighborhoods


def brute_force_KNN(queries, supports, k):

    neighborhoods = []
    for query in queries: 
        query = query[None, :]
        distance = cdist(query, supports,'sqeuclidean').reshape(-1)
        indices = np.argpartition(distance, k)[:k] #partition of our indices 
        neighborhoods.append(supports[indices])
    return neighborhoods


#Question 4-A
def kdtree_neighbors(queries, supports, leaf_size, radius):
    t0 = time.time()
    kdtree = KDTree(supports, leaf_size)
    t1 = time.time()
    neighborhood_indices = kdtree.query_radius(queries, radius, count_only=False, return_distance=False)
    neighborhoods = [supports[neighborhood_index] for neighborhood_index in neighborhood_indices]
    t2 = time.time()
    return neighborhoods, t1-t0, t2-t1


def hierarchical_spherical(queries, supports, radius, leaf_size=2):

    tree = KDTree(supports, leaf_size = leaf_size)
    neighborhoods = tree.query_radius(queries, radius)

    return neighborhoods


#Question 4-B - plt
def plot_exec(x, y, x_label='Radius', semilogx=False):

    plt.figure(figsize = (6, 6))
    if semilogx:
        plt.semilogx(x, y)
    else:
        plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel('Computing time')
    plt.grid()
    plt.title(f'Computing time of spherical neighborhoods using KDTree with respect to {x_label}')
    plt.show()
    plt.savefig('../time_Kdtree_vs_radius.png')



# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # Load point cloud
    # ****************
    #
    #   Load the file '../data/indoor_scan.ply'
    #   (See read_ply function)
    #

    # Path of the file
    file_path = '../data/indoor_scan.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T






    # Brute force neighborhoods
    # *************************
    #

    # If statement to skip this part if you want
    if True:

        # Define the search parameters
        neighbors_num = 100
        radius = 0.2
        num_queries = 10

        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        # Search spherical
        t0 = time.time()
        neighborhoods_spherical = brute_force_spherical(queries, points, radius)
        print(len(neighborhoods_spherical)) #search the number of neighborhoods for 10 queries
        t1 = time.time()

        # Search KNN      
        neighborhoods_knn = brute_force_KNN(queries, points, neighbors_num)
        print(len(neighborhoods_knn)) #search the number of neighborhoods for 10 queries
        t2 = time.time()

        # Print timing results
        print('{:d} spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))
        print('{:d} KNN computed in {:.3f} seconds'.format(num_queries, t2 - t1))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        total_KNN_time = points.shape[0] * (t2 - t1) / num_queries

        print('Computing spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))
        print('Computing KNN on whole cloud : {:.0f} hours'.format(total_KNN_time / 3600))

 
 
     # KDTree neighborhoods
    # ********************
    #

    # If statement to skip this part if wanted
    if True:

        # Define the general search parameters
        

        num_queries = 1000
        leaf_size = [2,4,6,10,14,18,25,30,40,50,60,80,100,120,150,200,250,1e3,1e4, 5e4, 1e5, 5e5,1e6]
        leaf_size = 1e4
        radius = np.linspace(0.1, 2.0, num=20)
        # Pick random queries
        random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
        queries = points[random_indices, :]

        
        ########################################### Question 4-A #####################################################################

        # Define the search parameters
        num_queries = 1000
        num_iterations = 5
        radius = 0.2

        
        radiuses = [0.1,0.2,0.4,0.6,0.8,1,1.2,1.4]


        # YOUR CODE
        #leaf_sizes = np.linspace(40,60,20, dtype=int)
        leaf_sizes = np.sort(np.concatenate([np.linspace(10,1000,20, dtype=int), np.linspace(40,60,20, dtype=int)]))
        times = np.zeros((len(leaf_sizes), num_iterations, 2))
        times = np.zeros((len(leaf_sizes), num_iterations, 2))
        
        
        for j in range(num_iterations):
            # Pick random queries
            random_indices = np.random.choice(points.shape[0], num_queries, replace=False)
            queries = points[random_indices, :]
            
            for i, leaf_size in enumerate(leaf_sizes):
                # launch neighborhood search
                neighborhoods, d1, d2 = kdtree_neighbors(queries, points, leaf_size, radius)
                times[i,j,0] = d1
                times[i,j,1] = d2
        
                # Print timing results
                print('{:d} kdtree of leaf size {} computed in {:.3f} seconds for iteration {}'.format(num_queries, leaf_size, d2, j))
            
        plt.plot(leaf_sizes, times[:,:,1].mean(axis=1))
        plt.xlabel('leaf size')
        plt.ylabel('time for queries')
        plt.savefig('../time_queries_vs_leaf_size.png')
        plt.show()
        
        opt_leaf_size = leaf_sizes[np.argmin(times[:,:,1].mean(axis=1))]
        print('Optimal leaf size: {}'.format(opt_leaf_size))


        ############################################################ Question 4-B ##################################################################

        #Search spherical 


        y=[]
        for r in radius:
           t0 = time.time()
           neighborhoods = hierarchical_spherical(queries, points, r, leaf_size= opt_leaf_size) #computation with the optimal leaf_size
           t1 = time.time()
           y.append(t1-t0)

        plot_exec(radius, y, x_label='radius', semilogx=False)

        t0 = time.time()
        neighborhoods = hierarchical_spherical(queries, points, 0.2, leaf_size= opt_leaf_size)
        s=sum([len(n) for n in neighborhoods])
        print(s / len(neighborhoods))
        t1 = time.time()
        # Print timing results
        print('{:d} KDTree spherical neighborhoods computed in {:.3f} seconds'.format(num_queries, t1 - t0))

        # Time to compute all neighborhoods in the cloud
        total_spherical_time = points.shape[0] * (t1 - t0) / num_queries
        print('Computing KDTree spherical neighborhoods on whole cloud : {:.0f} hours'.format(total_spherical_time / 3600))



        
        i = radiuses.index(0.2)
        d1,d2 = times.mean(axis=1)[i,0], times.mean(axis=1)[i,1]
        
        # Print timing results
        print('{:d} kd tree search computed in {:.3f} seconds'.format(num_queries, d2))

        # Time to compute all neighborhoods in the cloud
        total_tree_search_time = points.shape[0] * d2 / num_queries
        print('Computing kd tree search on whole cloud : {:.0f} hours'.format(total_tree_search_time  / 3600))