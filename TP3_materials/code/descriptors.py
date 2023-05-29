#
#
#      0=============================0
#      |    TP4 Point Descriptors    |
#      0=============================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Script of the practical session
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

# Import library to plot in python
from matplotlib import pyplot as plt

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time


# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def local_PCA(points):
    
    bary = points.mean(axis=0, keepdims=True)
    
    Q = points - bary
    
    cov = 1/len(points) * (Q.T @ Q)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    return eigenvalues, eigenvectors



# This function needs to compute PCA on the neighborhoods of all query_points in cloud_points
def compute_local_PCA(query_points, cloud_points, radius):

    all_eigenvalues = np.zeros((query_points.shape[0], 3))
    all_eigenvectors = np.zeros((query_points.shape[0], 3, 3))
    
    
    kdtree = KDTree(cloud_points)  
    
    neighborhood_indices = kdtree.query_radius(query_points, r=radius, count_only=False, return_distance=False)
    
    for i, neighbor_index_list in enumerate(neighborhood_indices):
        neighborhoods = cloud_points[neighbor_index_list,:]
        eigenvalues, eigenvectors = local_PCA(neighborhoods)
        all_eigenvalues[i,:] = eigenvalues
        all_eigenvectors[i,:] = eigenvectors
    
    return all_eigenvalues, all_eigenvectors



def compute_features(query_points, cloud_points, radius):

    # Compute the features for all query points in the cloud
    epsilon = 1e-16
    all_eigenvalues, all_eigenvectors = compute_local_PCA(query_points, cloud_points, radius)
    normal_vectors = all_eigenvectors[:,0]
    
    verticality = 2 * np.arcsin(np.abs(normal_vectors[:,-1]))/np.pi
    linearity = 1 - all_eigenvalues[:,1]/(all_eigenvalues[:,2]+epsilon)
    planarity = (all_eigenvalues[:,1]-all_eigenvalues[:,0])/(all_eigenvalues[:,2]+epsilon)
    sphericity = all_eigenvalues[:,0]/(all_eigenvalues[:,2]+epsilon)

    return verticality, linearity, planarity, sphericity


# ------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
# 
#   Here you can define the instructions that are called when you execute this file
#

if __name__ == '__main__':

    # PCA verification
    # ****************
    #       

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # Compute PCA on the whole cloud
        eigenvalues, eigenvectors = local_PCA(cloud)

        # Print your result
        print(eigenvalues)

        # Expected values :
        #
        #   [lambda_3; lambda_2; lambda_1] = [ 5.25050177 21.7893201  89.58924003]
        #
        #   (the convention is always lambda_1 >= lambda_2 >= lambda_3)
        #



    # Normal computation
    # ******************
    #

    if False:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # YOUR CODE
        
        # parameters choosen
        radius = 0.30
        
        # computations
        eigenvalues, eigenvectors = compute_local_PCA(cloud, cloud, radius)
        
        # ratio = eigenvalues[:,0]/eigenvalues[:,1]
        
        new_cloud = np.concatenate([cloud, eigenvectors[:,0]], axis=1)
        
        # save
        write_ply('../data/Lille_street_small_{}.ply'.format(radius), [new_cloud], ['x', 'y', 'z', 'nx', 'ny', 'nz'])


    # Features computation
    # ********************
    #

    if True:

        # Load cloud as a [N x 3] matrix
        cloud_path = '../data/Lille_street_small.ply'
        cloud_ply = read_ply(cloud_path)
        cloud = np.vstack((cloud_ply['x'], cloud_ply['y'], cloud_ply['z'])).T

        # YOUR CODE
        radius = 0.3
        
        verticality, linearity, planarity, sphericity = compute_features(cloud, cloud, radius)
        new_cloud = np.concatenate([cloud, verticality[:,None], linearity[:,None], planarity[:,None], sphericity[:,None]], axis=1)
        
        write_ply('../data/Lille_street_small_featured_{}.ply'.format(radius), [new_cloud], ['x', 'y', 'z', 'verticality', 'linearity', 'planarity', 'sphericity'])