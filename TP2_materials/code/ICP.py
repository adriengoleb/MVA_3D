#
#
#      0===================================0
#      |    TP2 Iterative Closest Point    |
#      0===================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Script of the practical session
#
#------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 17/01/2018
#


#------------------------------------------------------------------------------------------
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
from visu import show_ICP

import sys


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def best_rigid_transform(data, ref):
    '''
    Computes the least-squares best-fit transform that maps corresponding points data to ref.
    Inputs :
        data = (d x N) matrix where "N" is the number of points and "d" the dimension
         ref = (d x N) matrix where "N" is the number of points and "d" the dimension
    Returns :
           R = (d x d) rotation matrix
           T = (d x 1) translation vector
           Such that R * data + T is aligned on ref
    '''

    # YOUR CODE
    bary = data.mean(axis=1, keepdims=True)
    bary_ref = ref.mean(axis=1, keepdims=True)
    
    Q = data - bary
    Q_ref = ref - bary_ref
    
    H = Q @ Q_ref.T
    
    #svd computation
    U, _, Vt = np.linalg.svd(H)
    
    R = Vt.T @ U.T

    #trick like explained in the instructions
    if np.linalg.det(R)<0:
        U[-1] *= -1 
        R = Vt.T @ U.T
    T = bary_ref - R @ bary

    return R, T





def icp_point_to_point(data, ref, max_iter, RMS_threshold):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = []
    
    # YOUR CODE
    i = 0
    old_R = np.identity(len(data))
    old_T = np.zeros((len(data), 1))
    
    rms = RMS_threshold + 1
    old_rms = None
    
    kdtree = KDTree(ref.T)
    

    #stop when reaching a certain number of iterations and when the RMS error between clouds gets below a threshold.
    while i<max_iter and rms > RMS_threshold:
        

        #called kdtree to compute the nearest neighbors and retrieve the indexes
        neighborhood_indices = kdtree.query(data_aligned.T, k=1, return_distance=False).squeeze()
        neighborhoods = ref[:,neighborhood_indices]

        R, T = best_rigid_transform(data_aligned, neighborhoods)
        
        data_aligned = R @ data_aligned + T

        #computation of rms for each iteration
        rms = np.sqrt(np.mean(np.linalg.norm(neighborhoods-data_aligned, axis=0)))
        
        #update liste
        R_list.append(old_R @ R)
        T_list.append(R @ old_T + T)
        neighbors_list.append(neighborhood_indices)
        rms_list.append(rms)
        
        if old_rms:
            if old_rms-rms<RMS_threshold:
                break
        
        old_rms = rms
        old_R = old_R @ R
        old_T = R @ old_T + T
        
        i += 1

    return data_aligned, R_list, T_list, neighbors_list, rms_list






def icp_point_to_point_stochastic(data, ref, max_iter, RMS_threshold, sampling_limit=1000, final_overlap=1):
    '''
    Iterative closest point algorithm with a point to point strategy.
    Inputs :
        data = (d x N_data) matrix where "N_data" is the number of points and "d" the dimension
        ref = (d x N_ref) matrix where "N_ref" is the number of points and "d" the dimension
        max_iter = stop condition on the number of iterations
        RMS_threshold = stop condition on the distance
    Returns :
        data_aligned = data aligned on reference cloud
        R_list = list of the (d x d) rotation matrices found at each iteration
        T_list = list of the (d x 1) translation vectors found at each iteration
        neighbors_list = At each iteration, you search the nearest neighbors of each data point in
        the ref cloud and this obtain a (1 x N_data) array of indices. This is the list of those
        arrays at each iteration
           
    '''

    # Variable for aligned data
    data_aligned = np.copy(data)

    # Initiate lists
    R_list = []
    T_list = []
    neighbors_list = []
    rms_list = []
    
    # YOUR CODE
    i = 0
    old_R = np.identity(len(data))
    old_T = np.zeros((len(data), 1))
    
    rms = RMS_threshold + 1
    kdtree = KDTree(ref.T)
    
    while i<max_iter and rms > RMS_threshold:
        
        data_aligned_sampled_indices = np.random.choice(np.arange(data_aligned.shape[1]), size=sampling_limit, replace=False)
        data_aligned_sampled = data_aligned[:, data_aligned_sampled_indices]
        
        distances, neighborhood_indices = kdtree.query(data_aligned_sampled.T, k=1, return_distance=True)
        distances = distances.squeeze()
        neighborhood_indices = neighborhood_indices.squeeze()
        neighborhoods = ref[:,neighborhood_indices]
        
        if final_overlap != 1:
            nb_indices_to_select = int(len(distances) * final_overlap)
            kept_indices = np.argpartition(distances, nb_indices_to_select)[:nb_indices_to_select]
            data_aligned_sampled2 = data_aligned_sampled[:, kept_indices]
            neighborhoods2 = neighborhoods[:, kept_indices]
        else:
            data_aligned_sampled2 = data_aligned_sampled
            neighborhoods2 = neighborhoods


        #update parameters
        
        R, T = best_rigid_transform(data_aligned_sampled2, neighborhoods2)
        
        data_aligned = R @ data_aligned + T

        rms = np.sqrt(np.mean(np.linalg.norm(neighborhoods-data_aligned_sampled, axis=0)))
        
        R_list.append(old_R @ R)
        T_list.append(R @ old_T + T)
        neighbors_list.append(neighborhood_indices)
        rms_list.append(rms)
        
        old_R = old_R @ R
        old_T = R @ old_T + T
        
        i += 1

    return data_aligned, R_list, T_list, neighbors_list, rms_list



#------------------------------------------------------------------------------------------
#
#           Main
#       \**********/
#
#
#   Here you can define the instructions that are called when you execute this file
#


if __name__ == '__main__':
   
    # Transformation estimation
    # *************************
    #

    # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_returned.ply'

		# Load clouds
        bunny_o_ply = read_ply(bunny_o_path)
        bunny_r_ply = read_ply(bunny_r_path)
        bunny_o = np.vstack((bunny_o_ply['x'], bunny_o_ply['y'], bunny_o_ply['z']))
        bunny_r = np.vstack((bunny_r_ply['x'], bunny_r_ply['y'], bunny_r_ply['z']))

        # Find the best transformation
        R, T = best_rigid_transform(bunny_r, bunny_o)

        # Apply the tranformation
        bunny_r_opt = R.dot(bunny_r) + T

        # Save cloud
        write_ply('../bunny_r_opt', [bunny_r_opt.T], ['x', 'y', 'z'])

        # Compute RMS
        distances2_before = np.sum(np.power(bunny_r - bunny_o, 2), axis=0)
        RMS_before = np.sqrt(np.mean(distances2_before))
        distances2_after = np.sum(np.power(bunny_r_opt - bunny_o, 2), axis=0)
        RMS_after = np.sqrt(np.mean(distances2_after))

        print('Average RMS between points :')
        print('Before = {:.3f}'.format(RMS_before))
        print(' After = {:.3f}'.format(RMS_after))
   

    # Test ICP and visualize
    # **********************
    #

     # If statement to skip this part if wanted
    if False:

        # Cloud paths
        ref2D_path = '../data/ref2D.ply'
        data2D_path = '../data/data2D.ply'

        # Load clouds
        ref = read_ply(ref2D_path)
        ref = np.vstack((ref['x'], ref['y']))
        
        data = read_ply(data2D_path)
        data = np.vstack((data['x'], data['y']))
        
        # parameters
        max_iter = 100
        RMS_threshold = 1e-5

        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(data, ref, max_iter, RMS_threshold)
        
        # Show ICP
        plt.plot(rms_list)
        plt.xlabel('iterations')
        plt.ylabel('rms')
        plt.savefig('../images/rms_plot_2D.png')
        show_ICP(data, ref, R_list, T_list, neighbors_list)
    
        

     # If statement to skip this part if wanted
    if False:

        # Cloud paths
        bunny_o_path = '../data/bunny_original.ply'
        bunny_r_path = '../data/bunny_perturbed.ply'

        # Load clouds
        ref = read_ply(bunny_o_path)
        ref = np.vstack((ref['x'], ref['y'], ref['z']))
        
        data = read_ply(bunny_r_path)
        data = np.vstack((data['x'], data['y'], data['z']))

        # parameters
        max_iter = 100
        RMS_threshold = 1e-5

        # Apply ICP
        data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point(data, ref, max_iter, RMS_threshold)
        
        # Show ICP
        write_ply('../data/bunny_ICP.ply', [data_aligned.T], ['x', 'y', 'z'])
        
        plt.plot(rms_list)
        plt.xlabel('iterations')
        plt.ylabel('rms')
        plt.savefig('../images/rms_plot_bunny_ICP.png')
        show_ICP(data, ref, R_list, T_list, neighbors_list)


    
    # Fast ICP
    # ********
    #

    # If statement to skip this part if wanted
    if True:

        # Cloud paths
        NDDC_1_path = '../data/Notre_Dame_Des_Champs_1.ply'
        NDDC_2_path = '../data/Notre_Dame_Des_Champs_2.ply'

        # Load clouds
        ref = read_ply(NDDC_1_path)
        ref = np.vstack((ref['x'], ref['y'], ref['z']))
        
        data = read_ply(NDDC_2_path)
        data = np.vstack((data['x'], data['y'], data['z']))
        
        # parameters
        max_iter = 100
        RMS_threshold = 1e-5
        sampling_limits = [1000, 10000]
        rms_dict = {}
        
        # Apply fast ICP for different values of the sampling_limit parameter
        for sampling_limit in sampling_limits:
            print('Sampling_limit = {}'.format(sampling_limit))
            
            data_aligned, R_list, T_list, neighbors_list, rms_list = icp_point_to_point_stochastic(data, ref, max_iter, RMS_threshold, sampling_limit)
            rms_dict[sampling_limit] = rms_list
            
            # Plot RMS
            plt.clf()
            plt.plot(rms_list)
            plt.xlabel('iterations')
            plt.ylabel('rms')
            plt.savefig('../images/rms_plot_nddc_{}.png'.format(sampling_limit))
            
            write_ply('../data/nddc_transformed_{}.ply'.format(sampling_limit), [data_aligned.T], ['x', 'y', 'z'])
            #
            # => To plot something in python use the function plt.plot() to create the figure and 
            #    then plt.show() to display it
        for k,v in rms_dict.items():
            plt.plot(v, label=k)
        plt.legend()
        plt.savefig('../images/rms_plot_nddc_sampling_limits.png')
    


