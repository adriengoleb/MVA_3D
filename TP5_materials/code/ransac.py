#
#
#      0===========================================================0
#      |                      TP6 Modelisation                     |
#      0===========================================================0
#
#
#------------------------------------------------------------------------------------------
#
#      Plane detection with RANSAC
#
#------------------------------------------------------------------------------------------
#
#      Xavier ROYNARD - 19/02/2018
#


#------------------------------------------------------------------------------------------
#
#          Imports and global variables
#      \**********************************/
#


# Import numpy package and name it "np"
import numpy as np

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.neighbors import KDTree


#------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define usefull functions to be used in the main
#


def compute_plane(points):
    
    #point_plane = np.zeros((3,1))
    #normal_plane = np.zeros((3,1))
    point_plane = points[0][:, None]

    vec1 = points[1] - points[0]
    vec2 = points[2] - points[0]

    normal_plane = np.cross(vec1, vec2)
    normal_plane = normal_plane / (np.linalg.norm(normal_plane))
    return point_plane, normal_plane



def in_plane(points, pt_plane, normal_plane, threshold_in=0.1):
    
    #indexes = np.zeros(len(points), dtype=bool)
    
    dists = np.abs((points - pt_plane.T).dot(normal_plane))
    indexes = dists < threshold_in
    
    return np.squeeze(indexes)


def variance(points, n=50, radius=0.3):
    # compute variance as the mean number of neighbors of each point in a spherical region
    # first we sample n random points for which we will compute the number of neighbors 
    if len(points)<n:
        n=len(points) 
    sample_points = points[np.random.choice(n, 3, replace = False)]
    tree = KDTree(points, leaf_size = 10)
    neighborhoods = tree.query_radius(sample_points, radius)
    s = sum([len(n) for n in neighborhoods]) / len(neighborhoods)

    return s

def variance_(points):
    centroid = np.mean(points,axis=0)
    diff = points - centroid
    dist = np.sum(np.linalg.norm(diff, axis=-1))
    normalized_dist = dist / len(points)
    return normalized_dist

def sample(points, N, nb_draws, k = 2000):
    
    tree = KDTree(points)
    seeds = np.random.choice(N, nb_draws, replace = False)
    first_pt = points[seeds]
    neigh = tree.query(first_pt, 2*k, return_distance=False)
    neigh = neigh[:,k:] # to avoid taking points too close
    random_mask = [np.random.choice(k, 2, replace = False) for _ in range(nb_draws)]
    sec_third_points = np.array([neigh[i,random_mask[i]] for i in range(nb_draws)])
    indices = np.hstack((seeds.reshape(-1,1), sec_third_points))
    return indices
    

def RANSAC(points, nb_draws=100, threshold_in=0.1, thresh_variance=None, random_sampling=True):
    
    best_pt_plane = np.zeros((3,1))
    best_normal_plane = np.zeros((3,1))
    best_vote = 3
    N = len(points)
    if not random_sampling: #better way for sampling
        indices = sample(points, N, nb_draws, k = 2000)
        
    for i in range(nb_draws):
        if random_sampling:
            sample_points = points[np.random.choice(N, 3, replace = False)]
        else:
            sample_points = points[indices[i]]
        pt_plane, normal_plane = compute_plane(sample_points)
        indexes = in_plane(points, pt_plane, normal_plane, threshold_in)
        # computing variance of planes,
        if thresh_variance is not None:
            var = variance(points[indexes])
            ok_var = var>thresh_variance
        else: ok_var=True
        vote = indexes.sum()
        if vote > best_vote and ok_var: 
            best_vote = vote
            best_pt_plane = pt_plane
            best_normal_plane = normal_plane
                
    return best_pt_plane, best_normal_plane, best_vote


def recursive_RANSAC(points, nb_draws=100, threshold_in=0.1, nb_planes=2, thresh_variance=None, random_sampling=True):
    
    nb_points = len(points)
    plane_inds = np.arange(0,0)
    plane_labels = np.arange(0,0)
    remaining_inds = np.arange(0,nb_points).astype(int)

    for i in tqdm(range(nb_planes)):
        best_pt_plane, best_normal_plane,_ = RANSAC(points[remaining_inds], nb_draws, threshold_in, thresh_variance, random_sampling)
        in_plane_mask = in_plane(points[remaining_inds], best_pt_plane, best_normal_plane, threshold_in)
        in_plane_indices = remaining_inds[in_plane_mask]
        plane_inds = np.append(plane_inds, in_plane_indices, 0)
        plane_labels = np.append(plane_labels, i*np.ones(len(in_plane_indices)), 0)
        remaining_inds = remaining_inds[~in_plane_mask]

    return plane_inds, remaining_inds, plane_labels


#------------------------------------------------------------------------------------------
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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']
    nb_points = len(points)
    

    # Computes the plane passing through 3 randomly chosen points
    # ************************
    #
    
    print('\n--- 1) and 2) ---\n')
    
    # Define parameter
    threshold_in = 0.10

    # Take randomly three points
    pts = points[np.random.randint(0, nb_points, size=3)]
    
    # Computes the plane passing through the 3 points
    t0 = time.time()
    pt_plane, normal_plane = compute_plane(pts)
    t1 = time.time()
    print('plane computation done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    t0 = time.time()
    points_in_plane = in_plane(points, pt_plane, normal_plane, threshold_in)
    t1 = time.time()
    print('plane extraction done in {:.3f} seconds'.format(t1 - t0))
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save extracted plane and remaining points
    write_ply('../plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Computes the best plane fitting the point cloud
    # ***********************************
    #
    #
    
    print('\n--- 3) ---\n')

    # Define parameters of RANSAC
    nb_draws = 100
    threshold_in = 0.10

    # Find best plane by RANSAC
    t0 = time.time()
    best_pt_plane, best_normal_plane, best_vote = RANSAC(points, nb_draws, threshold_in)
    t1 = time.time()
    print('RANSAC done in {:.3f} seconds'.format(t1 - t0))
    
    # Find points in the plane and others
    points_in_plane = in_plane(points, best_pt_plane, best_normal_plane, threshold_in)
    plane_inds = points_in_plane.nonzero()[0]
    remaining_inds = (1-points_in_plane).nonzero()[0]
    
    # Save the best extracted plane and remaining points
    write_ply('../best_plane.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    write_ply('../remaining_points_best_plane.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    

    # Find "all planes" in the cloud
    # ***********************************
    #
    #
    
    print('\n--- 4) ---\n')
    
    # Define parameters of recursive_RANSAC
    nb_draws = 100
    threshold_in = 0.10
    nb_planes = 2
    
    # Recursively find best plane by RANSAC
    t0 = time.time()
    plane_inds, remaining_inds, plane_labels = recursive_RANSAC(points, nb_draws, threshold_in, nb_planes)
    t1 = time.time()
    print('recursive RANSAC done in {:.3f} seconds'.format(t1 - t0))
                
    # Save the best planes and remaining points
    write_ply('../best_planes.ply', [points[plane_inds], colors[plane_inds], labels[plane_inds], plane_labels.astype(np.int32)], ['x', 'y', 'z', 'red', 'green', 'blue', 'label', 'plane_label'])
    write_ply('../remaining_points_best_planes.ply', [points[remaining_inds], colors[remaining_inds], labels[remaining_inds]], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])
    
    
    
    print('Done')
    