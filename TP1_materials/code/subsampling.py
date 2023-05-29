#
#
#      0===========================================================0
#      |    TP1 Basic structures and operations on point clouds    |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Second script of the practical session. Subsampling of a point cloud
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

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time
from tqdm import tqdm

# ------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#
#
#   Here you can define useful functions to be used in the main
#

def decimate(x, factor):
    return np.array([ x[i] for i in range(0,len(x), factor) ])

def cloud_decimation(points, colors, labels, factor):

    decimated_points = decimate(points, factor)
    decimated_colors = decimate(colors, factor)
    decimated_labels = decimate(labels, factor)

    return decimated_points, decimated_colors, decimated_labels

def grid_subsampling(points, colors, voxel_size):
    point_color = np.hstack((points, colors))
    point_0 = points.min(axis=0) #we take the minimum

    dic = {}
    print('Grouping points')
    for pc in tqdm(point_color):
        index_voxel = tuple(np.floor((pc[:3] - point_0)//voxel_size))
        if index_voxel in dic.keys():
            dic[index_voxel].append(pc)
        else:
            dic[index_voxel] = [pc]

    subsampled_points = []
    subsampled_colors = []
    print('avereging')
    for key in tqdm(dic):
        pc = np.array(dic[key]).mean(axis=0)
        subsampled_points.append(pc[:3])
        subsampled_colors.append(pc[3:])
    return np.array(subsampled_points), np.array(subsampled_colors, dtype=np.uint8)



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
    colors = np.vstack((data['red'], data['green'], data['blue'])).T
    labels = data['label']    

    # Decimate the point cloud
    # ************************
    #

    # Define the decimation factor
    factor = 300

    # Decimate
    t0 = time.time()
    decimated_points, decimated_colors, decimated_labels = cloud_decimation(points, colors, labels, factor)
    print(points.shape, decimated_points.shape)
    t1 = time.time()
    print('decimation done in {:.3f} seconds'.format(t1 - t0))

    # Save decimated version
    write_ply('../decimated.ply', [decimated_points, decimated_colors, decimated_labels], ['x', 'y', 'z', 'red', 'green', 'blue', 'label'])



    ##################################### BONUS - Ggrid subsampling part ###################################################################
    
    if True:
        #define the size of the grid
        voxel_size = 0.25
        t0 = time.time()
        subsampled_points, subsampled_colors = grid_subsampling(points, colors, voxel_size) # Subsample the point cloud on a grid with a fixed size
        print(points.shape, subsampled_points.shape)
        t1 = time.time()
        print('Grid subsampling done in {:.3f} seconds'.format(t1 - t0))

        # Save
        write_ply('../subsampled.ply', [subsampled_points, subsampled_colors], ['x', 'y', 'z', 'red', 'green', 'blue'])

        
    print('Done')
