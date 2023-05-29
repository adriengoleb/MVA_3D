#
#
#      0===========================================================0
#      |              TP4 Surface Reconstruction                   |
#      0===========================================================0
#
#
# ------------------------------------------------------------------------------------------
#
#      Jean-Emmanuel DESCHAUD - 02/02/2018
#


# Import numpy package and name it "np"
import numpy as np

# Import functions from scikit-learn
from sklearn.neighbors import KDTree

# Import functions to read and write ply files
from ply import write_ply, read_ply

# Import time package
import time

from skimage import measure

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import trimesh


# Hoppe surface reconstruction
def compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel):
    x = min_grid[0] + size_voxel[0] * np.arange(grid_resolution+1)
    y = min_grid[1] + size_voxel[1] * np.arange(grid_resolution+1)
    z = min_grid[2] + size_voxel[2] * np.arange(grid_resolution+1)
    vx, vy, vz = np.meshgrid(x, y, z)
    grid_points = np.stack((vx, vy, vz), 3).reshape(-1,3)

    tree = KDTree(points)
    neigh = tree.query(grid_points, 1, return_distance=False).squeeze()
    hoppe = np.sum(normals[neigh] * (grid_points-points[neigh]), axis=1)
    scalar_field[:,:] = hoppe.reshape(grid_resolution+1, grid_resolution+1, grid_resolution+1)    
    return scalar_field
    

# IMLS surface reconstruction
def compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn):
    # YOUR CODE
    x = min_grid[0] + size_voxel[0] * np.arange(grid_resolution+1)
    y = min_grid[1] + size_voxel[1] * np.arange(grid_resolution+1)
    z = min_grid[2] + size_voxel[2] * np.arange(grid_resolution+1)
    vx, vy, vz = np.meshgrid(x, y, z)
    grid_points = np.stack((vx, vy, vz), 3).reshape(-1,3)

    k = 10
    
    tree = KDTree(points)
    neigh = tree.query(grid_points, knn, return_distance=False).squeeze()
    
    xpi = grid_points[:,np.newaxis,:] - points[neigh]
    xpi_norm = np.linalg.norm(xpi, axis=2)
    #h=0.001
    h = np.clip(xpi_norm, 0.001, None)

    theta = np.exp(-(xpi_norm**2)/h**2)
    
    nxpi = np.sum(normals[neigh] * xpi, axis=2)
    imls = np.sum(nxpi * theta, axis=1) / np.sum(theta, axis = 1)

    scalar_field[:,:] = imls.reshape(grid_resolution+1, grid_resolution+1, grid_resolution+1)    

    return scalar_field



if __name__ == '__main__':

    t0 = time.time()
    
    # Path of the file
    file_path = '../data/bunny_normals.ply'

    # Load point cloud
    data = read_ply(file_path)

    # Concatenate data
    points = np.vstack((data['x'], data['y'], data['z'])).T
    normals = np.vstack((data['nx'], data['ny'], data['nz'])).T

	# Compute the min and max of the data points
    min_grid = np.amin(points, axis=0)
    max_grid = np.amax(points, axis=0)
				
	# Increase the bounding box of data points by decreasing min_grid and inscreasing max_grid
    min_grid = min_grid - 0.10*(max_grid-min_grid)
    max_grid = max_grid + 0.10*(max_grid-min_grid)

	# grid_resolution is the number of voxels in the grid in x, y, z axis
    grid_resolution = 30 #100
    knn = 30
    size_voxel = np.array([(max_grid[0]-min_grid[0])/grid_resolution,(max_grid[1]-min_grid[1])/grid_resolution,(max_grid[2]-min_grid[2])/grid_resolution])
	
	# Create a volume grid to compute the scalar field for surface reconstruction
    scalar_field = np.zeros((grid_resolution+1,grid_resolution+1,grid_resolution+1),dtype = np.float32)

	# Compute the scalar field in the grid
    #scalar_field = compute_hoppe(points,normals,scalar_field,grid_resolution,min_grid,size_voxel)
    scalar_field = compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,knn)


    #compute_imls(points,normals,scalar_field,grid_resolution,min_grid,size_voxel,30)

	# Compute the mesh from the scalar field based on marching cubes algorithm
    verts, faces, normals_tri, values_tri = measure.marching_cubes(scalar_field, level=0.0, spacing=(size_voxel[0],size_voxel[1],size_voxel[2]))
	
    # Export the mesh in ply using trimesh lib
    mesh = trimesh.Trimesh(vertices = verts, faces = faces)
    mesh.export(file_obj='../bunny_mesh_imls_30.ply', file_type='ply')
	
    print("Total time for surface reconstruction : ", time.time()-t0)
	

