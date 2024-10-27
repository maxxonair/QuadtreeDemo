import numpy as np
import sys
from math import sqrt

from PointCloud import PointCloud

class Node():
  
  def __init__(self, P1: np.array, P2: np.array):
    """
    Initialise a node.
    
    A node is defined by two key points P1 and P2 as follows:
    
    P1-------|
    |        |
    |        |
    |        |
    -------- P2
    
    
    Note: The following needs to be true for all nodes
    P1[0] < P2[0]
    P1[1] < P2[1]
    
    Initialising a node that doesn't meet these requirements will 
    result in a runtime error!
    
    """
    # Sanity check that node corner coordinates are sensible
    if P1[0] >= P2[0]:
      raise RuntimeError('Attempt to initialise node with zero or negative size P1_x >= P2_x')
    if P1[1] >= P2[1]:
      raise RuntimeError('Attempt to initialise node with zero or negative size P1_y >= P2_y')
    
    self.P1 = P1
    self.P2 = P2
    
    # TODO future members
    self.num_valid_points = 0
    
    self.list_valid_points = []
    
    self.covariance = []
    self.eigenvalues = np.zeros(3, dtype=float)
    self.eigenvectors = []
    
    
  def _compute_pca(self, point_cloud: PointCloud) -> bool:
    """
    Compute principle compenent Analysis, including
    * covariance matrix
    * eigenvalues / eigenvectors
    
    Points for this cluster are copied from the point cloud

    Args:
        point_cloud (PointCloud): point cloud for the entire depth map
        
    Returns:
        bool : True if PCA was successful
    """
    # Get 3D points in the camera frame for all points that are within the node
    for u_index in range(self.P1[0], self.P2[0], 1):
      for v_index in range(self.P1[1], self.P2[1], 1):
        point = point_cloud.get_point_from_uv(u_index, v_index)
        
        # Only add valid points to the list (exclude zero depth)
        if np.linalg.norm(point) > sys.float_info.min:
          self.list_valid_points.append(point_cloud.get_point_from_uv(u_index, v_index))
        
    if len(self.list_valid_points) == 0:
      return False
        
    point_array = np.array(self.list_valid_points)
    self.covariance = np.cov(point_array, rowvar=False)
    
    
    self.eigenvalues, self.eigenvectors = np.linalg.eig(self.covariance)
    
    # TODO sort eigenvalue/eigenvector pairs from smallest to largest eigenvalue
    
    return True
    
  def get_center_point(self) -> np.array:
    """
    Return the coordinate of the center point for this node

    Returns:
        np.array: center point for this node
    """
    return [round((self.P2[0] - self.P1[0]) / 2), round((self.P2[1] - self.P1[1]) / 2)]
  
  def get_smallest_dimension(self) -> int:
    """
    Get the length of the smallest edge of this node
    
    
    """
    width = (self.P2[0] - self.P1[0])
    height = (self.P2[1] - self.P1[1])
    
    if width > height:
      return height
    else:
      return width
    
  def _get_smallest_eigenvalue(self) -> float:
    """
    Return the smallest eigenvector of the principle component analysis

    Returns:
        float: smallest eigenvalue
    """
    return min(abs(self.eigenvalues))
  
  def get_2sigma_theshold(self) -> float:
    """
    Return 2-sigma threshold

    Returns:
        float: _description_
    """
    return 2 * sqrt(self._get_smallest_eigenvalue())