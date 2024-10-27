
import numpy as np
import sys 
import logging
from logging import info, debug, warning
import time

from Node import Node
from PointCloud import PointCloud, CameraParameters

# Default value for smallest allowed node size (size of the smallest 
# node edge length) Nodes smaller than this threshold won't be split 
# further
DEFAULT_MIN_NODE_SIZE_PX = 1

class Quadtree():
  """
  Generic quadtree search to segment an image array 
  
  """
  
  # Maximum number of tree search iterations. The search will be cut 
  # short if this threshold is reached.
  MAX_ITER = 1000
  
  def __init__(self):
    """
    _summary_
    """
    # Smallest allowed node size (size of the smallest node edge length)
    # Nodes smaller than this threshold won't be split further
    self.min_node_size_px = DEFAULT_MIN_NODE_SIZE_PX
    
    
  def _split_node(self, node: Node) -> list:
    """
    Split node and return list of four split nodes
    
    P1     x1        
    |-------|-------|
    |  1    |   2   |
    |       |       |
    |-------|-------| y1
    |  4    |   3   |
    |       |       |
    |-------|-------|
                     P2
    Args:
        node (Node): Input node

    Returns:
        list: List of split nodes
    """
    # Pre-construct helping points: width height ad the plitting point
    x1 = round((node.P2[0] - node.P1[0]) / 2) + node.P1[0]
    y1 = round((node.P2[1] - node.P1[1]) / 2) + node.P1[1]
  
    node_list = []
    # -- Node 1 --
    node_list.append(Node(P1=[node.P1[0],node.P1[1]], 
                          P2=[x1,y1]))
    # -- Node 2 --
    node_list.append(Node(P1=[x1 ,node.P1[1]], 
                          P2=[node.P2[0], y1]))
    # -- Node 3 --
    node_list.append(Node(P1=[x1, y1], 
                          P2=[node.P2[0], node.P2[1]]))
    # -- Node 4 --
    node_list.append(Node(P1=[node.P1[0], y1], 
                          P2=[x1, node.P2[1]]))
    
    return node_list
    
  def _is_to_split_node(self, node: Node):
    """
    Function to decide if a node needs to be split or not

    Args:
        node (Node): Input Node

    Returns:
        bool: True if node needs to be split up
    """
    # Initialise flags
    is_to_split = False
    is_leaf_node = False
    
    # Compute principle point analysis for this node 
    is_suc = node._compute_pca(self.point_cloud)
    # Check if cluster is considered planar
    is_planar = (node.get_2sigma_theshold() < self.planarity_thr_m)
    
    if not is_suc:
      # PCA failed -> do NOT split further, this node is invalid
      is_to_split = False
      is_leaf_node = False
    if node.get_smallest_dimension() < self.min_node_size_px and not is_planar:
      # End is reached -> do NOT split further
      is_to_split = False
      is_leaf_node = False
    elif is_planar:
      # End is reached -> do NOT split further
      is_to_split = False
      
      # TODO currently all end nodes are markes leaf nodes. Need
      # to add check here if this is a valid leaf node.
      is_leaf_node = True
    else:
      # End condition not reached -> split node
      is_to_split = True
      
    return is_to_split, is_leaf_node
    
    
  def do(self, image: np.array, camera_params: CameraParameters, 
         min_node_size_px: int = DEFAULT_MIN_NODE_SIZE_PX,
         planarity_thr_m: float = 0.03) -> list:
    """_summary_

    Args:
        image (np.array): Input image as numpy array
        
    Returns:
        list: List of leaf nodes 
    """
    # Set minimum node size threshold setting
    self.min_node_size_px = min_node_size_px
    self.planarity_thr_m = planarity_thr_m
    
    info('[x] Create point cloud from depth map')
    start_time = time.time()
    self.point_cloud = PointCloud(image, camera_params)
    
    # Log the time to perform the tree search 
    time_pc_s = (time.time() - start_time)
    info(f'Time to generate point cloud [s] : {time_pc_s:.2f}')
    
    # Make sure image is a numpy array 
    img = np.array(image, dtype=float)
    (h, w) = img.shape
    
    # Create list for processed and unprocessed nodes in the tree
    unprocessed_node_list = []
    leaf_node_list = []
    # List of nodes that are not valid, but cannot be split further
    invalid_node_list = []
    
    info('-- Initialise Search')
    info(f' - Root node dimension x/y [px] : {w} x {h}')
    info(f' - Number of points             : {w*h}')
    # Add the root node containing the entire image array
    unprocessed_node_list.append(Node(P1=[0,0], P2=[w, h]))
    
    # Initialise counters for number of splits and total number of tree
    # search iterations
    num_splits = 0 
    num_iter = 0 
    
    start_time = time.time()
    # Keep looping until no unprocessed nodes are left or the maximum 
    # iteration threshold is reached
    while (len(unprocessed_node_list) > 0 and num_iter < self.MAX_ITER):
      
      # -- DEBUG PRINT OUT unprocessed nodes for each search iteration
      debug('')
      debug('')
      for node in unprocessed_node_list:
        debug(f' {node.P1} {node.P2} --> {node.get_smallest_dimension()}')
      debug('---------------------')
      debug(f'Number of nodes to process: {len(unprocessed_node_list)}')
        
      # Take first node from the list of unprocessed nodes
      node = unprocessed_node_list[0]
      # Remove this node from the list of unprocessed nodes
      unprocessed_node_list.pop(0)
      
      # Check if node needs to be split
      is_split, is_leaf_node = self._is_to_split_node(node)
      if is_split:
        unprocessed_node_list.extend(self._split_node(node))
        num_splits += 1
      elif is_leaf_node:
        leaf_node_list.append(node)
      else:
        warning('[+] invalid node added')
        invalid_node_list.append(node)
        pass
      
      num_iter += 1
    
    # Log the time to perform the tree search 
    time_search_s = (time.time() - start_time)
   
    # -- DEBUG PRINT OUT search summary statistics
    info('')
    info('- - - Search finished - - - ')
    info(f' - Search time [s]          {time_search_s:.2f} << -')
    info(f' - Number of splits         {num_splits}')
    info(f' - Number of iterations     {num_iter}')
    info(f' - Number of leaf nodes     {len(leaf_node_list)}')
    info(f' - Number of invalid nodes  {len(invalid_node_list)}')
    
    debug('')
    debug('[x] Leaf nodes:')
    for node in leaf_node_list:
      debug(f' {node.P1} {node.P2} --> {node.get_smallest_dimension()}')
    
    return leaf_node_list, invalid_node_list