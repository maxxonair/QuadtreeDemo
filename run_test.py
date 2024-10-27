
"""
This script is to run a Quadtree search test with the provided depth map

"""
from PIL import Image
import numpy as np
import logging
from logging import info
import plotly.graph_objects as go

from Quadtree import Quadtree
from PointCloud import CameraParameters

## SETTING: Enable plotting the Quadtree search results
IS_PLOT_RESULTS = True

def plot_nodes(node_list: list, invalid_node_list: list, image: np.array):
  """
  Simple plot function to draw the leaf nodes and invalid nodes on top of the 
  depth map

  Args:
      node_list (list): valid leaf nodes
      invalid_node_list (list): invalid leaf nodes
      image (np.array): depth map
  """
  (w,h) = np.array(image).shape
  
  fig = go.Figure()
  
  fig.add_trace(go.Heatmap(z=image, colorscale="Greys"))
  
  for node in node_list:
    fig.add_shape(type="rect",
                  x0=node.P1[0], 
                  y0=node.P1[1], 
                  x1=node.P2[0], 
                  y1=node.P2[1],
                  line=dict(color="RoyalBlue"),
                  fillcolor="LightSkyBlue",
                  opacity=0.6,
    )
  
  for node in invalid_node_list:
    fig.add_shape(type="rect",
                  x0=node.P1[0], 
                  y0=node.P1[1], 
                  x1=node.P2[0], 
                  y1=node.P2[1],
                  line=dict(color="RoyalBlue"),
                  fillcolor="OrangeRed",
                  opacity=0.6,
    )
  fig.update_xaxes(range=[0, w], showgrid=False)
  fig.update_yaxes(range=[0, h], autorange="reversed")
  fig.update_yaxes(
    scaleanchor="x",
    scaleratio=1,
  )
  fig.update_layout(template='plotly_dark')
  fig.show()

def main():
  """
  Test function to dummy test the Quadtree search
  """
  # Load a test depth map
  test_img = Image.open(r"cube.png")
  
  # Create hard coded camera parameters
  # TODO read camera parameters from file
  params = CameraParameters(inv_fx_px=(1 / 526.37),
                            inv_fy_px=(1 / 526.37),
                            cx_px=313.68,
                            cy_px=259.02)
  
  # Initialise Quadtree
  quadTree = Quadtree()
  
  # Run the search
  info('')
  info(' - - - > Do Quadtree Search - - - ')
  list_leaf_nodes, invalid_node_list = quadTree.do(test_img,
                                                   params,
                                                   min_node_size_px=10,
                                                   planarity_thr_m=0.01)
  
  if IS_PLOT_RESULTS:
    info('Plot Quadtree Results:')
    plot_nodes(list_leaf_nodes, invalid_node_list, test_img)
    
    
if __name__=="__main__":
  logging.basicConfig(level=logging.INFO,
                      format="%(levelname)s: %(message)s")
  
  main()
  
  