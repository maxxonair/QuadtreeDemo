
import numpy as np
from dataclasses import dataclass

# Conversion factor from millimeters to meters
FACTOR_MM_TO_M = 1/1000

@dataclass(frozen=True)
class CameraParameters():
  """
  Simple data class the hold the most basic, intrincis camera parameters
  """
  cx_px: int
  cy_px: int
  inv_fx_px: float
  inv_fy_px: float
  stereo_baseline_m: float = None

class PointCloud():
  """
  Class to generate a point cloud from a depth image 
  
  Return a point for a given u/v coordinate in the depth image 
  """
  
  def __init__(self, image: np.array,
               camera_params: CameraParameters):

    # -> compute point cloud
    self._compute_point_cloud(image, camera_params)
    
  def _compute_point_cloud(self, image: np.array,
                           camera_params: CameraParameters):
    """
    Compute the point cloud for the entire depth map

    Args:
        image (np.array): depth map
        camera_params (CameraParameters): intrinsic camera parameters
    """
    img = np.array(image, dtype=float)
    (h, w) = img.shape
    # Initialise array 
    self.point_cloud_array = np.zeros((w, h, 3), dtype=float)
    # Loop over pixel 
    for u_index in range(w):
      for v_index in range(h):
        # Get depth from depth image
        depth_m = float(img[v_index, u_index]) * FACTOR_MM_TO_M
        
        # Initialise normalised vector for this pixel
        x_norm = (u_index - camera_params.cx_px) * camera_params.inv_fx_px
        y_norm = (v_index - camera_params.cy_px) * camera_params.inv_fy_px
        z_norm = 1
        
        # Scale normalised vector with depth 
        self.point_cloud_array[u_index, v_index, :] = np.array([x_norm, y_norm, z_norm], dtype=float) * depth_m
        
  def get_point_from_uv(self, u_index: int, v_index: int):
    """
    Get a point from the point cloud for a given u/v coordinate in the depth map

    Args:
        u_index (int): u index
        v_index (int): v index

    Returns:
        np.array: 3D point
    """
    return self.point_cloud_array[u_index, v_index, :]