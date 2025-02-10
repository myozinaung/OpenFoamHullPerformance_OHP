from dataclasses import dataclass
from typing import List, Optional, Tuple
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class BoundingBox:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    z: float
    
    @property
    def width(self) -> float:
        """Width in y direction"""
        return self.max_y - self.min_y
    
    @property
    def length(self) -> float:
        """Length in x direction"""
        return self.max_x - self.min_x

@dataclass
class Kink:
    z: float
    width: float
    slope_change: float

class HullAnalyzer:
    def __init__(self, stl_path: str):
        self.mesh = trimesh.load_mesh(stl_path)
        self.vertices = self.mesh.vertices
        
    def get_z_range(self) -> Tuple[float, float]:
        """Get the min and max z coordinates of the mesh."""
        return np.min(self.vertices[:, 2]), np.max(self.vertices[:, 2])
    
    def get_vertical_bounding_boxes(self, num_slices: int = 2) -> List[BoundingBox]:
        """Get 2D bounding boxes at evenly spaced heights of the STL model."""
        min_z, max_z = self.get_z_range()
        z_heights = np.linspace(min_z, max_z, num_slices)
        epsilon = (max_z - min_z) / (num_slices * 2)
        
        bboxes = []
        for z in z_heights:
            points = self.vertices[np.abs(self.vertices[:, 2] - z) < epsilon]
            if bbox := self._get_2d_bbox(points, z):
                bboxes.append(bbox)
        return bboxes
    
    @staticmethod
    def _get_2d_bbox(points: np.ndarray, z: float) -> Optional[BoundingBox]:
        """Calculate 2D bounding box from a set of 3D points."""
        if len(points) == 0:
            return None
            
        return BoundingBox(
            min_x=np.min(points[:, 0]),
            max_x=np.max(points[:, 0]),
            min_y=np.min(points[:, 1]),
            max_y=np.max(points[:, 1]),
            z=z
        )
    
    @staticmethod
    def detect_kinks(
        z_heights: List[float],
        widths: List[float],
        slope_threshold: float = 1.5,
        min_slope_change: float = 10
    ) -> List[Kink]:
        """Detect kinks (sudden changes in slope) in the width profile."""
        if len(z_heights) < 3:
            return []
        
        z = np.array(z_heights)
        w = np.array(widths)
        slopes = np.diff(w) / np.diff(z)
        
        kinks = []
        for i in range(1, len(slopes)):
            if abs(slopes[i-1]) < 1e-6:
                continue
                
            slope_ratio = abs(slopes[i] / slopes[i-1])
            
            if (slope_ratio > slope_threshold or slope_ratio < 1/slope_threshold) and slope_ratio > min_slope_change:
                kinks.append(Kink(
                    z=z_heights[i],
                    width=widths[i],
                    slope_change=slope_ratio
                ))
        
        return kinks

class HullVisualizer:
    def __init__(self, bboxes: List[BoundingBox], kinks: List[Kink]):
        self.bboxes = bboxes
        self.kinks = kinks
        self.z_heights = [bbox.z for bbox in bboxes]
        self.widths = [bbox.width for bbox in bboxes]
        
    def plot_width_profile(self, output_path: str = 'hull_width_profile.png'):
        """Create and save a plot of the hull width profile with kinks."""
        plt.figure(figsize=(10, 6))
        
        # Plot width profile
        plt.plot(self.widths, self.z_heights, 'b-', label='Width profile')
        
        # Plot kinks if any
        if self.kinks:
            kink_z = [kink.z for kink in self.kinks]
            kink_widths = [kink.width for kink in self.kinks]
            plt.plot(kink_widths, kink_z, 'gx', label='Detected kinks', markersize=10)
        
        plt.grid(True)
        plt.xlabel('Width')
        plt.ylabel('Z-height')
        plt.title('Hull Width Profile')
        plt.legend()
        plt.yticks(self.z_heights)
        
        plt.savefig(output_path)
        plt.close()
    
    def print_kink_info(self):
        """Print information about detected kinks."""
        print("\nDetected kinks in width profile:")
        min_z = min(self.z_heights)
        for kink in self.kinks:
            distance_from_bottom = kink.z - min_z
            print(f"Kink at z={kink.z:.2f}, width={kink.width:.2f}, "
                  f"slope change ratio: {kink.slope_change:.2f}, "
                  f"distance from bottom: {distance_from_bottom:.2f}")

def main():
    stl_file = "geometry/Heel00/hull.stl"
    num_slices = 50
    
    # Analyze hull
    analyzer = HullAnalyzer(stl_file)
    bboxes = analyzer.get_vertical_bounding_boxes(num_slices)
    
    # Get width profile and detect kinks
    z_heights = [bbox.z for bbox in bboxes]
    widths = [bbox.width for bbox in bboxes]
    kinks = HullAnalyzer.detect_kinks(z_heights, widths, min_slope_change=100)
    
    # Visualize results
    visualizer = HullVisualizer(bboxes, kinks)
    visualizer.plot_width_profile()
    visualizer.print_kink_info()

if __name__ == '__main__':
    main()
