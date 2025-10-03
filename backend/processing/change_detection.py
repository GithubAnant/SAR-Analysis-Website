import numpy as np
import cv2
from scipy import ndimage
from utils.config import config

class ChangeDetector:
    @staticmethod
    def detect_changes(image_before, image_after, threshold=None):
        """
        Detect changes between two images
        
        Args:
            image_before: numpy array (H, W, 3) - before image
            image_after: numpy array (H, W, 3) - after image
            threshold: change threshold (default from config)
        
        Returns:
            dict with change mask, statistics, and bounding boxes
        """
        if image_before is None or image_after is None:
            return None
        
        if threshold is None:
            threshold = config.CHANGE_THRESHOLD
        
        # Convert to grayscale
        gray_before = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(image_after, cv2.COLOR_RGB2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(gray_before, gray_after)
        
        # Apply Gaussian blur to reduce noise
        diff_blur = cv2.GaussianBlur(diff, (5, 5), 0)
        
        # Threshold to get binary mask
        _, change_mask = cv2.threshold(diff_blur, threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)
        change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            change_mask, connectivity=8
        )
        
        # Filter small regions
        min_area = config.MIN_CHANGE_AREA
        bounding_boxes = []
        
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                bounding_boxes.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(area),
                    'centroid': [float(centroids[i][0]), float(centroids[i][1])]
                })
        
        # Calculate statistics
        total_pixels = image_before.shape[0] * image_before.shape[1]
        changed_pixels = np.sum(change_mask > 0)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return {
            'change_mask': change_mask,
            'difference_image': diff,
            'bounding_boxes': bounding_boxes,
            'statistics': {
                'total_pixels': int(total_pixels),
                'changed_pixels': int(changed_pixels),
                'change_percentage': float(change_percentage),
                'num_regions': len(bounding_boxes)
            }
        }
    
    @staticmethod
    def visualize_changes(image_before, image_after, change_result):
        """
        Create visualization with bounding boxes
        
        Args:
            image_before: numpy array (H, W, 3)
            image_after: numpy array (H, W, 3)
            change_result: result from detect_changes()
        
        Returns:
            dict with visualized images
        """
        if not change_result:
            return None
        
        # Create copies for drawing
        vis_before = image_before.copy()
        vis_after = image_after.copy()
        
        # Draw bounding boxes
        for bbox in change_result['bounding_boxes']:
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            
            # Draw on both images
            cv2.rectangle(vis_before, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(vis_after, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Add area label
            label = f"{bbox['area']}px"
            cv2.putText(vis_after, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Create change overlay (red for changes)
        change_overlay = np.zeros_like(image_after)
        change_mask = change_result['change_mask']
        change_overlay[:, :, 0] = change_mask  # Red channel
        
        # Blend with after image
        vis_overlay = cv2.addWeighted(image_after, 0.7, change_overlay, 0.3, 0)
        
        # Create difference visualization
        diff_img = change_result['difference_image']
        diff_colored = cv2.applyColorMap(diff_img, cv2.COLORMAP_JET)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
        
        return {
            'before_annotated': vis_before,
            'after_annotated': vis_after,
            'change_overlay': vis_overlay,
            'difference_map': diff_colored
        }
    
    @staticmethod
    def calculate_area_km2(num_pixels, image_width, image_height, bbox_size_degrees):
        """
        Calculate approximate area in km²
        
        Args:
            num_pixels: number of pixels
            image_width: image width in pixels
            image_height: image height in pixels
            bbox_size_degrees: size of bounding box in degrees
        
        Returns:
            area in km²
        """
        # Approximate: 1 degree ~ 111 km at equator
        km_per_degree = 111
        
        bbox_width_km = bbox_size_degrees * km_per_degree
        bbox_height_km = bbox_size_degrees * km_per_degree
        
        total_area_km2 = bbox_width_km * bbox_height_km
        
        total_pixels = image_width * image_height
        area_per_pixel = total_area_km2 / total_pixels
        
        return num_pixels * area_per_pixel

change_detector = ChangeDetector()