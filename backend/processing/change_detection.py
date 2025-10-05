import numpy as np
import cv2
from scipy import ndimage
from utils.config import config

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

class ChangeDetector:
    @staticmethod
    def detect_changes(image_before, image_after, threshold=None):
        """
        Detect meaningful changes between two images using advanced techniques
        
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
        
        # Resize images to same size if different
        if image_before.shape != image_after.shape:
            h, w = min(image_before.shape[0], image_after.shape[0]), min(image_before.shape[1], image_after.shape[1])
            image_before = cv2.resize(image_before, (w, h))
            image_after = cv2.resize(image_after, (w, h))
        
        # NEW: Primary method - Perceptual color change detection
        change_mask_color = ChangeDetector._perceptual_color_change_detection(
            image_before, image_after, threshold
        )
        
        # Convert to grayscale for other methods
        gray_before = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(image_after, cv2.COLOR_RGB2GRAY)
        
        # Method 2: Structural Similarity (SSIM) based change detection
        change_mask_ssim = ChangeDetector._ssim_change_detection(gray_before, gray_after, threshold)
        
        # Method 3: Statistical change detection with adaptive thresholding
        change_mask_adaptive = ChangeDetector._adaptive_change_detection(gray_before, gray_after)
        
        # Combine methods with weighted consensus
        # Give more weight to color-based detection as it's more perceptually accurate
        consensus = (change_mask_color.astype(float) * 1.5 + 
                    change_mask_ssim.astype(float) * 1.0 + 
                    change_mask_adaptive.astype(float) * 1.0)
        
        # A pixel is considered changed if the weighted consensus is above threshold
        change_mask = (consensus >= 1.2).astype(np.uint8) * 255
        
        # Post-processing: remove noise and small artifacts
        change_mask = ChangeDetector._post_process_mask(change_mask)
        
        # Calculate difference image for visualization
        diff = cv2.absdiff(gray_before, gray_after)
        
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
    def _perceptual_color_change_detection(img_before, img_after, threshold):
        """
        Detect significant color changes using perceptual color difference
        Only detects VAST color changes, ignoring minor variations
        
        Args:
            img_before: RGB image before (H, W, 3)
            img_after: RGB image after (H, W, 3)
            threshold: sensitivity threshold
        
        Returns:
            Binary mask where significant color changes occurred
        """
        # Convert to LAB color space for perceptual color difference
        lab_before = cv2.cvtColor(img_before, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_after = cv2.cvtColor(img_after, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Calculate Delta E (perceptual color difference) for each pixel
        # Delta E > 50 is considered a very noticeable difference
        # Delta E > 100 is considered a completely different color
        
        delta_L = lab_after[:, :, 0] - lab_before[:, :, 0]  # Lightness
        delta_a = lab_after[:, :, 1] - lab_before[:, :, 1]  # Green-Red
        delta_b = lab_after[:, :, 2] - lab_before[:, :, 2]  # Blue-Yellow
        
        # Calculate Delta E (CIE76 formula - simpler but effective)
        delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
        
        # Adaptive threshold based on image statistics
        median_delta = np.median(delta_e)
        std_delta = np.std(delta_e)
        
        # Only consider changes that are significantly above the noise level
        # This prevents minor compression artifacts and noise from being detected
        adaptive_threshold = max(
            50.0,  # Minimum perceptual threshold
            median_delta + 2.5 * std_delta,  # Statistical outlier threshold
            threshold * 10  # User-defined threshold scaling
        )
        
        # Create initial mask for significant color changes
        significant_changes = delta_e > adaptive_threshold
        
        # Additional filtering: Only keep changes in regions with sufficient contrast
        # This helps ignore changes in very dark or very bright uniform areas
        gray_before = cv2.cvtColor(img_before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(img_after, cv2.COLOR_RGB2GRAY)
        
        # Calculate local contrast (standard deviation in 5x5 neighborhood)
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean_before = cv2.filter2D(gray_before.astype(np.float32), -1, kernel)
        local_mean_after = cv2.filter2D(gray_after.astype(np.float32), -1, kernel)
        
        local_var_before = cv2.filter2D((gray_before.astype(np.float32) - local_mean_before)**2, -1, kernel)
        local_var_after = cv2.filter2D((gray_after.astype(np.float32) - local_mean_after)**2, -1, kernel)
        
        local_contrast = np.sqrt((local_var_before + local_var_after) / 2)
        
        # Only keep changes where there's sufficient local contrast (not in uniform areas)
        contrast_threshold = 10.0  # Minimum contrast required
        sufficient_contrast = local_contrast > contrast_threshold
        
        # Combine color change and contrast requirements
        final_mask = significant_changes & sufficient_contrast
        
        # Apply morphological operations to clean up small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        # Convert to proper binary mask
        return (final_mask * 255).astype(np.uint8)

    @staticmethod
    def _ssim_change_detection(img1, img2, threshold):
        """
        Use Structural Similarity Index to detect changes
        """
        # Calculate SSIM
        try:
            # Use SSIM if available
            if SKIMAGE_AVAILABLE:
                score, diff = ssim(img1, img2, full=True)
                diff = (1 - diff) * 255
            else:
                # Fallback: use a simple correlation-based approach
                diff = cv2.absdiff(img1, img2)
                # Apply adaptive threshold
                diff = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        except Exception as e:
            # Generic fallback in case of any error with SSIM calculation
            print(f"SSIM calculation failed: {e}")
            diff = cv2.absdiff(img1, img2)
            # Simple thresholding as last resort
            _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        
        # Apply threshold
        _, change_mask = cv2.threshold(diff.astype(np.uint8), threshold * 10, 255, cv2.THRESH_BINARY)
        return change_mask
    
    @staticmethod
    def _feature_change_detection(img1, img2):
        """
        Use feature detection to find meaningful changes
        """
        # Detect SIFT features
        sift = cv2.SIFT_create()
        
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        
        change_mask = np.zeros_like(img1, dtype=np.uint8)
        
        if des1 is not None and des2 is not None:
            # Match features
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
            
            # Create mask based on unmatched keypoints (potential changes)
            matched_kp1_idx = set(m.queryIdx for m in good_matches)
            matched_kp2_idx = set(m.trainIdx for m in good_matches)
            
            # Mark unmatched keypoints as potential changes
            for i, kp in enumerate(kp1):
                if i not in matched_kp1_idx:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    cv2.circle(change_mask, (x, y), 10, 255, -1)
            
            for i, kp in enumerate(kp2):
                if i not in matched_kp2_idx:
                    x, y = int(kp.pt[0]), int(kp.pt[1])
                    cv2.circle(change_mask, (x, y), 10, 255, -1)
        
        return change_mask
    
    @staticmethod
    def _adaptive_change_detection(img1, img2):
        """
        Use conservative adaptive thresholding for change detection
        Only detects significant structural changes, ignoring minor variations
        """
        # Calculate difference
        diff = cv2.absdiff(img1, img2)
        
        # Apply strong bilateral filter to heavily suppress noise
        diff_filtered = cv2.bilateralFilter(diff, 15, 100, 100)
        
        # Calculate adaptive threshold based on image statistics
        mean_diff = np.mean(diff_filtered)
        std_diff = np.std(diff_filtered)
        
        # Use a more conservative threshold - only detect changes significantly above noise
        conservative_threshold = max(
            30,  # Minimum absolute threshold
            mean_diff + 3.0 * std_diff  # Statistical outlier threshold (99.7% confidence)
        )
        
        # Apply conservative thresholding
        _, binary = cv2.threshold(diff_filtered, conservative_threshold, 255, cv2.THRESH_BINARY)
        
        # Apply stronger morphological operations to remove small artifacts
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Additional filtering: remove very small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        min_component_size = 100  # Minimum size for a valid change region
        
        filtered_binary = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
                filtered_binary[labels == i] = 255
        
        return filtered_binary
    
    @staticmethod
    def _post_process_mask(mask):
        """
        Clean up the change mask by removing noise while preserving meaningful changes
        """
        # Apply moderate noise removal
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
        
        # Fill holes to consolidate change regions
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
        # Remove small connected components moderately
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        
        # Calculate minimum area based on image size
        image_area = mask.shape[0] * mask.shape[1]
        min_area = max(100, int(image_area * 0.00005))  # At least 0.005% of image area
        
        cleaned_mask = np.zeros_like(mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_area:
                # Additional check: ensure the change region has reasonable aspect ratio
                # This helps filter out thin lines that might be artifacts
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                aspect_ratio = max(width, height) / max(min(width, height), 1)
                
                # Keep regions with reasonable aspect ratios (allow more elongated shapes)
                if aspect_ratio <= 20.0:  # Max aspect ratio of 20:1 (more permissive)
                    cleaned_mask[labels == i] = 255
        
        # Final smoothing to create cleaner boundaries
        kernel_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_CLOSE, kernel_smooth, iterations=1)
        
        return cleaned_mask
    
    @staticmethod
    def visualize_changes(image_before, image_after, change_result):
        """
        Create enhanced visualization with better change highlighting
        
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
        
        # Draw bounding boxes with different colors based on change area
        for i, bbox in enumerate(change_result['bounding_boxes']):
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            area = bbox['area']
            
            # Color based on change area (green for small, yellow for medium, red for large)
            if area < 500:
                color = (0, 255, 0)  # Green for small changes
            elif area < 2000:
                color = (255, 255, 0)  # Yellow for medium changes
            else:
                color = (255, 0, 0)  # Red for large changes
            
            # Draw on both images
            cv2.rectangle(vis_before, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(vis_after, (x, y), (x + w, y + h), color, 2)
            
            # Add area label with better positioning
            label = f"{area}px"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_after, (x, y - label_size[1] - 10), 
                         (x + label_size[0] + 5, y), color, -1)
            cv2.putText(vis_after, label, (x + 2, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add change region number
            cv2.putText(vis_after, f"#{i+1}", (x + w - 25, y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Create enhanced change overlay
        change_mask = change_result['change_mask']
        
        # Create a colored change overlay
        change_overlay = np.zeros_like(image_after)
        change_overlay[:, :, 0] = change_mask  # Red channel for changes
        
        # Create a more sophisticated blend
        # Make changes more visible with transparency
        vis_overlay = image_after.copy()
        change_indices = change_mask > 0
        vis_overlay[change_indices] = cv2.addWeighted(
            image_after[change_indices], 0.6, 
            change_overlay[change_indices], 0.4, 0
        )
        
        # Create enhanced difference visualization
        diff_img = change_result['difference_image']
        
        # Normalize difference image for better visualization
        diff_normalized = cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply different colormaps for better visualization
        diff_colored = cv2.applyColorMap(diff_normalized.astype(np.uint8), cv2.COLORMAP_HOT)
        diff_colored = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
        
        # Add statistics overlay on the difference map
        stats = change_result['statistics']
        stats_text = [
            f"Changed: {stats['change_percentage']:.2f}%",
            f"Regions: {stats['num_regions']}",
            f"Pixels: {stats['changed_pixels']:,}"
        ]
        
        # Add stats text to difference map
        diff_with_stats = diff_colored.copy()
        for i, text in enumerate(stats_text):
            cv2.putText(diff_with_stats, text, (10, 30 + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return {
            'before_annotated': vis_before,
            'after_annotated': vis_after,
            'change_overlay': vis_overlay,
            'difference_map': diff_with_stats
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

    @staticmethod
    def debug_change_detection(image_before, image_after, threshold=None):
        """
        Debug method to analyze change detection results and provide insights
        
        Returns detailed information about why changes were detected
        """
        if image_before is None or image_after is None:
            return None
            
        if threshold is None:
            threshold = config.CHANGE_THRESHOLD
            
        # Resize images to same size if different
        if image_before.shape != image_after.shape:
            h, w = min(image_before.shape[0], image_after.shape[0]), min(image_before.shape[1], image_after.shape[1])
            image_before = cv2.resize(image_before, (w, h))
            image_after = cv2.resize(image_after, (w, h))
        
        # Run each detection method separately
        change_mask_color = ChangeDetector._perceptual_color_change_detection(
            image_before, image_after, threshold
        )
        
        gray_before = cv2.cvtColor(image_before, cv2.COLOR_RGB2GRAY)
        gray_after = cv2.cvtColor(image_after, cv2.COLOR_RGB2GRAY)
        
        change_mask_ssim = ChangeDetector._ssim_change_detection(gray_before, gray_after, threshold)
        change_mask_adaptive = ChangeDetector._adaptive_change_detection(gray_before, gray_after)
        
        # Calculate statistics for each method
        total_pixels = image_before.shape[0] * image_before.shape[1]
        
        color_changes = np.sum(change_mask_color > 0)
        ssim_changes = np.sum(change_mask_ssim > 0)
        adaptive_changes = np.sum(change_mask_adaptive > 0)
        
        # Calculate color statistics
        lab_before = cv2.cvtColor(image_before, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab_after = cv2.cvtColor(image_after, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        delta_L = lab_after[:, :, 0] - lab_before[:, :, 0]
        delta_a = lab_after[:, :, 1] - lab_before[:, :, 1]
        delta_b = lab_after[:, :, 2] - lab_before[:, :, 2]
        delta_e = np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)
        
        return {
            'method_results': {
                'color_detection': {
                    'changed_pixels': int(color_changes),
                    'percentage': float(color_changes / total_pixels * 100),
                    'mask': change_mask_color
                },
                'ssim_detection': {
                    'changed_pixels': int(ssim_changes),
                    'percentage': float(ssim_changes / total_pixels * 100),
                    'mask': change_mask_ssim
                },
                'adaptive_detection': {
                    'changed_pixels': int(adaptive_changes),
                    'percentage': float(adaptive_changes / total_pixels * 100),
                    'mask': change_mask_adaptive
                }
            },
            'color_statistics': {
                'mean_delta_e': float(np.mean(delta_e)),
                'max_delta_e': float(np.max(delta_e)),
                'median_delta_e': float(np.median(delta_e)),
                'std_delta_e': float(np.std(delta_e)),
                'pixels_above_threshold_50': int(np.sum(delta_e > 50)),
                'pixels_above_threshold_100': int(np.sum(delta_e > 100))
            },
            'image_statistics': {
                'total_pixels': int(total_pixels),
                'image_shape': image_before.shape
            }
        }

change_detector = ChangeDetector()