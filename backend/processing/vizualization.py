import numpy as np
import cv2
from processing.image_loader import image_loader

class Visualizer:
    @staticmethod
    def create_side_by_side(image_before, image_after, labels=True):
        """
        Create side-by-side comparison
        
        Args:
            image_before: numpy array
            image_after: numpy array
            labels: add "Before" and "After" labels
        
        Returns:
            combined image
        """
        if image_before is None or image_after is None:
            return None
        
        # Ensure same size
        h1, w1 = image_before.shape[:2]
        h2, w2 = image_after.shape[:2]
        
        if (h1, w1) != (h2, w2):
            image_after = cv2.resize(image_after, (w1, h1))
        
        # Add padding between images
        padding = np.ones((h1, 10, 3), dtype=np.uint8) * 255
        
        # Concatenate
        combined = np.hstack([image_before, padding, image_after])
        
        if labels:
            # Add labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness = 2
            color = (255, 255, 255)
            
            # Before label
            cv2.putText(combined, "BEFORE", (20, 40),
                       font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(combined, "BEFORE", (20, 40),
                       font, font_scale, color, thickness)
            
            # After label
            after_x = w1 + 20
            cv2.putText(combined, "AFTER", (after_x, 40),
                       font, font_scale, (0, 0, 0), thickness + 2)
            cv2.putText(combined, "AFTER", (after_x, 40),
                       font, font_scale, color, thickness)
        
        return combined
    
    @staticmethod
    def add_statistics_overlay(image, statistics, bbox_size_degrees=0.1):
        """
        Add statistics text overlay to image
        
        Args:
            image: numpy array
            statistics: dict with change statistics
            bbox_size_degrees: size of bbox for area calculation
        
        Returns:
            image with overlay
        """
        if image is None or not statistics:
            return image
        
        img = image.copy()
        h, w = img.shape[:2]
        
        # Create semi-transparent overlay panel
        overlay = img.copy()
        panel_height = 120
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)
        y_start = h - panel_height + 25
        line_height = 25
        
        # Calculate area
        from processing.change_detection import change_detector
        area_km2 = change_detector.calculate_area_km2(
            statistics['changed_pixels'],
            w, h, bbox_size_degrees
        )
        
        texts = [
            f"Changed Pixels: {statistics['changed_pixels']:,}",
            f"Change Percentage: {statistics['change_percentage']:.2f}%",
            f"Affected Area: {area_km2:.2f} km²",
            f"Detected Regions: {statistics['num_regions']}"
        ]
        
        for i, text in enumerate(texts):
            y = y_start + (i * line_height)
            cv2.putText(img, text, (10, y), font, font_scale, color, thickness)
        
        return img
    
    @staticmethod
    def create_change_heatmap(difference_image):
        """
        Create heatmap visualization of changes
        
        Args:
            difference_image: grayscale difference image
        
        Returns:
            colored heatmap
        """
        if difference_image is None:
            return None
        
        # Normalize
        diff_norm = cv2.normalize(difference_image, None, 0, 255, cv2.NORM_MINMAX)
        diff_norm = diff_norm.astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        return heatmap
    
    @staticmethod
    def add_scale_bar(image, bbox_size_degrees=0.1, position='bottom-right'):
        """
        Add approximate scale bar to image
        
        Args:
            image: numpy array
            bbox_size_degrees: size of bbox in degrees
            position: 'bottom-right', 'bottom-left', etc.
        
        Returns:
            image with scale bar
        """
        if image is None:
            return None
        
        img = image.copy()
        h, w = img.shape[:2]
        
        # Calculate scale (approximate)
        km_per_degree = 111  # at equator
        total_km = bbox_size_degrees * km_per_degree
        
        # Scale bar: 10km or 1km depending on total size
        if total_km > 20:
            bar_km = 10
        elif total_km > 5:
            bar_km = 5
        else:
            bar_km = 1
        
        bar_width_px = int((bar_km / total_km) * w)
        bar_height = 5
        
        # Position
        margin = 20
        if 'right' in position:
            x1 = w - bar_width_px - margin
        else:
            x1 = margin
        
        if 'bottom' in position:
            y1 = h - margin - 30
        else:
            y1 = margin + 30
        
        x2 = x1 + bar_width_px
        y2 = y1 + bar_height
        
        # Draw scale bar
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 1)
        
        # Add label
        label = f"{bar_km} km"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        text_x = x1 + (bar_width_px - text_size[0]) // 2
        text_y = y1 - 5
        
        # Text with outline
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1)
        cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        
        return img
    
    @staticmethod
    def create_analysis_dashboard(image_before, image_after, change_mask, statistics, bbox_size_degrees=0.1):
        """
        Create comprehensive analysis dashboard with multiple views
        
        Args:
            image_before: numpy array - before image
            image_after: numpy array - after image  
            change_mask: numpy array - binary change mask
            statistics: dict - change statistics
            bbox_size_degrees: float - bbox size for area calculation
            
        Returns:
            combined dashboard image
        """
        if any(img is None for img in [image_before, image_after, change_mask]):
            return None
            
        # Resize all images to consistent size
        target_size = (400, 400)
        img_before = cv2.resize(image_before, target_size)
        img_after = cv2.resize(image_after, target_size)
        mask_resized = cv2.resize(change_mask, target_size)
        
        # Create change overlay on after image
        change_overlay = img_after.copy()
        change_colored = np.zeros_like(img_after)
        change_colored[mask_resized > 0] = [255, 0, 0]  # Red for changes
        change_overlay = cv2.addWeighted(change_overlay, 0.7, change_colored, 0.3, 0)
        
        # Create heatmap
        heatmap = Visualizer.create_change_heatmap(mask_resized)
        if heatmap is not None:
            heatmap = cv2.resize(heatmap, target_size)
        else:
            heatmap = np.zeros((*target_size[::-1], 3), dtype=np.uint8)
        
        # Create 2x2 grid
        top_row = np.hstack([img_before, img_after])
        bottom_row = np.hstack([change_overlay, heatmap])
        dashboard = np.vstack([top_row, bottom_row])
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        color = (255, 255, 255)
        outline_color = (0, 0, 0)
        
        labels = ["BEFORE", "AFTER", "CHANGES OVERLAY", "CHANGE HEATMAP"]
        positions = [(20, 40), (420, 40), (20, 440), (420, 440)]
        
        for label, (x, y) in zip(labels, positions):
            # Add outline
            cv2.putText(dashboard, label, (x, y), font, font_scale, outline_color, thickness + 2)
            # Add main text
            cv2.putText(dashboard, label, (x, y), font, font_scale, color, thickness)
        
        # Add statistics panel at bottom
        stats_panel_height = 100
        h, w = dashboard.shape[:2]
        stats_panel = np.zeros((stats_panel_height, w, 3), dtype=np.uint8)
        
        # Add statistics text
        if statistics:
            from processing.change_detection import change_detector
            area_km2 = change_detector.calculate_area_km2(
                statistics['changed_pixels'], w//2, h//2, bbox_size_degrees
            )
            
            stats_text = [
                f"Analysis Results:",
                f"Changed Pixels: {statistics['changed_pixels']:,} | "
                f"Change %: {statistics['change_percentage']:.2f}% | "
                f"Area: {area_km2:.2f} km² | "
                f"Regions: {statistics['num_regions']}"
            ]
            
            y_pos = 25
            for text in stats_text:
                cv2.putText(stats_panel, text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 255, 255), 1)
                y_pos += 25
        
        # Combine dashboard with stats panel
        final_dashboard = np.vstack([dashboard, stats_panel])
        
        return final_dashboard
    
    @staticmethod
    def create_temporal_comparison(images_list, timestamps=None, title="Temporal Analysis"):
        """
        Create temporal comparison of multiple images
        
        Args:
            images_list: list of numpy arrays
            timestamps: list of timestamp strings (optional)
            title: string title for the comparison
            
        Returns:
            combined temporal comparison image
        """
        if not images_list or len(images_list) < 2:
            return None
        
        # Resize all images to same size
        target_size = (300, 300)
        resized_images = []
        
        for img in images_list:
            if img is not None:
                resized = cv2.resize(img, target_size)
                resized_images.append(resized)
        
        if not resized_images:
            return None
        
        # Arrange images horizontally
        combined = np.hstack(resized_images)
        
        # Add timestamps if provided
        if timestamps and len(timestamps) == len(resized_images):
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            color = (255, 255, 255)
            outline_color = (0, 0, 0)
            
            for i, timestamp in enumerate(timestamps):
                x = i * target_size[0] + 10
                y = 30
                
                # Add outline
                cv2.putText(combined, timestamp, (x, y), font, font_scale, outline_color, thickness + 1)
                # Add main text
                cv2.putText(combined, timestamp, (x, y), font, font_scale, color, thickness)
        
        # Add title
        if title:
            title_height = 50
            title_panel = np.zeros((title_height, combined.shape[1], 3), dtype=np.uint8)
            
            cv2.putText(title_panel, title, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.0, (255, 255, 255), 2)
            
            combined = np.vstack([title_panel, combined])
        
        return combined
    
    @staticmethod
    def save_visualization(image, filename, quality=95):
        """
        Save visualization to file
        
        Args:
            image: numpy array
            filename: output filename
            quality: JPEG quality (0-100)
            
        Returns:
            bool - success status
        """
        if image is None:
            return False
        
        try:
            # Convert RGB to BGR for OpenCV
            if len(image.shape) == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Save with appropriate format
            if filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
                cv2.imwrite(filename, image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(filename, image_bgr)
            
            return True
        except Exception as e:
            print(f"Error saving visualization: {e}")
            return False

# Create global instance
visualizer = Visualizer()