import numpy as np
from PIL import Image
import io
import base64

class ImageLoader:
    @staticmethod
    def load_from_bytes(image_bytes):
        """
        Load image from bytes and convert to numpy array
        
        Args:
            image_bytes: Image data as bytes
        
        Returns:
            numpy array (H, W, C) or None
        """
        if not image_bytes:
            return None
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_array = np.array(image)
            
            # Convert to RGB if grayscale
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            # Ensure 3 channels
            if image_array.shape[2] == 4:
                image_array = image_array[:, :, :3]
            
            return image_array
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    @staticmethod
    def array_to_base64(image_array):
        """
        Convert numpy array to base64 encoded string
        
        Args:
            image_array: numpy array (H, W, C)
        
        Returns:
            base64 string
        """
        if image_array is None:
            return None
        
        try:
            # Ensure uint8
            if image_array.dtype != np.uint8:
                image_array = (image_array * 255).astype(np.uint8)
            
            image = Image.fromarray(image_array)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            buffer.seek(0)
            
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            return img_base64
        except Exception as e:
            print(f"Error converting to base64: {e}")
            return None
    
    @staticmethod
    def normalize_sar_image(image_array):
        """
        Normalize SAR image values to 0-255 range
        
        Args:
            image_array: numpy array
        
        Returns:
            normalized numpy array (uint8)
        """
        if image_array is None:
            return None
        
        # Convert to float
        img = image_array.astype(np.float32)
        
        # Normalize each channel independently
        for i in range(img.shape[2]):
            channel = img[:, :, i]
            
            # Remove outliers (clip to 2nd and 98th percentile)
            p2, p98 = np.percentile(channel, [2, 98])
            channel = np.clip(channel, p2, p98)
            
            # Normalize to 0-1
            if p98 - p2 > 0:
                channel = (channel - p2) / (p98 - p2)
            
            img[:, :, i] = channel
        
        # Convert to uint8
        img = (img * 255).astype(np.uint8)
        
        return img
    
    @staticmethod
    def convert_to_grayscale(image_array):
        """
        Convert RGB image to grayscale
        
        Args:
            image_array: numpy array (H, W, 3)
        
        Returns:
            grayscale array (H, W)
        """
        if image_array is None:
            return None
        
        if len(image_array.shape) == 2:
            return image_array
        
        # Standard RGB to grayscale conversion
        gray = np.dot(image_array[..., :3], [0.299, 0.587, 0.114])
        return gray.astype(np.uint8)

image_loader = ImageLoader()