import cv2
import numpy as np

class QualityEngine:
    @staticmethod
    def refine_edge(image, mask):
        """
        Refine the mask edges using the image content (Guided Filter).
        Ensures the paint sticks to the wall edge, not the pixel grid.
        SURGICAL VERSION: Uses small radius and safety erosion to prevent leaks.
        """
        # Ensure mask is 0-255 uint8 for morphological Ops
        if mask.dtype == bool:
            mask_u8 = (mask.astype(np.uint8)) * 255
        else:
            mask_u8 = mask.astype(np.uint8)
            
        # 1. SAFETY MARGIN: Removed to prevent edge gaps/halo
        # kernel = np.ones((2, 2), np.uint8)
        # mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
        
        mask_float = mask_u8.astype(np.float32) / 255.0
            
        # Guide = Original Image (Gray)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        guide = cv2.normalize(gray, None, 0.0, 1.0, cv2.NORM_MINMAX, cv2.CV_32F)
        
        # Guided Filter Parameters
        # Radius 2 is sharper for 720p-1024p images.
        radius = 2 
        eps = 1e-4
        
        try:
            # Try official implementation first (Fastest)
            refined = cv2.ximgproc.guidedFilter(
                guide=guide, 
                src=mask_float, 
                radius=radius, 
                eps=eps
            )
        except AttributeError:
            # Fallback: Custom Guided Filter
            ksize = (2 * radius + 1, 2 * radius + 1)
            mean_I = cv2.blur(guide, ksize)
            mean_p = cv2.blur(mask_float, ksize)
            mean_Ip = cv2.blur(guide * mask_float, ksize)
            mean_II = cv2.blur(guide * guide, ksize)
            
            cov_Ip = mean_Ip - mean_I * mean_p
            var_I = mean_II - mean_I * mean_I
            
            a = cov_Ip / (var_I + eps)
            b = mean_p - a * mean_I
            
            mean_a = cv2.blur(a, ksize)
            mean_b = cv2.blur(b, ksize)
            refined = mean_a * guide + mean_b
        
        # 2. FINAL SHARPENING: Stronger stretch for crisp architecture edges
        refined = (refined - 0.3) / 0.4
        refined = np.clip(refined, 0, 1)
        
        return refined

    @staticmethod
    def super_res_enhance(image):
        """
        Placeholder for Upscaling.
        For now, apply a subtle Unsharp Mask to simulate clarity.
        """
        gaussian_3 = cv2.GaussianBlur(image, (0, 0), 2.0)
        unsharp_image = cv2.addWeighted(image, 1.5, gaussian_3, -0.5, 0, image)
        return unsharp_image
