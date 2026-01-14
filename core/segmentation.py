import numpy as np
import torch
import cv2
import logging
import os
import numpy as np
from mobile_sam import sam_model_registry, SamPredictor

# Force disable OpenCL to prevent "Bad Argument" / "UMat" errors on Streamlit Cloud
cv2.ocl.setUseOpenCL(False)
os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEGMENTATION_VERSION = "1.5.2"

class SegmentationEngine:
    def __init__(self, checkpoint_path=None, model_type="vit_b", device=None, model_instance=None):
        """
        Initialize the SAM model.
        Args:
            checkpoint_path: Path to weights (if loading new).
            model_type: SAM architecture type.
            device: 'cuda' or 'cpu'.
            model_instance: Pre-loaded sam_model_registry instance (optional).
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initialized SegmentationEngine v{SEGMENTATION_VERSION} on {self.device}")
        if model_instance is not None:
             self.sam = model_instance
        elif checkpoint_path:
             logger.info(f"Loading SAM model ({model_type}) on {self.device}...")
             self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
             self.sam.to(device=self.device)
        else:
             raise ValueError("Either checkpoint_path or model_instance must be provided.")

        self.predictor = SamPredictor(self.sam)
        self.is_image_set = False

    def set_image(self, image_rgb):
        """
        Process the image and compute embeddings.
        Args:
            image_rgb: NumPy array (H, W, 3) in RGB format.
        """
        logger.info("Computing image embeddings...")
        self.predictor.set_image(image_rgb)
        self.is_image_set = True
        logger.info("Embeddings computed.")
        self.image_rgb = image_rgb # Store for cleanup logic

    def generate_mask(self, point_coords, point_labels=None, level=None, cleanup=True):
        """
        Generate a mask for a given point.
        Args:
            point_coords: List of [x, y] or NumPy array.
            point_labels: List of labels (1 for foreground, 0 for background).
            level: int (0, 1, 2) or None. 
                   0=Fine Details, 1=Sub-segment, 2=Whole Object. 
                   If None, auto-selects highest score.
            cleanup: bool. If True, removes disconnected components to prevent leaks.
        """
        if not self.is_image_set:
            raise RuntimeError("Image not set. Call set_image() first.")

        if point_labels is None:
            point_labels = [1] * len(point_coords)

        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        with torch.inference_mode():
            masks, scores, logits = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True # Generate multiple masks and choose best
            )

        if level is not None and 0 <= level <= 2:
            # User forced a specific level
            best_mask = masks[level]
        else:
            # Heuristic: Choose the mask with the highest score
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
        
        if cleanup:
            # Post-processing: Filter disconnected components
            # We only want the component that contains the clicked point.
            
            # Ensure mask is uint8 for OpenCV
            mask_uint8 = (best_mask * 255).astype(np.uint8)
            # --- ADAPTIVE THRESHOLDS ---
            # Use strict guards for "Big Surfaces" (Level 2) to prevent bleeding.
            # Use relaxed guards for "Small Details" (Level 0) or Auto to fill gaps/shadows.
            if level == 2: # Big Surfaces
                thresh_intensity = 100 # Balanced: Allows DEEP shadows on white (was 65)
                thresh_edge = 45      # Sweet Spot: Ignores molding shadows (was 40)
                erode_iters = 0       # No barrier thickening: Max coverage
                kernel_size = 7       # Strong healing: Smooths jagged lines (was 5)
            elif level == 0: # Small Details (Strict Precision)
                thresh_intensity = 45 # Strict: Stops at minor color changes
                thresh_edge = 40      # Greedy: Prioritize coverage (was 35)
                erode_iters = 0       # Max Coverage
                kernel_size = 3       # Low healing to avoid bridging gaps
            else: # Auto (Smart Balanced) - Furniture
                thresh_intensity = 60 # Relaxed: Allows fabric gradients
                thresh_edge = 55      # Soft Object: Ignores deep wrinkles (was 50)
                erode_iters = 0       
                kernel_size = 7       # Strong healing: Bridges wrinkle gaps (was 5)           # --- SMART COLOR SAFETY CHECK ---
            # Re-enabled with Chromaticity Logic to fix Leaking AND Shadows.
            if hasattr(self, 'image_rgb'):
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = self.image_rgb.shape[:2]
                cx, cy = max(0, min(cx, w-1)), max(0, min(cy, h-1))
                
                # Sample Seed
                seed_color = self.image_rgb[cy, cx].astype(np.float32)
                
                # Check if seed is Grayscale (Saturation check)
                seed_sat = np.max(seed_color) - np.min(seed_color)
                is_grayscale_seed = seed_sat < 20 # Low saturation
                
                # 1. Chromaticity (Color only, invariant to brightness/shadows)
                # OPTIMIZATION: Use uint16 for distance check to avoid heavy float32 conversion
                img_u16 = self.image_rgb.astype(np.uint16)
                img_sum = np.sum(img_u16, axis=2, keepdims=True)
                img_sum[img_sum == 0] = 1 # Prevent div by zero
                
                # Implemented robustly
                img_chroma = (img_u16[:, :, :2] << 8) // img_sum 
                seed_chroma = (seed_color[:2].astype(np.uint16) << 8) // np.sum(seed_color + 0.1)
                
                # Color Distance
                chroma_dist = np.sum(np.abs(img_chroma - seed_chroma), axis=2)
                
                # 2. Intensity (Brightness)
                intensity_dist = np.abs(np.mean(img_u16, axis=2) - np.mean(seed_color))
                
                # 3. Hybrid Thresholding
                if is_grayscale_seed:
                    # BALANCED GUARD (v1.5.5): Adaptive Tolerance
                    # Even for white walls, reject Strong Colors (like yellow sofa)
                    # Use relaxed chroma (60) to allow tinted shadows but stop objects.
                    valid_mask = (intensity_dist < 210) & (chroma_dist < 60)
                else:
                    # Chroma 38 is approx 0.15 in fixed point (0.15 * 256)
                    valid_mask = (chroma_dist < 38) & (intensity_dist < 180)

                valid_mask = valid_mask.astype(np.uint8)
                
                # Intersect with SAM mask
                mask_refined = (mask_uint8 & valid_mask)
                
                # --- EDGE GUARD (v1.5.1 - Restored for Line Detection) ---
                # Detects physical lines (shadows/creases) between similar colored objects
                try:
                    gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
                    # Use lighter blur to keep sharp lines
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    
                    # Standard Sobel
                    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                    grad = np.sqrt(sobelx**2 + sobely**2)
                    
                    # Normalize to 0-255
                    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    # Threshold to find strong lines (The "Barrier")
                    # BALANCED GUARD (v1.5.5): Adaptive Edge Threshold
                    
                    # DYNAMIC OVERRIDE: If painting White/Gray surfaces, use STRICTER edges (35)
                    # to catch faint shadow lines (like wall vs ceiling).
                    # If painting Colors, use RELAXED edges (45) to ignore molding shadows.
                    effective_thresh_edge = 45 if is_grayscale_seed and level == 2 else thresh_edge
                    
                    _, edge_mask = cv2.threshold(grad_norm, effective_thresh_edge, 255, cv2.THRESH_BINARY_INV)
                    
                    # Erode slightly to thicken the barrier
                    kernel = np.ones((2, 2), np.uint8)
                    edge_mask = cv2.erode(edge_mask, kernel, iterations=erode_iters)
                    
                    # Apply Edge Barrier
                    mask_refined = cv2.bitwise_and(mask_refined, mask_refined, mask=edge_mask)
                    
                    # BALANCED GUARD (v1.5.5): Adaptive Healing
                    kernel_heal = np.ones((kernel_size, kernel_size), np.uint8)
                    mask_refined = cv2.morphologyEx(mask_refined, cv2.MORPH_CLOSE, kernel_heal)
                    
                except Exception as e:
                    logger.error(f"Edge Guard Failed: {e}")

                # --- TEXTURE GUARD (v1.5.6) ---
                # Exclude high-frequency areas (screens, art, textured objects) even if color matches.
                # Use the 'grad_norm' we already computed for edges.
                # Walls are flat (Low Gradient). Detailed objects are busy (High Gradient).
                thresh_texture = 30 # Sensitivity: Higher = allows more lighting variations (was 15)
                
                # We need a dense variance map, not just edges.
                # Use a larger blur for "Busy-ness" detection
                # Optimization: Re-use 'gray' but compute local variance or just use dense edges
                
                # Simple Proxy: Check if area has HIGH density of edges
                texture_mask = (grad_norm < thresh_texture).astype(np.uint8)
                
                # Intersect: Must be Color Valid AND Low Texture (Flat)
                # But allow edges themselves (don't erase constraints).
                # Actually, simply AND-ing can kill valid textured walls (bricks).
                # Only apply Texture Guard if we are in "Level 2 (Big Surfaces)" where we expect flatness.
                # AND Only if NOT Grayscale/White Seed (to allow white moldings/shadows).
                if level == 2 and not is_grayscale_seed:
                    mask_refined = mask_refined & texture_mask
                
                # Safety Fallback for Textured Walls:
                # If we killed >50% of the mask, maybe it IS a textured wall.
                # For now, prioritize NOT painting TVs.
                
                # Safety Fallback
                if np.sum(mask_refined) > 50: # At least some pixels survived
                    mask_uint8 = mask_refined
            
            # Check if the click point is actually inside the mask (it should be, but just in case)
            if len(point_coords) > 0:
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                
                # Find connected components
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                
                if num_labels > 1:
                    h, w = mask_uint8.shape
                    cx = max(0, min(cx, w - 1))
                    cy = max(0, min(cy, h - 1))
                    
                    target_label = labels_im[cy, cx]
                    
                    if target_label != 0:
                        # Create a new mask keeping only the target component
                        best_mask = (labels_im == target_label)
                    else:
                        # Fallback: if click was somehow outside, keep largest component ignoring background
                        max_area = 0
                        max_label = 1
                        for i in range(1, num_labels):
                            if stats[i, cv2.CC_STAT_AREA] > max_area:
                                max_area = stats[i, cv2.CC_STAT_AREA]
                                max_label = i
                        best_mask = (labels_im == max_label)
        
        return best_mask
