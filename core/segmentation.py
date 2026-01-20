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

    def generate_mask(self, point_coords, point_labels=None, level=None, cleanup=True, use_texture_guard=True):
        """
        Generate a mask for a given point.
        Args:
            point_coords: List of [x, y] or NumPy array.
            point_labels: List of labels (1 for foreground, 0 for background).
            level: int (0, 1, 2) or None. 
                   0=Fine Details, 1=Sub-segment, 2=Whole Object. 
                   If None, auto-selects highest score.
            cleanup: bool. If True, removes disconnected components to prevent leaks.
            use_texture_guard: bool. If True, detects edges and prevents leaks into high-detail areas.
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
            if level == 2:
                best_mask = masks[1] 
            else:
                best_mask = masks[level]
        else:
            best_idx = np.argmax(scores)
            best_mask = masks[best_idx]
        
        mask_uint8 = (best_mask * 255).astype(np.uint8)

        if cleanup:
            # --- HOLE FILLING FIRST (NEW) ---
            # Fill small holes caused by lighting variations BEFORE applying fences
            # Increased kernel size for better coverage of shadow gaps
            kernel_fill = np.ones((9, 9), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_fill)
            
            # --- COLOR-BASED WALL SEPARATION (Stricter for Precision) ---
            # This prevents selecting the entire building when clicking one wall
            if use_texture_guard and self.image_rgb is not None and len(point_coords) > 0:
                # Get the clicked point's color
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                h, w = self.image_rgb.shape[:2]
                cx, cy = max(0, min(cx, w - 1)), max(0, min(cy, h - 1))
                
                # Sample a small region around the click for color reference
                sample_size = 15  # Reduced sample size for more local precision
                y1, y2 = max(0, cy - sample_size), min(h, cy + sample_size)
                x1, x2 = max(0, cx - sample_size), min(w, cx + sample_size)
                clicked_color = np.median(self.image_rgb[y1:y2, x1:x2], axis=(0, 1))
                
                # Calculate color distance map
                color_diff = np.sqrt(np.sum((self.image_rgb.astype(float) - clicked_color) ** 2, axis=2))
                
                # Create color fence: areas that are too different in color
                # RELAXED THRESHOLDS to reduce holes while keeping basic separation
                if level == 2:  # Walls/Ceilings
                    color_threshold = 60  # Slightly higher to tolerate shadows
                else:
                    color_threshold = 60  # Increased from 40 for better shadow tolerance
                    
                color_fence = (color_diff > color_threshold).astype(np.uint8) * 255
                
                # Dilation to close small gaps in the fence
                color_fence = cv2.dilate(color_fence, np.ones((3, 3), np.uint8), iterations=2)
                
                # Apply color fence
                mask_uint8[color_fence > 0] = 0

                # Additional closing to seal any remaining small gaps after fence removal
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            
            # --- SURGICAL BOUNDARY PROTECTION ---
            if use_texture_guard and self.image_rgb is not None:
                gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
                # Stronger edge detection to find wall corners/edges
                # Low threshold 30 (was 20) ignores more texture noise
                edges = cv2.Canny(gray, 30, 80)  
                fence = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
                
                # Apply strictly
                mask_uint8[fence > 0] = 0

            # Basic Morphological Cleanup
            if level == 2: # Walls/Large Surfaces
                # 1. Open/Close to remove noise
                # Reduced to 3x3 to preserve sharp corners and edges
                kernel_clean = np.ones((3, 3), np.uint8)
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_clean)
                
                # 2. REDUNDANT EROSION REMOVED (It was causing double shrinkage)
                
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_clean)
            else:
                kernel_small = np.ones((3, 3), np.uint8)
                mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_small)
            
            # Connected components to keep only the clicked area
            if len(point_coords) > 0:
                cx, cy = int(point_coords[0][0]), int(point_coords[0][1])
                num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                
                if num_labels > 1:
                    h, w = mask_uint8.shape
                    cx, cy = max(0, min(cx, w - 1)), max(0, min(cy, h - 1))
                    target_label = labels_im[cy, cx]
                    
                    if target_label != 0:
                        best_mask = (labels_im == target_label)
                    else:
                        # Fallback: Largest
                        max_area = 0
                        max_label = 1
                        for i in range(1, num_labels):
                            if stats[i, cv2.CC_STAT_AREA] > max_area:
                                max_area = stats[i, cv2.CC_STAT_AREA]
                                max_label = i
                        best_mask = (labels_im == max_label)
                else:
                    best_mask = (mask_uint8 > 0)
        
        return best_mask
