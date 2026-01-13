import cv2
import numpy as np
from PIL import Image
import streamlit as st

class ColorTransferEngine:
    @staticmethod
    def hex_to_rgb(hex_color):
        """Convert HEX string to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def apply_color(image_rgb, mask, target_color_hex, intensity=1.0):
        """
        Apply color with improved texture preservation using Luminosity blending simulation.
        """
        # Ensure input is standard format
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. Create Smooth Mask (Feathering)
        mask_float = mask.astype(np.float32)
        # Gentle blur for anti-aliasing edges
        mask_soft = cv2.GaussianBlur(mask_float, (5, 5), 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Prepare Target Color
        target_rgb = ColorTransferEngine.hex_to_rgb(target_color_hex)
        target_bgr = np.array([target_rgb[2], target_rgb[1], target_rgb[0]], dtype=np.uint8)
        
        # Create a solid color image of the same size
        colored_layer = np.full_like(image_rgb, target_rgb, dtype=np.uint8)
        
        # 3. Apply Color with Multiply/Overlay style blending to preserve shadows
        # Convert to LAB to handle lightness separately
        img_float = image_rgb.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
        L, A, B = cv2.split(img_lab)
        
        # Target color in LAB
        target_pixel = np.array([[[target_rgb[0], target_rgb[1], target_rgb[2]]]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_pixel.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        target_l = target_lab[0, 0, 0]
        target_a = target_lab[0, 0, 1]
        target_b = target_lab[0, 0, 2]
        
        # PRO COLOR ENGINE v2 (Asian Paints Quality)
        # Physics-based Albedo Simulation
        # L_out = L_orig * (Target_L / mean(L_orig))
        
        # Calculate mean luminance of the object (original albedo proxy)
        mask_u8 = (mask > 0).astype(np.uint8) * 255
        mean_l = cv2.mean(L, mask=mask_u8)[0]
        # Clamp mean to avoid division by zero or extreme blowing
        mean_l = max(mean_l, 10.0)
        
        # Scaling factor
        luminance_scale = target_l / mean_l
        
        # Apply scaling
        # We blend the scaling effect to allow for "Opacity" to control how much we overwrite
        # Actually, intensity controls transparency.
        # But for the base calculation, we want the "New Look".
        
        new_L = L.astype(np.float32) * luminance_scale
        new_L = np.clip(new_L, 0, 100).astype(np.float32)
        
        # Reconstruct LAB
        new_A = np.full_like(A, target_a)
        new_B = np.full_like(B, target_b)
        
        new_lab = cv2.merge([new_L, new_A, new_B])
        recolored_rgb = cv2.cvtColor(new_lab, cv2.COLOR_Lab2RGB)
        
        # 4. Blend based on mask
        # Opacity/Intensity modulation
        final_mask = mask_3ch * intensity
        
        result_float = (recolored_rgb * final_mask) + (img_float * (1.0 - final_mask))
        
        result_uint8 = np.clip(result_float * 255.0, 0, 255).astype(np.uint8)
        
        return result_uint8

    @staticmethod
    @st.cache_data
    def get_target_ab(color_hex):
        """Pre-calculate and cache the LAB A/B channels for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
    @staticmethod
    @st.cache_data
    def get_target_ab(color_hex):
        """Pre-calculate and cache the LAB A/B channels for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        return float(lab[0, 0, 1]), float(lab[0, 0, 2])

    @staticmethod
    @st.cache_data
    def get_target_l(color_hex):
        """Pre-calculate and cache the LAB L channel for a hex color."""
        rgb = ColorTransferEngine.hex_to_rgb(color_hex)
        pixel = np.array([[[rgb[0], rgb[1], rgb[2]]]], dtype=np.uint8)
        lab = cv2.cvtColor(pixel.astype(np.float32)/255.0, cv2.COLOR_RGB2Lab)
        return float(lab[0, 0, 0])

    @staticmethod
    def composite_multiple_layers(image_rgb, masks_data):
        """
        High-performance multi-layer compositor.
        Uses a single LAB merge pass and caches the expensive L channel.
        """
        if not masks_data:
            return image_rgb.copy()

        # 1. Get Base L channel (The texture/luminosity)
        # We cache this keyed by the base image identity to avoid heavy re-computation
        l_cache_key = f"base_l_{id(image_rgb)}"
        if l_cache_key not in st.session_state:
            # First time: full conversion
            img_f = image_rgb.astype(np.float32, copy=False) / 255.0
            img_lab = cv2.cvtColor(img_f, cv2.COLOR_RGB2Lab)
            L, A, B = cv2.split(img_lab)
            st.session_state[l_cache_key] = (L, A, B)
        
        L, base_A, base_B = st.session_state[l_cache_key]
        
        # 2. Accumulate A/B changes
        curr_L = L.astype(np.float32) # Convert to proper float accumulator
        curr_A = base_A.astype(np.float32)
        curr_B = base_B.astype(np.float32)

        for data in masks_data:
            mask = data['mask']
            color_hex = data.get('color')
            if not color_hex:
                continue
            
            # Shape Guard: skip masks that don't match current background size
            if mask.shape[:2] != curr_A.shape[:2]:
                continue
            
            # Use software-cached soft mask
            mask_soft = data.get('mask_soft')
            if mask_soft is None:
                mask_soft = cv2.GaussianBlur(mask.astype(np.float32, copy=False), (5, 5), 0)
                data['mask_soft'] = mask_soft

            # Get target LAB (cached for speed)
            target_a, target_b = ColorTransferEngine.get_target_ab(color_hex)
            target_l = ColorTransferEngine.get_target_l(color_hex)

            # Opacity Control
            opacity = data.get('opacity', 1.0)
            final_mask_soft = mask_soft * opacity
            
            # --- PRO LUMINANCE BLEND ---
            # Calculate object mean brightness to scale it
            # We assume the mask defines the object.
            # Performance note: cv2.mean is fast.
            mask_u8 = (mask > 0).astype(np.uint8) * 255
            mean_l = cv2.mean(L, mask=mask_u8)[0]
            mean_l = max(mean_l, 10.0) # Avoid zero-div
            
            # Scale L: L_new = L_orig * (Target / Mean)
            # This "Relights" the object to the match the paint's brightness
            factor = target_l / mean_l
            
            # Apply factor but blend with original based on selection opacity?
            # Actually, standard paint fully covers. 
            # We treat 'opacity' as Alpha.
             
            # Apply scaling
            layer_L_float = L.astype(np.float32) * factor
            
            # --- FINISH APPLICATION (Matte/Glossy) ---
            finish = data.get('finish', 'Standard')
            if finish != 'Standard':
                # Center around mean lightness to adjust contrast
                mean_l_val = np.mean(layer_L_float[mask_u8 > 0])
                
                if finish == 'Matte':
                    # Reduce contrast (0.85x) for flatter look
                    contrast = 0.85 
                elif finish == 'Glossy':
                    # Increase contrast (1.25x) for shiny look
                    contrast = 1.25
                else:
                    contrast = 1.0
                
                # Apply contrast: New = Mean + (Old - Mean) * Contrast
                layer_L_float = mean_l_val + (layer_L_float - mean_l_val) * contrast
            
            layer_L_float = np.clip(layer_L_float, 0, 255)
            
            # Cumulative Blend for L
            # Mix layer_L into curr_L
            curr_L = (layer_L_float * final_mask_soft) + (curr_L * (1.0 - final_mask_soft))

            # Cumulative Blend A/B
            # Result = Target * Mask + Background * (1 - Mask)
            curr_A = (target_a * final_mask_soft) + (curr_A * (1.0 - final_mask_soft))
            curr_B = (target_b * final_mask_soft) + (curr_B * (1.0 - final_mask_soft))

        # 3. Single Re-Merge
        # Use the modified curr_L
        curr_L = np.clip(curr_L, 0, 100).astype(np.float32)
        final_lab = cv2.merge([curr_L, curr_A, curr_B])
        final_rgb = cv2.cvtColor(final_lab, cv2.COLOR_Lab2RGB)
        
        return np.clip(final_rgb * 255.0, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_texture(image_rgb, mask, texture_rgb, opacity=0.8):
        """
        Apply a texture with blending to simulate surface material.
        """
        image_rgb = image_rgb.astype(np.uint8)
        
        # 1. Create Smooth Mask
        mask_float = mask.astype(np.float32)
        mask_soft = cv2.GaussianBlur(mask_float, (3, 3), 0)
        mask_3ch = np.stack([mask_soft] * 3, axis=-1)
        
        # 2. Tile Texture to fill image
        h, w, c = image_rgb.shape
        th, tw, tc = texture_rgb.shape
        #hello
        # Resize texture if too large (e.g. > 512px) to keep pattern visible
        if max(th, tw) > 512:
            scale = 512 / max(th, tw)
            texture_rgb = cv2.resize(texture_rgb, (0, 0), fx=scale, fy=scale)
            th, tw, tc = texture_rgb.shape
            
        tiled_texture = np.zeros_like(image_rgb)
        
        for i in range(0, h, th):
            for j in range(0, w, tw):
                # Calculate available space
                curr_h = min(th, h - i)
                curr_w = min(tw, w - j)
                tiled_texture[i:i+curr_h, j:j+curr_w] = texture_rgb[:curr_h, :curr_w]
                
        # 3. Blend Texture (Multiply/Overlay approach)
        # Simple Approach: Multiply original L with Texture
        
        img_float = image_rgb.astype(np.float32) / 255.0
        tex_float = tiled_texture.astype(np.float32) / 255.0
        
        # Luminosity preservation:
        # Result = Texture * Original_Luminance
        # This makes the texture look shadowed by the room's lighting.
        
        img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
        L, A, B = cv2.split(img_lab)
        
        # Blend: use texture color but keep original lightness structure
        # Optionally mix texture's own lightness with original lightness
        
        # Simplified: Alpha Blend texture over image, but modulated by mask
        # To make it look "on the wall", we ideally want:
        # Out = Texture * (Original_Gray) * 2.0 (Overlay-ish)
        
        gray = cv2.cvtColor(img_float, cv2.COLOR_RGB2GRAY)
        gray_3ch = np.stack([gray] * 3, axis=-1)
        
        # Hard Light / Multiply simulation
        blended = tex_float * gray_3ch * 1.5 # Boost brightness slightly
        
        blended = np.clip(blended, 0, 1.0)
        
        # 4. Composite
        # Result = (Blended * Mask * Opacity) + (Original * (1 - Mask*Opacity))
        
        final_mask = mask_3ch * opacity
        output = (blended * final_mask) + (img_float * (1.0 - final_mask))
        
        return np.clip(output * 255.0, 0, 255).astype(np.uint8)
