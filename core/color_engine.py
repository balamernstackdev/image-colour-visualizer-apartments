import cv2
import numpy as np

class ColorEngine:
    @staticmethod
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    @staticmethod
    def apply_material(image_rgb, mask, hex_color, finish="Standard", intensity=1.0):
        """
        Apply color with material awareness (Physics-Based Rendering approximation).
        Uses Frequency Separation to preserve wall texture while changing albedo.
        """
        # 1. Setup
        if not np.any(mask):
            return image_rgb
            
        # Ensure mask is suitable for blending
        if mask.dtype == bool:
             mask = mask.astype(np.float32)
        elif mask.dtype == np.uint8:
             mask = mask.astype(np.float32) / 255.0

        # Expand mask for broadcasting
        mask_3ch = mask[:, :, None]
        
        # 2. Prepare Colors (LAB)
        target_rgb = ColorEngine.hex_to_rgb(hex_color)
        target_pixel = np.array([[[target_rgb[0], target_rgb[1], target_rgb[2]]]], dtype=np.uint8)
        target_lab = cv2.cvtColor(target_pixel.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
        target_L = target_lab[0, 0, 0]
        target_A = target_lab[0, 0, 1]
        target_B = target_lab[0, 0, 2]

        # 3. Frequency Separation (Texture vs Lighting)
        img_float = image_rgb.astype(np.float32) / 255.0
        img_lab = cv2.cvtColor(img_float, cv2.COLOR_RGB2Lab)
        L_channel, A_channel, B_channel = cv2.split(img_lab)

        # High Pass Filter (Texture)
        # We use a small blur to extract the fine grain
        blur_L = cv2.GaussianBlur(L_channel, (0, 0), sigmaX=3)
        texture_L = L_channel - blur_L
        
        # 4. Albedo Replacement (Lighting Logic)
        mask_u8 = (mask > 0.1).astype(np.uint8) * 255
        mean_L_orig = cv2.mean(L_channel, mask=mask_u8)[0] if np.any(mask_u8) else 50.0
        mean_L_orig = max(mean_L_orig, 1.0)

        # LIGHTING PRESERVATION: Blend Additive (depth) and Multiplicative (contrast)
        l_diff = blur_L - mean_L_orig
        l_ratio = blur_L / mean_L_orig
        
        # Target lighting = 80% additive jump + 20% shadow contrast
        new_L_base = target_L + (l_diff * 0.8)
        new_L_base = (new_L_base * 0.7) + (target_L * l_ratio * 0.3)
        new_L_base = np.clip(new_L_base, 0, 100)

        # 5. Re-integrate Texture & Finish Effects
        texture_boost = {"Matte": 1.3, "Glossy": 0.6, "Satin": 1.1}.get(finish, 1.0)
        new_L_final = new_L_base + (texture_L * texture_boost)
        
        if finish == "Glossy":
            specular = np.clip((L_channel - 75) * 4, 0, 100) / 100.0
            new_L_final += (specular * 15.0)
        elif finish == "Satin":
            specular = np.clip((L_channel - 85) * 2, 0, 100) / 100.0
            new_L_final += (specular * 8.0)
            
        # EXTRA: Subtle 'Soft Light' overlay of original L for micro-depth
        new_L_final = new_L_final * (1.0 + (L_channel - 50) / 250.0)

        # 7. Reconstruct & Integrate
        new_L_final = np.clip(new_L_final, 0, 100)
        
        # Blend 90% Target Color with 10% Original Reflected Color for integration
        new_A = (np.full_like(A_channel, target_A) * 0.9) + (A_channel * 0.1)
        new_B = (np.full_like(B_channel, target_B) * 0.9) + (B_channel * 0.1)
        
        new_lab = cv2.merge([new_L_final, new_A, new_B])
        new_rgb = cv2.cvtColor(new_lab, cv2.COLOR_Lab2RGB)
        
        # 8. Composite with Intensity
        final_mask = mask_3ch * intensity
        result = img_float * (1.0 - final_mask) + new_rgb * final_mask
            
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
