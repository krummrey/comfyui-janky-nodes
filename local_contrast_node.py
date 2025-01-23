# clahe_node.py
import numpy as np
import torch
import cv2

class LocalContrastNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ([
					# Basic Adjustments
					"Subtle (Low Contrast Boost)",
					"Balanced (General Purpose)",
					"Strong (High Contrast Push)",
	
					# Creative Effects
					"Gentle Glow (Soft Diffusion)",
					"Neon Edges (High Frequency Boost)",
					"Grunge (Aggressive Texture)",
					"Solarized (Posterization Prep)",
	
					# Technical Use Cases
					"Medical Imaging (Crisp Features)",
					"Low-Light Recovery (Shadow Lift)",
					"Aerial Imaging (Haze Reduction)",
					"Document Scan (Text Enhancement)",
	
					# Specialized Processing
					"Portrait Retouching (Skin Smoothing)",
					"Architecture (Structural Clarity)",
					"Nature (Foliage Detail)",
					"Night Photography (Noise-Aware)"
				], {
					"default": "Balanced (General Purpose)"
				}),
                "color_mode": (["YUV Luminance", "Grayscale", "RGB Channels"],),
                "output_channels": (["Preserve Input", "Force Grayscale"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_clahe"
    CATEGORY = "image/postprocessing"

    def apply_clahe(self, image, preset, color_mode, output_channels):
        # Define preset values
        preset_values = {
			# Basic Adjustments
			"Subtle (Low Contrast Boost)": {
				"clip_limit": 1.5,
				"tile_grid_size": 10
			},
			"Balanced (General Purpose)": {
				"clip_limit": 2.5,
				"tile_grid_size": 8
			},
			"Strong (High Contrast Push)": {
				"clip_limit": 4.0,
				"tile_grid_size": 6
			},

			# Creative Effects
			"Gentle Glow (Soft Diffusion)": {
				"clip_limit": 1.8,
				"tile_grid_size": 16
			},
			"Neon Edges (High Frequency Boost)": {
				"clip_limit": 5.0,
				"tile_grid_size": 4
			},
			"Grunge (Aggressive Texture)": {
				"clip_limit": 8.0,
				"tile_grid_size": 3
			},
			"Solarized (Posterization Prep)": {
				"clip_limit": 6.5,
				"tile_grid_size": 8
			},

			# Technical Use Cases
			"Medical Imaging (Crisp Features)": {
				"clip_limit": 3.0,
				"tile_grid_size": 12
			},
			"Low-Light Recovery (Shadow Lift)": {
				"clip_limit": 4.5,
				"tile_grid_size": 6
			},
			"Aerial Imaging (Haze Reduction)": {
				"clip_limit": 2.8,
				"tile_grid_size": 14
			},
			"Document Scan (Text Enhancement)": {
				"clip_limit": 3.2,
				"tile_grid_size": 5
			},

			# Specialized Processing
			"Portrait Retouching (Skin Smoothing)": {
				"clip_limit": 2.2,
				"tile_grid_size": 9
			},
			"Architecture (Structural Clarity)": {
				"clip_limit": 3.8,
				"tile_grid_size": 7
			},
			"Nature (Foliage Detail)": {
				"clip_limit": 4.2,
				"tile_grid_size": 8
			},
			"Night Photography (Noise-Aware)": {
				"clip_limit": 2.0,
				"tile_grid_size": 12
			}
		}

        # Get values from the selected preset
        params = preset_values[preset]
        clip_limit = params["clip_limit"]
        tile_grid_size = params["tile_grid_size"]

        # Convert tensor to numpy array
        batch_size, height, width, _ = image.shape
        result = []
        
        for img in image:
            # Convert to 8-bit format for OpenCV
            img_np = img.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # Create CLAHE object
            clahe = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(tile_grid_size, tile_grid_size)
            )

            # Process based on color mode
            if color_mode == "YUV Luminance":
                # Convert to YUV and process Y channel
                yuv = cv2.cvtColor(img_np, cv2.COLOR_RGB2YUV)
                yuv[:,:,0] = clahe.apply(yuv[:,:,0])
                processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            elif color_mode == "Grayscale":
                # Convert to grayscale and process
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                processed = clahe.apply(gray)
                if output_channels == "Preserve Input":
                    processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:  # RGB Channels
                # Process each channel separately
                channels = cv2.split(img_np)
                processed_channels = [clahe.apply(c) for c in channels]
                processed = cv2.merge(processed_channels)

            # Handle output channel conversion
            if output_channels == "Force Grayscale" and color_mode != "Grayscale":
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)

            # Convert back to tensor
            processed = processed.astype(np.float32) / 255.0
            if processed.ndim == 2:
                processed = np.expand_dims(processed, axis=-1)
            result.append(torch.from_numpy(processed))

        # Stack batch and ensure proper dimensions
        result = torch.stack(result, dim=0)
        if result.shape[-1] == 1 and output_channels == "Preserve Input":
            result = result.repeat(1, 1, 1, 3)
            
        return (result,)

NODE_CLASS_MAPPINGS = {"LocalContrastNode": LocalContrastNode}  
NODE_DISPLAY_NAME_MAPPINGS = {"LocalContrastNode": "Local Contrast Enhancement"}  