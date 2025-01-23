# fast_global_smoother_node.py
import numpy as np
import torch
import cv2

class SmoothingNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": ([
					# Core Presets
					"Low Noise Reduction",
					"Mild Noise Reduction",
					"Moderate Noise Reduction", 
					"Strong Noise Reduction",
					"Severe Noise Reduction",
				
					# Experimental Presets
					"Moderate Detail Enhancement",
					"Micro-Texture Amplifier",
					"Astrophotography Cleanup",
					"Ink Wash Painting",
					"Forensic Detail Recovery",
					"Neon Glow Preparation",
					"Sensor Failure Artifact Repair"], {
                    "default": "Mild Smoothing"
                }),
                "color_mode": (["RGB Channels", "Grayscale"],),
                "output_channels": (["Preserve Input", "Force Grayscale"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_smoothing"
    CATEGORY = "image/postprocessing"

    def apply_smoothing(self, image, preset, color_mode, output_channels):
        # Define preset values
        preset_values = {
			# Noise Reduction Series
			"Low Noise Reduction": {
				"lambda_": 500.0,
				"sigma_color": 0.5,
				"lambda_attenuation": 0.25,
				"num_iterations": 3
			},
			"Mild Noise Reduction": {
				"lambda_": 750.,
				"sigma_color": 1.0,
				"lambda_attenuation": 0.25,
				"num_iterations": 3
			},
			"Moderate Noise Reduction": {
				"lambda_": 1000,
				"sigma_color": 1.5,
				"lambda_attenuation": 0.25,
				"num_iterations": 3
			},
			"Strong Noise Reduction": {
				"lambda_": 1500,
				"sigma_color": 2.0,
				"lambda_attenuation": 0.20,
				"num_iterations": 4
			},
			"Severe Noise Reduction": {
				"lambda_": 3000.0,
				"sigma_color": 3.0,
				"lambda_attenuation": 0.15,
				"num_iterations": 5
			},

			# ===== Experimental Presets =====
			# Detail Enhancement Series
			"Moderate Detail Enhancement": {
				"lambda_": 1500.0,
				"sigma_color": 0.5,
				"lambda_attenuation": 0.5,
				"num_iterations": 5
			},

			# Edge Case Specializations
			"Micro-Texture Amplifier": {
				"lambda_": 800.0,
				"sigma_color": 0.3,
				"lambda_attenuation": 0.8,
				"num_iterations": 7  # Extreme iterations for texture analysis
			},
	
			"Astrophotography Cleanup": {
				"lambda_": 2500.0,
				"sigma_color": 4.0,
				"lambda_attenuation": 0.15,  # Maintain strong smoothing
				"num_iterations": 4  # For starfield noise patterns
			},

			# Hybrid Effects
			"Ink Wash Painting": {
				"lambda_": 1800.0,
				"sigma_color": 12.0,  # Aggressive color merging
				"lambda_attenuation": 0.9,  # Progressive softening
				"num_iterations": 5
			},

			# Technical Applications
			"Forensic Detail Recovery": {
				"lambda_": 300.0,
				"sigma_color": 0.1,  # Ultra-edge preservation
				"lambda_attenuation": 0.05,  # Consistent processing
				"num_iterations": 10  # Iterative refinement
			},

			# Creative Stylization
			"Neon Glow Preparation": {
				"lambda_": 1200.0,
				"sigma_color": 8.0,
				"lambda_attenuation": 0.6,
				"num_iterations": 3  # For glowing edge effects
			},

			# Extreme Case Handling
			"Sensor Failure Artifact Repair": {
				"lambda_": 3500.0,
				"sigma_color": 5.0,
				"lambda_attenuation": 0.05,  # Maximum persistent smoothing
				"num_iterations": 5  # For hardware defect compensation
			}
		}

        # Get values from the selected preset
        params = preset_values[preset]
        lambda_ = params["lambda_"]
        sigma_color = params["sigma_color"]
        lambda_attenuation = params["lambda_attenuation"]
        num_iterations = params["num_iterations"]

        # Convert tensor to numpy array
        batch_size, height, width, _ = image.shape
        result = []
        
        for img in image:
            # Convert to 8-bit format for OpenCV
            img_np = img.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            
            # Convert to grayscale if needed
            if color_mode == "Grayscale":
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)  # Ensure 3 channels for filter

            # Create Fast Global Smoother filter
            smoother = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide=img_np,
                lambda_=lambda_,
                sigma_color=sigma_color,
                lambda_attenuation=lambda_attenuation,
                num_iter=num_iterations
            )

            # Apply smoothing
            smoothed = smoother.filter(img_np)

            # Handle output channel conversion
            if output_channels == "Force Grayscale":
                smoothed = cv2.cvtColor(smoothed, cv2.COLOR_RGB2GRAY)
                smoothed = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2RGB)

            # Convert back to tensor
            smoothed = smoothed.astype(np.float32) / 255.0
            if smoothed.ndim == 2:
                smoothed = np.expand_dims(smoothed, axis=-1)
            result.append(torch.from_numpy(smoothed))

        # Stack batch and ensure proper dimensions
        result = torch.stack(result, dim=0)
        if result.shape[-1] == 1 and output_channels == "Preserve Input":
            result = result.repeat(1, 1, 1, 3)
            
        return (result,)

NODE_CLASS_MAPPINGS = {"SmoothingNode": SmoothingNode} 
NODE_DISPLAY_NAME_MAPPINGS = {"SmoothingNode": "Edge-Aware Smoothing"} 