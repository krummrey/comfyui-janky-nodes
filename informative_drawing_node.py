# informative_drawing_node.py
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import folder_paths
from .model import Generator  # Removed unused import: cv2
from torchvision.transforms.functional import resize

class InformativeDrawingNode:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

        self.models_dir = os.path.join(folder_paths.models_dir, "informative_drawing_models")
        self.model_files = self._scan_for_models()

    def _scan_for_models(self):
        """Scan the models directory for netG_A_latest.pth files and return a sorted list."""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            return []

        model_files = [os.path.join(root, "netG_A_latest.pth")
                      for root, _, files in os.walk(self.models_dir)
                      if "netG_A_latest.pth" in files]
        return sorted(model_files)

    @classmethod
    def INPUT_TYPES(cls):
        """Define the input types for the node."""
        model_files = cls()._scan_for_models()
        model_list = [os.path.basename(os.path.dirname(m)) for m in model_files] if model_files else ["default"]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_list,),
                "size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "invert_image": ("BOOLEAN", {"default": False}),  # Toggle switch for inverting the image
            }
        }

    RETURN_TYPES = ("IMAGE",)  # Output image tensor
    RETURN_NAMES = ("output_image",)  # Name of the output
    FUNCTION = "process_image"  # Function to call
    CATEGORY = "Image Processing"  # Category in ComfyUI

    def process_image(self, image, model_name, size, invert_image):
        """Process the input image using the selected model."""
        try:
            model_path = next((m for m in self.model_files if os.path.basename(os.path.dirname(m)) == model_name), None)
            if not model_path:
                raise ValueError(f"Model '{model_name}' not found in models directory: {self.models_dir}")
            
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load and prepare model
            net_G = Generator(input_nc=3, output_nc=1, n_residual_blocks=3).to(self.device)
            net_G.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            net_G.eval()
            
            # Directly resize the tensor using PyTorch functions

            # ComfyUI's IMAGE tensor is in BHWC format. Permute to BCHW for resizing.
            img_tensor = image.permute(0, 3, 1, 2)  # BHWC -> BCHW (batch, channels, height, width)
            img_tensor = resize(
                img_tensor,
                size=size,  # Resize smaller edge to "size" (maintains aspect ratio)
                interpolation=transforms.InterpolationMode.BICUBIC
            ).to(self.device)     
            
            with torch.no_grad():
                output = net_G(img_tensor)

            # Post-process output to match ComfyUI format
            output = output.cpu()

            # Scale output to [0, 1] and convert to 3-channel format
            output = (output + 1) * 0.5  # Scale to [0, 1]
            output = output.repeat(1, 3, 1, 1)  # Convert 1-channel output to 3-channel (grayscale to RGB)

            # Simplified normalization with full tonal range preservation
            min_val = output.min()
            max_val = output.max()
            output = (output - min_val) / (max_val - min_val + 1e-7)  # Add small epsilon to avoid division by zero

            # Invert the image if the toggle is enabled
            if invert_image:
                output = 1 - output  # Invert the image

            # Directly convert the tensor to ComfyUI format (BHWC with values in [0, 1])
            output = output.permute(0, 2, 3, 1)  # Convert from BCHW to BHWC

            # Return output in ComfyUI format (BHWC)
            return (output,)

        except Exception as e:
            print(f"Error in InformativeDrawingNode: {str(e)}")
            raise