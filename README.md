# Janky Post-Processing Nodes for ComfyUI

Experimental nodes for image post-processing in ComfyUI. 

## Features

### Informative Drawing Node
![Sample](https://github.com/krummrey/comfyui-janky-nodes/blob/main/examples/Informative%20Drawing%20Sample.png)
- Implementation of [Informative Drawings](https://github.com/carolineec/informative-drawings) project
- Supports custom trained models
- Anime-style drawing conversion
- Aspect ratio preservation

### 2. Image Enhancement Nodes:
- **Edge-Aware Smoothing**:
![Sample](https://github.com/krummrey/comfyui-janky-nodes/blob/8982ae83c54efcef231ecd41ce96821a3b2568b4/examples/Edge%20Aware%20Smoothing%20Sample.png)
  - 5 preset modes (Cartoon, Noise Reduction, etc.)
  - Grayscale/RGB processing
  - Iterative filtering

- **Local Contrast Enhancement**:
![Sample](https://github.com/krummrey/comfyui-janky-nodes/blob/8982ae83c54efcef231ecd41ce96821a3b2568b4/examples/Local%20Contrast%20Enhancer%20Sample.png)
  - 15 creative presets (Neon Glow, Grunge, etc.)
  - YUV/RGB processing modes
  - CLAHE-based enhancement

## Installation

1. **Clone Repository**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/comfyui-janky-nodes.git```
   
2. **Load models**:
	download the models provided on the [original repo](https://github.com/carolineec/informative-drawings) and place them in:
	```models/informative_drawing_models/```
   
## Credits
- Informative Drawings implementation based on original work by [Caroline Chan](https://github.com/carolineec/informative-drawings)
- OpenCV ximgproc module for advanced computer vision algorithms
