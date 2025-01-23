# Janky Post-Processing Nodes for ComfyUI

![Banner Placeholder](screenshots/banner.png) *<!-- Add banner later -->*

Experimental nodes for image post-processing in ComfyUI. Handle with care - these are "works in progress" with rough edges!

## Features

### 🎨 1. Informative Drawing Node
- Implementation of [Informative Drawings](https://github.com/carolineec/informative-drawings) project
- Supports custom trained models
- Anime-style drawing conversion
- Aspect ratio preservation

### ✨ 2. Image Enhancement Nodes:
- **Edge-Aware Smoothing**:
  - 5 preset modes (Cartoon, Noise Reduction, etc.)
  - Grayscale/RGB processing
  - Iterative filtering

- **Local Contrast Enhancement**:
  - 15 creative presets (Neon Glow, Grunge, etc.)
  - YUV/RGB processing modes
  - CLAHE-based enhancement

## Installation

1. **Clone Repository**:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/yourusername/comfyui-janky-nodes.git
   
## Credits
- Informative Drawings implementation based on original work by [Caroline Chan](https://github.com/carolineec/informative-drawings)
- OpenCV ximgproc module for advanced computer vision algorithms
