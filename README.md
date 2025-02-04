# Image Processing Scripts Collection

This repository contains a collection of Python scripts for various image processing tasks. Below you'll find descriptions of each script and its purpose.

## 1. Stitching Frames to Video
Located in `/stiching_frames_to_video/`

A utility for creating videos from image sequences. Features include:
- Frame stitching with configurable parameters
- Video creation with customizable settings
- Configuration via JSON file
- Main script: `create_video.py`

## 2. Photo Annotation Utility
Located in `/photo_anotation_util/`

A tool for annotating and processing photos with various features:
- Photo annotation capabilities
- Black border utility for image processing
- Configurable settings via JSON
- Main scripts:
  - `photo-annotator.py`: Main annotation tool
  - `black_border_util.py`: Utility for handling image borders

## 3. Pose Data Visualization
Located in `/pose-data-visualisation/`

Tools for pose detection and visualization:
- Pose detection from images
- Data extraction and visualization
- Support for processing weightlifter images
- Main scripts:
  - `pose-detection.py`: Basic pose detection
  - `pose-detection-extraction.py`: Enhanced pose detection with data extraction

## Setup and Requirements

Each project directory contains its own `requirements.txt` file for dependencies. To set up any project:

1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Linux/Mac
   # or
   .\env\Scripts\activate  # On Windows
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

Please refer to individual project directories for specific setup instructions and configurations. 