#!/usr/bin/env python3
"""
Enhanced MediaPipe Pose Detection and Data Extraction Script
---------------------------------------------------------

This script processes a folder of image files using MediaPipe's pose detection model and:
1. Outputs images with pose skeleton overlays
2. Extracts pose landmark data to a dataframe
3. Saves the dataframe as a parquet file
4. Maintains a processing log

Usage:
    python pose-detection-extraction.py -i <input_folder> -o <output_folder> [options]

Required Arguments:
    -i, --input         Path to input folder containing images
    -o, --output        Path to output folder for processed files

Optional Arguments:
    --line-color        Color for skeleton lines in B,G,R format (default: 255,0,0 for blue)
    --point-color       Color for skeleton points in B,G,R format (default: 0,255,0 for green)
    --line-thickness    Thickness of skeleton lines (default: 2)
    --point-radius      Radius of landmark points (default: 2)
    --model-complexity  MediaPipe model complexity (0, 1, or 2, default: 2)
    --min-confidence    Minimum confidence threshold (0.0-1.0, default: 0.5)
    --normalize         Normalize coordinates relative to image dimensions (default: False)

NB! NORMALIZATION NOT TESTED
Normalization performed by:

Using the image dimensions as the reference frame Converting all x,y coordinates to be
relative to image width/height (resulting in values between 0-1) For z coordinates, normalizing
relative to the depth range in the frame using min-max normalization

This makes the pose data comparable across images of different resolutions and 
helps reduce the impact of camera distance variations.

"""

import cv2
import mediapipe as mp
import argparse
import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

# Default configuration
DEFAULT_CONFIG = {
    'INPUT_PATH': None,
    'OUTPUT_PATH': None,
    'LINE_COLOR': (61, 53, 119),
    'POINT_COLOR': (233, 240, 245),
    'LINE_THICKNESS': 4,
    'POINT_RADIUS': 2,
    'MODEL_COMPLEXITY': 2,
    'MIN_CONFIDENCE': 0.5,
    'NORMALIZE': False
}

def setup_logging(output_folder):
    """Set up logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(output_folder, f'pose_processing_{timestamp}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_image_creation_time(image_path):
    """Extract image creation timestamp from metadata"""
    try:
        return datetime.fromtimestamp(os.path.getctime(image_path))
    except Exception as e:
        logging.warning(f"Could not extract creation time for {image_path}: {e}")
        return datetime.now()

def normalize_coordinates(landmarks, image_width, image_height):
    """Normalize landmark coordinates relative to image dimensions"""
    normalized_landmarks = []
    
    if not landmarks:
        return None
        
    # Extract all z coordinates for z-normalization
    z_coords = [landmark.z for landmark in landmarks.landmark]
    z_min, z_max = min(z_coords), max(z_coords)
    z_range = z_max - z_min if z_max != z_min else 1.0

    for landmark in landmarks.landmark:
        # x and y are normalized by image dimensions
        norm_x = landmark.x / image_width
        norm_y = landmark.y / image_height
        # z is normalized to range [0,1] based on the frame's depth range
        norm_z = (landmark.z - z_min) / z_range if z_range != 0 else 0
        
        normalized_landmarks.extend([norm_x, norm_y, norm_z, landmark.visibility])
    
    return normalized_landmarks

def extract_landmarks(landmarks, image_width=None, image_height=None, normalize=False):
    """Extract landmark coordinates and visibility scores"""
    if not landmarks:
        return None
        
    if normalize and (image_width is not None) and (image_height is not None):
        return normalize_coordinates(landmarks, image_width, image_height)
    
    # Extract raw coordinates
    coordinates = []
    for landmark in landmarks.landmark:
        coordinates.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return coordinates

def process_image(input_path, output_path, config, df_poses):
    """Process a single image and update the poses dataframe"""
    logger = logging.getLogger(__name__)
    
    # Check if image was already processed
    output_image_path = os.path.join(output_path, f"processed_{os.path.basename(input_path)}")
    if os.path.exists(output_image_path):
        logger.info(f"Skipping already processed image: {input_path}")
        return df_poses

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=config['MODEL_COMPLEXITY'],
        enable_segmentation=True,
        min_detection_confidence=config['MIN_CONFIDENCE']
    )

    try:
        # Read and process image
        image = cv2.imread(input_path)
        if image is None:
            logger.error(f"Could not read image: {input_path}")
            return df_poses

        # Get image dimensions for normalization
        height, width = image.shape[:2]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            logger.warning(f"No pose landmarks detected in: {input_path}")
            return df_poses
            
        # Extract landmarks
        landmarks = extract_landmarks(
            results.pose_landmarks,
            width,
            height,
            config['NORMALIZE']
        )
        
        if landmarks:
            # Create pose data row
            participant = os.path.splitext(os.path.basename(input_path))[0]
            creation_time = get_image_creation_time(input_path)
            
            # Prepare landmark columns
            landmark_data = {}
            for i, coord in enumerate(landmarks):
                coord_type = i % 4  # 0=x, 1=y, 2=z, 3=visibility
                landmark_idx = i // 4
                if coord_type == 0:
                    landmark_data[f'x{landmark_idx}'] = coord
                elif coord_type == 1:
                    landmark_data[f'y{landmark_idx}'] = coord
                elif coord_type == 2:
                    landmark_data[f'z{landmark_idx}'] = coord
                else:
                    landmark_data[f'v{landmark_idx}'] = coord
            
            # Add row to dataframe
            new_row = {
                'participant': participant,
                'timestamp': creation_time,
                **landmark_data
            }
            df_poses = pd.concat([df_poses, pd.DataFrame([new_row])], ignore_index=True)
            
            # Draw pose landmarks on image
            annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=config['POINT_COLOR'],
                    thickness=config['LINE_THICKNESS'],
                    circle_radius=config['POINT_RADIUS']
                ),
                mp_drawing.DrawingSpec(
                    color=config['LINE_COLOR'],
                    thickness=config['LINE_THICKNESS']
                )
            )
            
            # Save processed image
            cv2.imwrite(output_image_path, annotated_image)
            logger.info(f"Processed image saved: {output_image_path}")
            
        pose.close()
        return df_poses
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        return df_poses

def main():
    parser = argparse.ArgumentParser(
        description='Process folder of images for pose detection and data extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Set up arguments
    parser.add_argument('-i', '--input', required=True, help='Input folder path')
    parser.add_argument('-o', '--output', required=True, help='Output folder path')
    parser.add_argument('--line-color', type=str, 
                       default=f"{DEFAULT_CONFIG['LINE_COLOR'][0]},{DEFAULT_CONFIG['LINE_COLOR'][1]},{DEFAULT_CONFIG['LINE_COLOR'][2]}")
    parser.add_argument('--point-color', type=str,
                       default=f"{DEFAULT_CONFIG['POINT_COLOR'][0]},{DEFAULT_CONFIG['POINT_COLOR'][1]},{DEFAULT_CONFIG['POINT_COLOR'][2]}")
    parser.add_argument('--line-thickness', type=int, default=DEFAULT_CONFIG['LINE_THICKNESS'])
    parser.add_argument('--point-radius', type=int, default=DEFAULT_CONFIG['POINT_RADIUS'])
    parser.add_argument('--model-complexity', type=int, choices=[0,1,2], default=DEFAULT_CONFIG['MODEL_COMPLEXITY'])
    parser.add_argument('--min-confidence', type=float, default=DEFAULT_CONFIG['MIN_CONFIDENCE'])
    parser.add_argument('--normalize', action='store_true', default=DEFAULT_CONFIG['NORMALIZE'],
                       help='Normalize coordinates relative to image dimensions')
    
    args = parser.parse_args()
    
    # Validate and create output directory
    output_dir = Path(args.output)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    # Setup logging
    logger = setup_logging(args.output)
    logger.info(f"Starting pose detection processing")
    logger.info(f"Input folder: {args.input}")
    logger.info(f"Output folder: {args.output}")
    
    # Initialize poses dataframe
    df_poses = pd.DataFrame()
    
    # Get list of image files
    image_files = [f for f in os.listdir(args.input) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    total_images = len(image_files)
    logger.info(f"Found {total_images} images to process")
    
    # Process each image
    for idx, image_file in enumerate(image_files, 1):
        input_path = os.path.join(args.input, image_file)
        logger.info(f"Processing image {idx}/{total_images}: {image_file}")
        
        df_poses = process_image(
            input_path,
            args.output,
            {
                'LINE_COLOR': tuple(map(int, args.line_color.split(','))),
                'POINT_COLOR': tuple(map(int, args.point_color.split(','))),
                'LINE_THICKNESS': args.line_thickness,
                'POINT_RADIUS': args.point_radius,
                'MODEL_COMPLEXITY': args.model_complexity,
                'MIN_CONFIDENCE': args.min_confidence,
                'NORMALIZE': args.normalize
            },
            df_poses
        )
    
    # Save poses dataframe to parquet
    if not df_poses.empty:
        parquet_path = os.path.join(args.output, 'pose_landmarks.parquet')
        df_poses.to_parquet(parquet_path, index=False)
        logger.info(f"Saved pose landmarks to: {parquet_path}")
    else:
        logger.warning("No pose data was collected - dataframe is empty")
    
    logger.info("Processing completed")

if __name__ == "__main__":
    main()