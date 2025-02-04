#!/usr/bin/env python3
"""
MediaPipe Pose Detection Script
------------------------------

This script processes an image file using MediaPipe's pose detection model and
outputs a new image with the pose skeleton overlay.

Configuration:
    You can set default values by modifying the DEFAULT_CONFIG variables at the top
    of this script. Command-line arguments will override these defaults.

Usage:
    python pose_detection.py -i <input_image> -o <output_image> [options]

Required Arguments:
    -i, --input         Path to input image file (jpg, png)
    -o, --output        Path to save output image file

Optional Arguments:
    --line-color        Color for skeleton lines in B,G,R format (default: 255,0,0 for blue)
    --point-color       Color for skeleton points in B,G,R format (default: 0,255,0 for green)
    --line-thickness    Thickness of skeleton lines (default: 2)
    --point-radius      Radius of landmark points (default: 2)
    --model-complexity  MediaPipe model complexity (0, 1, or 2, default: 2)
    --min-confidence    Minimum confidence threshold (0.0-1.0, default: 0.5)

Example Usage:
    # Basic usage with default settings
    python pose_detection.py -i person.jpg -o output.jpg

    # Custom colors and thickness
    python pose_detection.py -i person.jpg -o output.jpg --line-color 0,0,255 --line-thickness 3

    # Full customization
    python pose_detection.py -i person.jpg -o output.jpg \
        --line-color 0,0,255 \
        --point-color 255,0,0 \
        --line-thickness 3 \
        --point-radius 4 \
        --model-complexity 1 \
        --min-confidence 0.7

Requirements:
    - OpenCV (cv2)
    - MediaPipe
    
Install dependencies:
    pip install opencv-python mediapipe
"""

import cv2
import mediapipe as mp
import argparse

# Default configuration - modify these values to change default behavior
DEFAULT_CONFIG = {
    # Input/Output paths (set to None to require command-line arguments)
    'INPUT_PATH': None,  # e.g., 'input/person.jpg'
    'OUTPUT_PATH': None, # e.g., 'output/result.jpg'
    
    # Visualization settings
    'LINE_COLOR': (61, 53, 119),    # Blue in BGR || RGB - SET RED (119, 53, 61)
    'POINT_COLOR': (233, 240, 245),   # Green in BGR || RGB - SET BACKGROUND (245, 240, 233)
    'LINE_THICKNESS': 40,
    'POINT_RADIUS': 20,
    
    # Model settings
    'MODEL_COMPLEXITY': 2,        # 0, 1, or 2
    'MIN_CONFIDENCE': 0.5         # 0.0 to 1.0
}

def parse_color(color_str):
    """Convert color string 'B,G,R' to tuple (B,G,R)"""
    try:
        b, g, r = map(int, color_str.split(','))
        if not all(0 <= x <= 255 for x in (b, g, r)):
            raise ValueError
        return (b, g, r)
    except:
        raise argparse.ArgumentTypeError(
            "Color must be in format 'B,G,R' with values between 0-255"
        )

def process_image(input_path, output_path, config):
    """
    Process an image with MediaPipe Pose and draw the skeleton overlay.
    
    Args:
        input_path (str): Path to input image
        output_path (str): Path to save output image
        config (dict): Configuration dictionary containing all parameters
    """
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=config['MODEL_COMPLEXITY'],
        enable_segmentation=True,
        min_detection_confidence=config['MIN_CONFIDENCE']
    )

    # Read the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError(f"Could not read image at {input_path}")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image
    results = pose.process(image_rgb)
    
    # Convert back to BGR for saving
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Draw pose landmarks
    if results.pose_landmarks:
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
        print("Pose landmarks detected and drawn")
    else:
        print("Warning: No pose landmarks detected in the image")
    
    # Save the output image
    cv2.imwrite(output_path, annotated_image)
    print(f"Processed image saved to: {output_path}")
    
    # Clean up
    pose.close()

def color_tuple_to_str(color_tuple):
    """Convert BGR color tuple to string format"""
    return f"{color_tuple[0]},{color_tuple[1]},{color_tuple[2]}"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply MediaPipe Pose detection to an image',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Set up arguments with defaults from DEFAULT_CONFIG
    parser.add_argument('-i', '--input',
                        default=DEFAULT_CONFIG['INPUT_PATH'],
                        required=DEFAULT_CONFIG['INPUT_PATH'] is None,
                        help='Path to input image')
    parser.add_argument('-o', '--output',
                        default=DEFAULT_CONFIG['OUTPUT_PATH'],
                        required=DEFAULT_CONFIG['OUTPUT_PATH'] is None,
                        help='Path to save output image')
    parser.add_argument('--line-color',
                        type=parse_color,
                        default=color_tuple_to_str(DEFAULT_CONFIG['LINE_COLOR']),
                        help=f'Line color in B,G,R format (default: {color_tuple_to_str(DEFAULT_CONFIG["LINE_COLOR"])})')
    parser.add_argument('--point-color',
                        type=parse_color,
                        default=color_tuple_to_str(DEFAULT_CONFIG['POINT_COLOR']),
                        help=f'Point color in B,G,R format (default: {color_tuple_to_str(DEFAULT_CONFIG["POINT_COLOR"])})')
    parser.add_argument('--line-thickness',
                        type=int,
                        default=DEFAULT_CONFIG['LINE_THICKNESS'],
                        help=f'Line thickness (default: {DEFAULT_CONFIG["LINE_THICKNESS"]})')
    parser.add_argument('--point-radius',
                        type=int,
                        default=DEFAULT_CONFIG['POINT_RADIUS'],
                        help=f'Point radius (default: {DEFAULT_CONFIG["POINT_RADIUS"]})')
    parser.add_argument('--model-complexity',
                        type=int,
                        choices=[0, 1, 2],
                        default=DEFAULT_CONFIG['MODEL_COMPLEXITY'],
                        help=f'Model complexity (default: {DEFAULT_CONFIG["MODEL_COMPLEXITY"]})')
    parser.add_argument('--min-confidence',
                        type=float,
                        default=DEFAULT_CONFIG['MIN_CONFIDENCE'],
                        help=f'Minimum confidence threshold (default: {DEFAULT_CONFIG["MIN_CONFIDENCE"]})')
    
    args = parser.parse_args()
    
    # Validate parameters
    if args.line_thickness < 1:
        parser.error("Line thickness must be at least 1")
    if args.point_radius < 1:
        parser.error("Point radius must be at least 1")
    if not (0.0 <= args.min_confidence <= 1.0):
        parser.error("Minimum confidence must be between 0.0 and 1.0")
    
    # Create runtime configuration by updating defaults with command-line arguments
    runtime_config = DEFAULT_CONFIG.copy()
    runtime_config.update({
        'LINE_COLOR': args.line_color,
        'POINT_COLOR': args.point_color,
        'LINE_THICKNESS': args.line_thickness,
        'POINT_RADIUS': args.point_radius,
        'MODEL_COMPLEXITY': args.model_complexity,
        'MIN_CONFIDENCE': args.min_confidence
    })
    
    # Process the image
    process_image(args.input, args.output, runtime_config)