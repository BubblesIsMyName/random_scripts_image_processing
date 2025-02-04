import subprocess
from pathlib import Path
from PIL import Image
import json
import sys
import time

def parse_resolution_ratio(ratio_str):
    """
    Parse resolution ratio string (e.g., "3:5" or "1:2")
    Returns the decimal ratio or None if invalid/not provided
    """
    if not ratio_str:
        return None
        
    try:
        numerator, denominator = map(int, ratio_str.split(':'))
        if denominator == 0:
            print("Warning: Invalid resolution ratio (division by zero). Using original dimensions.")
            return None
        return numerator / denominator
    except (ValueError, AttributeError):
        print("Warning: Invalid resolution ratio format. Using original dimensions.")
        return None

def apply_resolution_ratio(width, height, ratio):
    """
    Apply resolution ratio to dimensions
    """
    if ratio is None:
        return width, height
        
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # Ensure dimensions are at least 2 pixels
    new_width = max(2, new_width)
    new_height = max(2, new_height)
    
    return new_width, new_height

def ensure_even_dimensions(width, height):
    """
    Ensure dimensions are even numbers as required by libx264.
    
    Args:
        width: Original width
        height: Original height
        
    Returns:
        tuple: (width, height) adjusted to even numbers
    """
    return (width if width % 2 == 0 else width - 1,
            height if height % 2 == 0 else height - 1)

def get_first_image_dimensions(root_dir):
    """
    Get dimensions of the first image file found in the root directory.
    
    Args:
        root_dir: Path object pointing to directory with images
        
    Returns:
        tuple: (width, height) of the first image, or None if no images found
    """
    # Look for image files in the directory
    image_files = sorted(root_dir.glob("*.jpeg")) + sorted(root_dir.glob("*.jpg")) + \
                  sorted(root_dir.glob("*.png")) + sorted(root_dir.glob("*.webp"))
    
    if not image_files:
        return None
        
    try:
        with Image.open(image_files[0]) as img:
            width, height = img.size
            # Ensure dimensions are even numbers
            return ensure_even_dimensions(width, height)
    except Exception as e:
        print(f"Error reading image dimensions: {e}")
        return None

def create_video_from_images(image_pattern, output_file, frame_rate, width, height):
    """
    Creates a video from a sequence of images using ffmpeg.

    Args:
        image_pattern: A string pattern for the input image files (e.g., "%d.jpeg", "%03d.png").
        output_file: The name of the output video file (e.g., "output.mp4").
        frame_rate: The frame rate of the output video (e.g., "1/2" for a slow slideshow, "1" for 1 second per image).
        width: The desired width of the output video (must be even).
        height: The desired height of the output video (must be even).
    """
    # Double-check dimensions are even
    width, height = ensure_even_dimensions(width, height)

    command = [
        "ffmpeg",
        "-framerate", frame_rate,
        "-i", f'{str(image_pattern)}',  # Handle spaces in paths
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",  # Maintain aspect ratio and pad
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_file),  # Convert Path to string for ffmpeg
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()  # Wait for the process to finish

        if process.returncode == 0:
            print(f"Video created successfully: {output_file}")
        else:
            print(f"Error creating video:")
            print(stdout.decode())
            print(stderr.decode())

    except FileNotFoundError:
        print("Error: ffmpeg not found. Please ensure it's installed and in your PATH.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def print_config_preview(config, config_path):
    """Print the configuration and ask for user confirmation."""
    print("\nConfiguration found at:", config_path)
    print("\nConfigurations to process:")
    for key, cfg in config.items():
        print(f"\n{key}:")
        print(json.dumps(cfg, indent=4))
    print("\nDo you want to proceed with these configurations?")
    print("Press Enter to continue or Ctrl+C to abort...")

def load_config(root_folder=None):
    """
    Load configuration from JSON file.
    First tries to find config in root_folder if provided,
    then falls back to script directory.
    """
    script_dir = Path(__file__).parent
    config_locations = []
    
    if root_folder:
        root_folder = Path(root_folder)
        config_locations.append((root_folder / "config.json", "provided root folder"))
    
    # Always check script directory as fallback
    config_locations.append((script_dir / "config.json", "script directory"))
    
    for config_path, location_type in config_locations:
        print(f"Checking {config_path} for config.json")
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"\nUsing config file from {location_type}: {config_path}")
                return config, config_path
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in config file at {config_path}")
                continue
    
    print("Error: No valid configuration file found.")
    print("Please create a config.json file with the following structure:")
    print("""
{
    "slow": {
        "root_dir": "/path/to/your/images",
        "movement_type": "clean",
        "name": "my_video",
        "frame_rate": "3",
        "input_image_pattern": "%d.jpeg",
        "resolution_ratio": "3:5"
    },
    "medium": {
        "root_dir": "/path/to/your/images",
        "movement_type": "clean",
        "name": "my_video",
        "frame_rate": "5",
        "input_image_pattern": "%d.jpeg",
        "resolution_ratio": "3:5"
    }
}
    """)
    sys.exit(1)

def process_single_config(config_key, config_data):
    """Process a single configuration and create its video."""
    try:
        root_dir = Path(config_data['root_dir'])
        movement_type = config_data['movement_type']
        name = config_data['name']
        frame_rate = config_data['frame_rate']
        input_image_pattern = config_data['input_image_pattern']
        # Get optional resolution ratio
        resolution_ratio = config_data.get('resolution_ratio')
    except KeyError as e:
        print(f"Error in configuration '{config_key}': Missing required key: {e}")
        print(f"Required keys are: root_dir, movement_type, name, frame_rate, input_image_pattern")
        print(f"Optional keys are: resolution_ratio (e.g., '3:5' for 3/5 of original resolution)")
        return False
    
    print(f"\nProcessing configuration: {config_key}")
    
    # Get dimensions from first image
    dimensions = get_first_image_dimensions(root_dir)
    if dimensions:
        width, height = dimensions
        print(f"Original source image dimensions: {width}x{height}")
        
        # Apply resolution ratio if provided
        ratio = parse_resolution_ratio(resolution_ratio)
        if ratio:
            width, height = apply_resolution_ratio(width, height, ratio)
            print(f"Scaled dimensions (ratio {resolution_ratio}): {width}x{height}")
            
        # Ensure dimensions are even
        width, height = ensure_even_dimensions(width, height)
        print(f"Final dimensions (after ensuring even numbers): {width}x{height}")
    else:
        print("Using default dimensions")
        width = 1280
        height = 720
    
    image_pattern = root_dir / input_image_pattern
    output_file = root_dir / f"{config_key}_{movement_type}_{name}_{width}x{height}_{frame_rate.replace('/', '_')}.mp4"

    create_video_from_images(image_pattern, output_file, frame_rate, width, height)
    return True

if __name__ == "__main__":
    # Get root folder from command line argument if provided
    root_folder = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Load configuration
    config, config_path = load_config(root_folder)
    
    # Print configuration preview and wait for user confirmation
    print_config_preview(config, config_path)
    try:
        input()
    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(0)
    
    # Process each configuration
    successful = 0
    failed = 0
    
    for config_key, config_data in config.items():
        print(f"\n{'='*50}")
        print(f"Processing {config_key}")
        print(f"{'='*50}")
        
        if process_single_config(config_key, config_data):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"{'='*50}")

