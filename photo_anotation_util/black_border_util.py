from PIL import Image
import os
from pathlib import Path

def add_instagram_border(input_path, output_path=None):
    """
    Add black borders to an image to make it square for Instagram.
    The original image is centered in the square.
    
    Args:
        input_path (str): Path to the input image
        output_path (str, optional): Path for the output image. If None, will create
            a new file with '_instagram' suffix in the same directory.
    
    Returns:
        str: Path to the output image

    TODO:
    - Update the script to output the images in a spearate folder in the folder above with suffix _bb
    """
    # Open the image
    with Image.open(input_path) as img:
        # Convert to RGB if image is in RGBA mode
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        # Get image dimensions
        width, height = img.size
        
        # Calculate the target size (use the larger dimension)
        target_size = max(width, height)
        
        # Create new black square image
        new_img = Image.new('RGB', (target_size, target_size), 'black')
        
        # Calculate position to paste original image (centered)
        paste_x = (target_size - width) // 2
        paste_y = (target_size - height) // 2
        
        # Paste original image onto black background
        new_img.paste(img, (paste_x, paste_y))
        
        # Generate output path if not provided
        if output_path is None:
            input_file = Path(input_path)
            output_path = input_file.parent / f"{input_file.stem}_instagram{input_file.suffix}"
        
        # Save the new image
        new_img.save(output_path, quality=95)
        
        return output_path

def process_directory(input_dir, output_dir=None):
    """
    Process all images in a directory.
    
    Args:
        input_dir (str): Directory containing images to process
        output_dir (str, optional): Directory for output images. If None,
            will create images in the same directory as input.
    """
    input_path = Path(input_dir)
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for file in input_path.iterdir():
        if file.suffix.lower() in image_extensions:
            if output_dir:
                output_file = output_path / f"{file.stem}_instagram{file.suffix}"
                add_instagram_border(str(file), str(output_file))
            else:
                add_instagram_border(str(file))

# Example usage
if __name__ == "__main__":
    # Process a single image

    # image_path = "/home/gatis/Pictures/IG_posts/spain_w_joe_24/inthe_bowl_w_joe.jpeg"
    # add_instagram_border(image_path)
    
    folder_path = "/home/gatis/Pictures/IG_posts/porto_icSPORTS24/"
    # Or process an entire directory
    process_directory(folder_path)