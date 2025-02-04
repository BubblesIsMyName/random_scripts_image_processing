"""
Photo annotation tool with separate info panel window.
Supports JPG, JPEG, PNG, and ARW (Sony Raw) files.

Launch the script with "python photo_annotator.py"

Controls:
- 'q' to quit
- 'a' to view previous image
- 'd' to view next image
- 'w' to copy current image to destination folder
- 'e' to remove current image from destination folder
- 'i' to toggle info panel
- Mouse wheel to adjust info panel text size
"""

import cv2
import json
import os
from pathlib import Path
import shutil
from datetime import datetime
import ctypes
import rawpy
import gc
from functools import lru_cache
import signal
from contextlib import contextmanager
import logging
import numpy as np
import platform

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Timeout context manager that's platform-aware"""
    if platform.system() == 'Windows':
        # On Windows, we can't use SIGALRM
        yield
    else:
        def signal_handler(signum, frame):
            raise TimeoutException("Timed out!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

def get_screen_resolution():
    """Get screen resolution in a platform-independent way"""
    try:
        if platform.system() == 'Windows':
            user32 = ctypes.windll.user32
            return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
        else:
            # Fallback to a reasonable default if can't detect
            return 1920, 1080
    except:
        return 1920, 1080

class InfoPanel:
    DEFAULT_SETTINGS = {
        "width": 600,
        "height": 400,
        "font_size": 14,
        "background_color": [0, 0, 0],
        "text_color": [255, 255, 255],
        "visible": True,
        "position": {"x": 800, "y": 0}
    }

    def __init__(self, settings):
        # Deep merge of provided settings with defaults
        self.settings = self.DEFAULT_SETTINGS.copy()
        if "info_panel" in settings:
            self._update_nested_dict(self.settings, settings["info_panel"])
        
        self.visible = self.settings["visible"]
        self.window_name = "Info Panel"
        self.status_text = ""
        self._ensure_window_on_screen()
        self.setup_window()

    def _update_nested_dict(self, d, u):
        """Deep update of nested dictionary"""
        for k, v in u.items():
            if isinstance(v, dict) and k in d:
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v

    def _ensure_window_on_screen(self):
        """Ensure window position is within screen boundaries"""
        screen_width, screen_height = get_screen_resolution()
        self.settings["position"]["x"] = max(0, min(self.settings["position"]["x"], 
                                                  screen_width - self.settings["width"]))
        self.settings["position"]["y"] = max(0, min(self.settings["position"]["y"], 
                                                  screen_height - self.settings["height"]))

    def setup_window(self):
        if self.visible:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 
                           self.settings["width"], 
                           self.settings["height"])
            cv2.moveWindow(self.window_name, 
                         self.settings["position"]["x"], 
                         self.settings["position"]["y"])
            
            # Platform-specific mouse callback setup
            if platform.system() == 'Windows':
                cv2.setMouseCallback(self.window_name, self.mouse_callback_windows)
            else:
                cv2.setMouseCallback(self.window_name, self.mouse_callback_unix)

    def mouse_callback_windows(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if (flags >> 16) > 0 else -1
            self._adjust_font_size(delta)

    def mouse_callback_unix(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = 1 if flags > 0 else -1
            self._adjust_font_size(delta)

    def _adjust_font_size(self, delta):
        """Adjust font size with bounds checking"""
        new_size = max(8, min(30, self.settings["font_size"] + delta))
        if new_size != self.settings["font_size"]:
            self.settings["font_size"] = new_size

    def toggle(self):
        self.visible = not self.visible
        if self.visible:
            self.setup_window()
        else:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass  # Window might already be closed

    def set_status(self, text):
        self.status_text = text

    def render(self, text_lines):
        if not self.visible:
            return

        try:
            # Create blank panel
            panel = np.zeros((self.settings["height"], self.settings["width"], 3), dtype=np.uint8)
            panel[:] = self.settings["background_color"]

            # Calculate text positioning
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = self.settings["font_size"] / 14.0
            line_spacing = int(self.settings["font_size"] * 2)
            
            # Draw main text
            y_position = 40
            for line in text_lines:
                # Ensure text fits in window
                while True:
                    (text_width, _), _ = cv2.getTextSize(line, font, font_scale, 1)
                    if text_width < self.settings["width"] - 40 or font_scale <= 0.5:
                        break
                    font_scale *= 0.9

                cv2.putText(panel, line, (20, y_position), font, font_scale,
                           self.settings["text_color"], 1, cv2.LINE_AA)
                y_position += line_spacing

            # Draw status bar
            if self.status_text:
                # Draw separator line
                cv2.line(panel, 
                        (0, self.settings["height"]-40),
                        (self.settings["width"], self.settings["height"]-40),
                        self.settings["text_color"], 1)
                # Draw status text
                cv2.putText(panel, self.status_text,
                           (20, self.settings["height"]-15),
                           font, font_scale * 0.8,
                           self.settings["text_color"], 1, cv2.LINE_AA)

            cv2.imshow(self.window_name, panel)
        except Exception as e:
            logger.error(f"Error rendering info panel: {e}")

    def save_state(self):
        """Save current window state"""
        if self.visible:
            try:
                if hasattr(cv2, 'getWindowImageRect'):  # Check if method exists
                    pos = cv2.getWindowImageRect(self.window_name)
                    if pos is not None:
                        self.settings["position"]["x"] = pos[0]
                        self.settings["position"]["y"] = pos[1]
                        self.settings["width"] = pos[2]
                        self.settings["height"] = pos[3]
            except:
                pass  # Keep existing position if can't get current
        self.settings["visible"] = self.visible
        return self.settings

    def cleanup(self):
        """Cleanup resources"""
        if self.visible:
            try:
                cv2.destroyWindow(self.window_name)
            except:
                pass

class PhotoAnnotator:
    def __init__(self, settings_path="/home/gatis/Documents/OTHER/random_scripts/photo_anotation_util/settings.json"):
        self.settings_path = settings_path
        self.load_settings()
        self.info_panel = InfoPanel(self.settings)
        self.collect_image_paths()
        self.current_index = self.settings.get("last_viewed_index", 0)
        self.screen_height = self.settings.get("screen_height", 1080)

    def load_settings(self):
        if not os.path.exists(self.settings_path):
            self.settings = {
                "source_paths": [],
                "destination": "",
                "collected_counter": 0,
                "last_viewed_index": 0,
                "info_panel": {
                    "width": 600,
                    "height": 400,
                    "font_size": 14,
                    "background_color": [0, 0, 0],
                    "text_color": [255, 255, 255],
                    "visible": True,
                    "position": {"x": 800, "y": 0}
                }
            }
            self.save_settings()
        else:
            with open(self.settings_path, 'r') as f:
                self.settings = json.load(f)

    def save_settings(self):
        # Update info panel settings before saving
        self.settings["info_panel"] = self.info_panel.save_state()
        with open(self.settings_path, 'w') as f:
            json.dump(self.settings, f, indent=4)

    def collect_image_paths(self):
        """Collect all image paths from source directories"""
        self.image_paths = []
        # Use only lowercase extensions and normalize during checking
        SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.arw']
        
        for source_path in self.settings["source_paths"]:
            path = Path(source_path)
            if path.exists():
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith(tuple(SUPPORTED_EXTENSIONS)):
                            self.image_paths.append(Path(root) / file)
        
        self.image_paths = list(set(self.image_paths))  # Remove any possible duplicates
        self.image_paths.sort()
        logger.info(f"Found {len(self.image_paths)} images")

    def get_destination_path(self, image_path):
        """Get the destination path for an image"""
        return Path(self.settings["destination"]) / image_path.name

    def copy_to_destination(self, image_path):
        """Copy image to destination folder and update counter"""
        dest_path = self.get_destination_path(image_path)
        if not dest_path.exists():
            try:
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image_path, dest_path)
                self.settings["collected_counter"] += 1
                self.save_settings()
                return True
            except Exception as e:
                logger.error(f"Error copying file {image_path}: {e}")
                return False
        return False

    def remove_from_destination(self, image_path):
        """Remove image from destination folder and update counter"""
        dest_path = self.get_destination_path(image_path)
        if dest_path.exists():
            try:
                dest_path.unlink()
                self.settings["collected_counter"] -= 1
                self.save_settings()
                return True
            except Exception as e:
                logger.error(f"Error removing file {dest_path}: {e}")
                return False
        return False

    @lru_cache(maxsize=4)  # Reduced cache size to prevent memory issues
    def load_raw_image(self, image_path):
        """Load and process RAW image files with timeout"""
        try:
            with timeout(30):  # 30 second timeout for raw processing
                with rawpy.imread(str(image_path)) as raw:
                    image = raw.postprocess()
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    return image
        except TimeoutException:
            logger.error(f"Timeout processing RAW file: {image_path}")
            return None
        except rawpy.LibRawError as e:
            logger.error(f"Error reading RAW file {image_path}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing {image_path}: {e}")
            return None
        finally:
            gc.collect()  # Clean up memory after processing RAW file

    def is_raw_file(self, image_path):
        """Check if the file is a RAW format"""
        return str(image_path).lower().endswith('.arw')

    def load_image(self, image_path):
        """Load image with appropriate handling based on file type"""
        try:
            if self.is_raw_file(image_path):
                return self.load_raw_image(image_path)
            else:
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.error(f"Failed to load image: {image_path}")
                return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def run(self):
        """Main application loop"""
        if not self.image_paths:
            logger.error("No images found in source paths!")
            return

        cv2.namedWindow('Photo Annotator', cv2.WINDOW_NORMAL)

        while True:
            current_image_path = self.image_paths[self.current_index]
            
            # Load image with appropriate handler
            image = self.load_image(current_image_path)
            
            if image is None:
                self.info_panel.set_status("Error: Failed to load image")
                self.current_index = min(len(self.image_paths) - 1, self.current_index + 1)
                continue

            # Calculate and set main window size
            target_height = int(self.screen_height * 0.7)
            scale = target_height / image.shape[0]
            window_width = int(image.shape[1] * scale)
            window_height = target_height
            cv2.resizeWindow('Photo Annotator', window_width, window_height)

            # Prepare info text
            info_text = [
                f"Image {self.current_index + 1} of {len(self.image_paths)}",
                f"File: {current_image_path.name}",
                f"Collected: {self.settings['collected_counter']}",
                "",
                "Controls:",
                "q - quit",
                "a - previous image",
                "d - next image",
                "w - collect image",
                "e - remove from collection",
                "i - toggle info panel",
                "",
                "Mouse wheel - adjust text size"
            ]

            # Update info panel
            dest_path = self.get_destination_path(current_image_path)
            if dest_path.exists():
                self.info_panel.set_status("Status: Image is in collection")
            else:
                self.info_panel.set_status("Status: Image not collected")

            # Show info panel
            self.info_panel.render(info_text)

            # Show main image
            cv2.imshow('Photo Annotator', image)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('a'):
                self.current_index = max(0, self.current_index - 1)
            elif key == ord('d'):
                self.current_index = min(len(self.image_paths) - 1, self.current_index + 1)
            elif key == ord('w'):
                if self.copy_to_destination(current_image_path):
                    self.info_panel.set_status("Status: Image added to collection")
            elif key == ord('e'):
                if self.remove_from_destination(current_image_path):
                    self.info_panel.set_status("Status: Image removed from collection")
            elif key == ord('i'):
                self.info_panel.toggle()

            # Save last viewed index
            self.settings["last_viewed_index"] = self.current_index
            self.save_settings()

        # Cleanup
        cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        annotator = PhotoAnnotator()
        annotator.run()
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)