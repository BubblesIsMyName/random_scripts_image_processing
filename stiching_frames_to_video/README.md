# Create videos from images

## Usage

```bash
python create_video.py root_dir
```

# USEFUL COMMANDS

To copy all the processed videos to a new folder afterwards


```bash

find . -type f -name "*.mp4" -print0 | xargs -0 -I {} cp {} /path/to/destination/folder

```


