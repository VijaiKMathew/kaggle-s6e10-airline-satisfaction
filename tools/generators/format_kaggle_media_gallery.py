#!/usr/bin/env python3
"""Format and letterbox all figures for Kaggle Writeup Media Gallery and Thumbnail.
Kaggle Media Gallery requirements:
1. Minimum resolution: 640 x 360 (16:9).
2. Recommended native 16:9 canvas: 1920 x 1080 (Full HD, 300 DPI equivalent).
3. Aspect ratio 16:9 prevents the Kaggle crop preview tool from chopping titles, axes, or legends.
4. Thumbnail: Also generate a 16:9 1280x720 / 640x360 as well as 560x280.
"""

from pathlib import Path
from PIL import Image

OUT_DIR = Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/playground-series-s6e10/docs/images/kaggle_media_gallery")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_IMAGES = [
    Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/archive/playground-series-s6e10/docs/images/figure1_models_and_weights.png"),
    Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/archive/playground-series-s6e10/docs/images/figure2_feature_ablation.png"),
    Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/archive/playground-series-s6e10/docs/images/figure3_leaderboard_progression.png"),
    Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/archive/playground-series-s6e10/docs/images/figure4_regularization_sweep.png"),
    Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/archive/playground-series-s6e10/docs/images/figure5_feature_importance.png"),
    Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/archive/playground-series-s6e10/docs/images/cv_vs_lb_progression.png"),
    Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/playground-series-s6e10/docs/images/figure0_system_hld.png"),
]

TARGET_W, TARGET_H = 1920, 1080  # Standard 16:9 Full HD canvas
BG_COLOR = (255, 255, 255, 255)  # Clean white letterbox background for white scientific charts
DARK_BG_COLOR = (15, 23, 42, 255) # Slate dark for HLD / thumbnail

for src in SOURCE_IMAGES:
    if not src.exists():
        print(f"Skipping missing: {src}")
        continue
    
    with Image.open(src) as img:
        img = img.convert("RGBA")
        w, h = img.size
        
        # Determine background color: check corner pixels
        corner_pixel = img.getpixel((10, 10))
        # If dark theme (e.g. figure0_system_hld) use dark canvas, else pure white canvas
        is_dark = (corner_pixel[0] + corner_pixel[1] + corner_pixel[2]) / 3 < 120
        bg_fill = (15, 23, 42, 255) if is_dark else (255, 255, 255, 255)
        
        # Calculate scaling to fit within 1840 x 1000 with a safe margin
        max_w, max_h = 1840, 1020
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create 16:9 canvas
        canvas = Image.new("RGBA", (TARGET_W, TARGET_H), bg_fill)
        offset_x = (TARGET_W - new_w) // 2
        offset_y = (TARGET_H - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y), resized)
        
        dest = OUT_DIR / src.name
        canvas.save(dest, format="PNG", optimize=True)
        print(f"Created 16:9 canvas (1920x1080): {dest.name}")

# Also create a 1280x720 (16:9) and 640x360 version of the thumbnail so it fits both upload boxes
thumb_src = Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/playground-series-s6e10/docs/images/thumbnail_560x280.png")
if thumb_src.exists():
    with Image.open(thumb_src) as img:
        img = img.convert("RGBA")
        w, h = img.size
        
        # Target 1280x720 (16:9)
        canvas169 = Image.new("RGBA", (1280, 720), (15, 23, 42, 255))
        # Scale thumbnail to fill nicely with margin
        scale = min(1200 / w, 660 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas169.paste(resized, ((1280 - new_w) // 2, (720 - new_h) // 2), resized)
        
        dest_169 = OUT_DIR / "thumbnail_16x9_1280x720.png"
        canvas169.save(dest_169, format="PNG", optimize=True)
        print(f"Created 16:9 thumbnail: {dest_169.name}")
