#!/usr/bin/env python3
"""Generate an ultra-crisp, high-contrast, perfectly readable 560x280 thumbnail for Kaggle.
Exact dimensions: 560 x 280 pixels.
Designed for high legibility even at 50% display scale on retina / mobile displays.
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H = 560, 280

# 1. Base Image - Dark sleek gradient background
img = Image.new("RGBA", (W, H), (15, 23, 42, 255)) # Slate 900
draw = ImageDraw.Draw(img)

# Subtle background gradient / paneling
for y in range(H):
    # Gradient from #0d1527 to #1e293b
    ratio = y / H
    r = int(11 + (22 - 11) * ratio)
    g = int(17 + (34 - 17) * ratio)
    b = int(32 + (55 - 32) * ratio)
    draw.line([(0, y), (W, y)], fill=(r, g, b, 255))

# Outer subtle border with cyan glow
draw.rounded_rectangle([10, 10, W - 10, H - 10], radius=16, outline=(56, 189, 248, 180), width=2)
# Inner card background
draw.rounded_rectangle([12, 12, W - 12, H - 12], radius=14, fill=(15, 23, 42, 220))

# Try loading high quality system fonts, fallback to default
def get_font(size, bold=False):
    font_paths = [
        # macOS system fonts
        "/System/Library/Fonts/SFProText-Bold.otf" if bold else "/System/Library/Fonts/SFProText-Regular.otf",
        "/System/Library/Fonts/SFProDisplay-Bold.otf" if bold else "/System/Library/Fonts/SFProDisplay-Regular.otf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Bold.ttf" if bold else "/Library/Fonts/Arial.ttf",
    ]
    for p in font_paths:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()

font_badge = get_font(12, bold=True)
font_title = get_font(26, bold=True)
font_title_sub = get_font(22, bold=True)
font_pill_label = get_font(11, bold=True)
font_pill_val = get_font(14, bold=True)
font_footer = get_font(12, bold=True)

# 1. Top Eyebrow Badge
# Draw badge container
draw.rounded_rectangle([32, 24, 230, 48], radius=6, fill=(2, 132, 199, 255)) # Cyan 600
draw.text((42, 28), "KAGGLE PLAYGROUND S6E10", fill=(255, 255, 255, 255), font=font_badge)

# Target tag right aligned
draw.rounded_rectangle([375, 24, 528, 48], radius=6, fill=(30, 41, 59, 255), outline=(71, 85, 105, 255), width=1)
draw.text((385, 28), "AIRLINE SATISFACTION", fill=(148, 163, 184, 255), font=font_badge)

# 2. Main High-Impact Title (High Contrast Crisp White & Sky Blue)
draw.text((32, 60), "Orthogonal Multi-Family Stacking", fill=(255, 255, 255, 255), font=font_title)
draw.text((32, 94), "& The Gold Summit", fill=(56, 189, 248, 255), font=font_title_sub) # #38bdf8

# 3. Big Metric Cards / Badges (Centered, Large, readable from far away)
# Card 1: Peak Gold Rank 11
draw.rounded_rectangle([32, 134, 185, 206], radius=8, fill=(30, 41, 59, 255), outline=(234, 179, 8, 255), width=2)
draw.text((44, 142), "PUBLIC LEADERBOARD", fill=(148, 163, 184, 255), font=font_pill_label)
draw.text((44, 162), "Rank 11 (Gold)", fill=(250, 204, 21, 255), font=font_pill_val) # Yellow gold
draw.text((44, 184), "0.96213 ROC AUC", fill=(226, 232, 240, 255), font=font_badge)

# Card 2: 5-Fold Stratified OOF CV
draw.rounded_rectangle([198, 134, 351, 206], radius=8, fill=(30, 41, 59, 255), outline=(56, 189, 248, 255), width=2)
draw.text((210, 142), "5-FOLD LOCAL CV", fill=(148, 163, 184, 255), font=font_pill_label)
draw.text((210, 162), "0.962240 OOF", fill=(56, 189, 248, 255), font=font_pill_val) # Cyan
draw.text((210, 184), "Champion Stack", fill=(226, 232, 240, 255), font=font_badge)

# Card 3: Final Private Result
draw.rounded_rectangle([364, 134, 528, 206], radius=8, fill=(30, 41, 59, 255), outline=(52, 211, 153, 255), width=2)
draw.text((376, 142), "FINAL PRIVATE LB", fill=(148, 163, 184, 255), font=font_pill_label)
draw.text((376, 162), "0.96238 Score", fill=(52, 211, 153, 255), font=font_pill_val) # Emerald
draw.text((376, 184), "Bronze Medal Tier", fill=(226, 232, 240, 255), font=font_badge)

# 4. Tech Highlights Bar / Footer
draw.rounded_rectangle([32, 222, 528, 256], radius=6, fill=(15, 23, 42, 255), outline=(51, 65, 85, 255), width=1)
draw.text((42, 231), "TabPFN-3.5  |  PyTabKit RealMLP  |  Survey Zero N/A  |  Logit Stacking", fill=(203, 213, 225, 255), font=font_footer)

# Save exact 560x280 PNG
out_path = Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/playground-series-s6e10/docs/images/thumbnail_560x280.png")
img.save(out_path, format="PNG", optimize=True)
print(f"Generated exact 560x280 thumbnail: {out_path.stat().st_size} bytes")
