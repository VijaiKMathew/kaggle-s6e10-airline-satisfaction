#!/usr/bin/env python3
"""Update KAGGLE_PUBLISHING_AND_PORTFOLIO_GUIDE.md with all recent hard-won discoveries and production standards:
1. Kaggle Writeup Form Constraints:
   - Title: max 80 chars.
   - Subtitle: max 140 chars.
   - Writeup URL Slug: 3 to 50 chars max (auto-generated slugs will break if title > 50 chars).
   - Card / Thumbnail: 560 x 280 px PNG.
   - Media Gallery: 16:9 aspect ratio, minimum 640 x 360 px (recommended 1920 x 1080 px).
2. Image & Formatting Rules:
   - Strict prohibition of raw ```mermaid``` in Kaggle discussions (parser prints raw code).
   - Strict prohibition of wide ASCII boxes (+---+) due to mobile word wrapping.
   - Requirement to strip all EXIF/camera metadata from PNGs before upload.
   - Clean Markdown tables (no backticks or <br> inside cells).
3. Zero PII & Hardware sanitization policy (no Apple Silicon/CPU models).
4. Code Publication: Packaging and pushing companion Kaggle kernels via Kaggle CLI.
"""

from pathlib import Path
import re

guide_path = Path("/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/playground-series-s6e10/KAGGLE_PUBLISHING_AND_PORTFOLIO_GUIDE.md")
content = guide_path.read_text(encoding="utf-8")

# Enhance Section 5 (The Kaggle Writeup Formula) with exact Kaggle UI specs
section5_old = """## 5. The Kaggle Writeup Formula (Maximizing Discussion Upvotes)

The Kaggle discussion forum is where your solution reaches thousands of data scientists. A Gold-tier solution writeup establishes technical authority, drives traffic to your GitHub repo, and earns Kaggle Discussion Medals.

### The 7-Element High-Upvote Anatomy

1. **The Hook Title:**  
   Format: `[Rank X / Medal] - Approach Name: Core Innovation + Score`  
   *Example:* `[Top 1% Gold] S6E10 Airline Satisfaction: Unbounded Logit Stacking & Structural Zeros (AUC: 0.9634)`
2. **The 30-Second TL;DR:**  
   Bullet points at the very top summarizing metric score, model families, and the single biggest factor in your performance.
3. **Visual Architecture Diagram:**  
   Include an image or Mermaid chart. Posts with visual diagrams receive **3x to 5x more upvotes** than plain text.
4. **Validation Strategy & CV-LB Alignment:**  
   Prove your validation strategy held up. Show that local CV improvements translated monotonically to the Private Leaderboard.
5. **Feature Engineering & Ablation Table:**  
   Show what features were added, what was discarded, and the exact CV impact of each.
6. **"What Didn't Work" (The Integrity Section):**  
   The hallmark of an authentic senior engineer is sharing failed experiments. Kaggle Grandmasters actively praise writeups with strong negative results.
7. **Actionable Links:**  
   Provide direct links to:
   - Your public GitHub repository.
   - Your Kaggle inference verification notebook.
   - Your LinkedIn profile."""

section5_new = """## 5. The Kaggle Writeup Formula (Maximizing Discussion Upvotes & Form Compliance)

The Kaggle discussion forum and official **Solution Writeup** tab are where your solution reaches thousands of data scientists. A Gold-tier solution writeup establishes technical authority, drives traffic to your GitHub repo, and earns Kaggle Discussion Medals.

### The 7-Element High-Upvote Anatomy

1. **The Hook Title:**  
   Format: `[Rank X / Medal] - Approach Name: Core Innovation + Score`  
   *Example:* `Orthogonal Multi-Family Stacking & The Gold Summit (Rank 11 / 0.96213)`
2. **The 30-Second TL;DR:**  
   Bullet points at the very top summarizing metric score, model families, and the single biggest factor in your performance.
3. **Visual Architecture Diagram (Rendered Image Only):**  
   Always embed a clean, rendered 16:9 diagram (`docs/images/kaggle_media_gallery/figure0_system_hld.png`). Posts with visual diagrams receive **3x to 5x more upvotes** than plain text.
4. **Validation Strategy & CV-LB Alignment:**  
   Prove your validation strategy held up. Show that local CV improvements translated monotonically to the Private Leaderboard.
5. **Feature Engineering & Ablation Table:**  
   Show what features were added, what was discarded, and the exact CV impact of each.
6. **"What Didn't Work" (The Integrity Section):**  
   The hallmark of an authentic senior engineer is sharing failed experiments. Kaggle Grandmasters actively praise writeups with strong negative results.
7. **Actionable Links:**  
   Provide direct links to:
   - Your public GitHub repository.
   - Your public verified Kaggle inference kernel.
   - Your dataset sources.

---

### Strict Kaggle Solution Writeup Form Specifications

When using Kaggle's official **"+ New Solution Writeup"** form, adhere strictly to these character and dimension constraints:

| Field | Requirement | Common Failure Mode & Fix |
| :--- | :--- | :--- |
| **Title \*** | **$\le$ 80 characters** | Exceeding 80 chars blocks submission. Keep it punchy and outcome-driven. |
| **Subtitle \*** | **$\le$ 140 characters** | Must explain project in one high-signal sentence under 140 chars. |
| **Writeup URL Slug \*** | **3 to 50 characters** | **CRITICAL:** Kaggle auto-populates the slug from your title. If your title is > 50 chars, the form displays: *"Writeup URL must be between 3 and 50 characters long"*. You **must click "Edit"** and shorten the slug manually (e.g. `s6e10-champion-orthogonal-stacking`). |
| **Card & Thumbnail Image** | **Exact 560 × 280 px** | Kaggle strictly crops or scales. Use large bold fonts (24–28pt) and high-contrast metric pills so text remains legible at 50% scale on mobile. |
| **Media Gallery** | **16:9 Aspect Ratio ($\ge$ 640 × 360 px)** | Wide panoramic charts get chopped on the edges by Kaggle's 16:9 crop box. Standardize all publication charts on a **1920 × 1080 (16:9)** letterboxed canvas with safe margins. |
| **Tags** | **2 to 5 tags** | Select 2–5 relevant tags (e.g. `Tabular`, `Ensemble`, `Binary Classification`, `Feature Engineering`). |
| **Content \*** | **Clean Markdown Parser** | • **Never use raw ````mermaid````:** Kaggle does not render Mermaid in discussions; it displays raw code syntax. Embed rendered PNGs instead.<br>• **Avoid wide ASCII box art (`+----+`):** Responsive mobile screens wrap characters mid-line, destroying ASCII alignment. Use clean markdown subheaders (`### Stage X: ...`) and arrow flows (`[Stage 1] -> [Stage 2] ...`).<br>• **Clean Table Cells:** Prohibit backticks and `<br>` tags inside markdown table cells; they break Kaggle's table parser. |
| **EXIF & Privacy Warning** | **Strip all image metadata** | Kaggle displays: *"Make sure any EXIF data is stripped from your image to avoid sharing your location"*. Programmatically re-save PNGs as clean pixel arrays without camera or software tags before uploading. |"""

content = content.replace(section5_old, section5_new)

# Sanitize any remaining Apple Silicon or hardware references in the guide
content = re.sub(r'Apple Silicon M-series 10-core CPU', 'Modern Multi-Core Workstation CPU', content, flags=re.I)
content = re.sub(r'Apple Silicon', 'Multi-Core CPU', content, flags=re.I)
content = re.sub(r'M-series', 'Multi-Core', content, flags=re.I)

guide_path.write_text(content, encoding="utf-8")
print(f"Successfully updated {guide_path}")
