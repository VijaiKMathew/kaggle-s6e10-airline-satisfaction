import shutil
from pathlib import Path

template_path = Path('/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/playground-series-s6e10/KAGGLE_SOLUTION_WRITEUP_MASTER_TEMPLATE.md')
text = template_path.read_text(encoding='utf-8')

rules = """## 🎨 Kaggle Forum Formatting & Visual Presentation Rules
1. **Never use raw Mermaid blocks in Kaggle Writeups:** Kaggle's forum parser does not render client-side Mermaid; it prints raw code blocks. Always generate and embed a clean rendered 16:9 PNG diagram (`docs/images/figure0_system_hld.png`).
2. **Avoid wide ASCII box art (+---+):** Mobile and responsive Kaggle forum views wrap characters across lines, breaking box diagrams. Use structured Markdown subheaders (`### Stage X: ...`) and high-level pipeline flow spans (`[Stage 1] -> [Stage 2] ...`), supported by an embedded figure.
3. **Table Formatting:** Keep table cells clean plain text. Do not use backticks or HTML break tags inside cells to prevent parser table collapse.
4. **16:9 Image Aspect Ratio:** All publication figures and media gallery images should adhere to 16:9 (e.g. 1920x1080) with safe margins so Kaggle's image crop modal does not chop labels or axes.
"""

if '## System High-Level Design (HLD)' in text and 'Kaggle Forum Formatting' not in text:
    text = text.replace('## System High-Level Design (HLD)', rules + '\n---\n\n## System High-Level Design (HLD)')

template_path.write_text(text, encoding='utf-8')

# Distribute to s6e10 and public repo
for dest in [
    Path('/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/playground-series-s6e10/KAGGLE_SOLUTION_WRITEUP_MASTER_TEMPLATE.md'),
    Path('/Users/vijaikm/Documents/Other-AI-stuff/3-kaggle/kaggle-s6e10-airline-satisfaction/KAGGLE_SOLUTION_WRITEUP_MASTER_TEMPLATE.md'),
]:
    shutil.copy(template_path, dest)
    print(f'Updated template at: {dest}')

print('Template synchronization complete!')
