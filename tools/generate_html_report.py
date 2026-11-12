#!/usr/bin/env python3
"""
tools/generate_html_report.py
Generates an all-in-one interactive SPA document for the Kaggle Solution Showcase.
Includes side-by-side or stacked deep-dive sections:
- HLD & Systems Architecture (with Mermaid + High-Res Diagrams)
- Full Solution Writeup (Gold-tier narrative, Math proofs, TL;DR)
- Ablation Study & Failure Modes (Waterfall table, Feature dilution, negative results)
"""

import argparse
import base64
import os
import re
from pathlib import Path
import markdown

REPO_ROOT = Path(__file__).resolve().parent.parent


def get_image_base64_map():
    images_dir = REPO_ROOT / "docs" / "images"
    b64_map = {}
    if images_dir.is_dir():
        for img_path in images_dir.glob("*.png"):
            try:
                data = img_path.read_bytes()
                b64 = base64.b64encode(data).decode("utf-8")
                b64_map[img_path.name] = f"data:image/png;base64,{b64}"
            except Exception as e:
                print(f"Warning: could not encode {img_path}: {e}")
    return b64_map


def clean_markdown_to_html(md_text: str, img_map: dict) -> str:
    # Replace relative markdown image paths with base64 data URIs
    for img_name, b64_uri in img_map.items():
        pattern = rf"!\[([^\]]*)\]\((?:\./)?(?:docs/)?images/{re.escape(img_name)}\)"
        md_text = re.sub(pattern, rf'![\1]({b64_uri})', md_text)
        # also match plain images/filename.png
        pattern2 = rf"!\[([^\]]*)\]\((?:images/)?{re.escape(img_name)}\)"
        md_text = re.sub(pattern2, rf'![\1]({b64_uri})', md_text)

    # Convert markdown to HTML with tables, fenced_code, etc.
    html = markdown.markdown(
        md_text,
        extensions=[
            "extra",
            "tables",
            "fenced_code",
            "codehilite",
            "toc",
            "sane_lists"
        ]
    )
    return html


def build_unified_html() -> str:
    img_map = get_image_base64_map()

    writeup_md = (REPO_ROOT / "WRITEUP.md").read_text(encoding="utf-8") if (REPO_ROOT / "WRITEUP.md").exists() else "# Writeup not found"
    arch_md = (REPO_ROOT / "docs" / "ARCHITECTURE.md").read_text(encoding="utf-8") if (REPO_ROOT / "docs" / "ARCHITECTURE.md").exists() else "# Architecture not found"
    ablation_md = (REPO_ROOT / "docs" / "ABLATION_STUDY.md").read_text(encoding="utf-8") if (REPO_ROOT / "docs" / "ABLATION_STUDY.md").exists() else "# Ablation study not found"

    writeup_html = clean_markdown_to_html(writeup_md, img_map)
    arch_html = clean_markdown_to_html(arch_md, img_map)
    ablation_html = clean_markdown_to_html(ablation_md, img_map)

    hld_img_b64 = img_map.get("figure6_harness_architecture.png", img_map.get("figure1_models_and_weights.png", ""))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kaggle S6E10 Champion Solution — Unified Technical Showcase</title>
  
  <!-- MathJax for KaTeX/LaTeX equations -->
  <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
  <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
  
  <!-- Mermaid for Architecture Diagrams -->
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>

  <style>
    :root {{
      --bg-primary: #f8fafc;
      --bg-surface: #ffffff;
      --text-main: #0f172a;
      --text-sub: #334155;
      --text-muted: #64748b;
      --accent: #0284c7;
      --accent-hover: #0369a1;
      --accent-soft: #f0f9ff;
      --accent-border: #bae6fd;
      --gold: #b45309;
      --gold-bg: #fef3c7;
      --gold-border: #fcd34d;
      --green: #047857;
      --green-bg: #d1fae5;
      --red: #b91c1c;
      --red-bg: #fee2e2;
      --border: #e2e8f0;
      --border-dark: #cbd5e1;
      --shadow-sm: 0 1px 3px 0 rgb(0 0 0 / 0.06), 0 1px 2px -1px rgb(0 0 0 / 0.04);
      --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.08), 0 2px 4px -2px rgb(0 0 0 / 0.04);
      --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.08), 0 4px 6px -4px rgb(0 0 0 / 0.04);
    }}

    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    
    html {{
      scroll-behavior: smooth;
    }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg-primary);
      color: var(--text-main);
      line-height: 1.7;
    }}

    /* Top Sticky Navigation Bar */
    .top-nav {{
      position: sticky;
      top: 0;
      z-index: 1000;
      background: rgba(255, 255, 255, 0.95);
      backdrop-filter: blur(8px);
      border-bottom: 1px solid var(--border);
      padding: 0.75rem 1.5rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
      box-shadow: var(--shadow-sm);
    }}
    .nav-brand {{
      font-weight: 800;
      font-size: 1.1rem;
      color: var(--accent);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      text-decoration: none;
    }}
    .nav-links {{
      display: flex;
      gap: 0.5rem;
      list-style: none;
      flex-wrap: wrap;
    }}
    .nav-link {{
      text-decoration: none;
      color: var(--text-sub);
      font-size: 0.88rem;
      font-weight: 600;
      padding: 0.4rem 0.85rem;
      border-radius: 9999px;
      transition: all 0.2s ease;
      border: 1px solid transparent;
    }}
    .nav-link:hover, .nav-link.active {{
      background: var(--accent-soft);
      color: var(--accent);
      border-color: var(--accent-border);
    }}

    /* Main Container & Layout */
    .layout {{
      max-width: 1200px;
      margin: 2rem auto;
      padding: 0 1.5rem;
    }}

    /* Hero Banner */
    .hero {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 2.5rem 2rem;
      text-align: center;
      margin-bottom: 2rem;
      box-shadow: var(--shadow-md);
    }}
    .badge-gold {{
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      background: var(--gold-bg);
      color: var(--gold);
      border: 1px solid var(--gold-border);
      font-weight: 700;
      font-size: 0.85rem;
      padding: 0.3rem 0.85rem;
      border-radius: 9999px;
      margin-bottom: 0.75rem;
    }}
    .hero h1 {{
      font-size: 2.25rem;
      font-weight: 800;
      letter-spacing: -0.025em;
      margin-bottom: 0.5rem;
      color: var(--text-main);
    }}
    .hero p.lead {{
      color: var(--text-sub);
      font-size: 1.1rem;
      max-width: 760px;
      margin: 0 auto 1.5rem;
    }}
    .meta-pills {{
      display: flex;
      justify-content: center;
      gap: 1rem;
      flex-wrap: wrap;
    }}
    .pill {{
      background: var(--bg-primary);
      border: 1px solid var(--border);
      padding: 0.35rem 0.85rem;
      border-radius: 0.5rem;
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--text-sub);
    }}

    /* High-Level Architecture (HLD) Visual Card */
    .hld-card {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 2rem;
      margin-bottom: 2.5rem;
      box-shadow: var(--shadow-md);
    }}
    .section-title {{
      font-size: 1.5rem;
      font-weight: 800;
      margin-bottom: 1.25rem;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.6rem;
      padding-bottom: 0.5rem;
      border-bottom: 2px solid var(--accent-soft);
    }}
    .section-title::before {{
      content: "";
      width: 5px;
      height: 1.3rem;
      background: var(--accent);
      border-radius: 3px;
    }}
    .hld-visual-wrap {{
      text-align: center;
      background: #fafafa;
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.5rem;
      margin: 1.5rem 0;
    }}
    .hld-visual-wrap img {{
      max-width: 100%;
      height: auto;
      border-radius: 0.5rem;
      box-shadow: var(--shadow-sm);
    }}

    /* Tabs & Content Switcher */
    .tab-nav-container {{
      position: sticky;
      top: 60px;
      z-index: 900;
      background: rgba(248, 250, 252, 0.98);
      backdrop-filter: blur(8px);
      padding: 1rem 0;
      margin-bottom: 1.5rem;
      border-bottom: 1px solid var(--border);
    }}
    .tab-bar {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1rem;
    }}
    .tab-btn {{
      background: var(--bg-surface);
      border: 2px solid var(--border);
      border-radius: 0.75rem;
      padding: 1.1rem 1.25rem;
      cursor: pointer;
      text-align: left;
      transition: all 0.2s ease;
      box-shadow: var(--shadow-sm);
    }}
    .tab-btn:hover {{
      border-color: var(--accent-border);
      transform: translateY(-2px);
      box-shadow: var(--shadow-md);
    }}
    .tab-btn.active {{
      border-color: var(--accent);
      background: var(--accent-soft);
      box-shadow: var(--shadow-md);
    }}
    .tab-btn-title {{
      font-weight: 800;
      font-size: 1rem;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 0.5rem;
      margin-bottom: 0.25rem;
    }}
    .tab-btn.active .tab-btn-title {{
      color: var(--accent);
    }}
    .tab-btn-desc {{
      font-size: 0.82rem;
      color: var(--text-muted);
      line-height: 1.4;
    }}

    /* Pane styling */
    .tab-pane {{
      display: none;
      animation: fadeIn 0.3s ease-in-out;
    }}
    .tab-pane.active {{
      display: block;
    }}
    @keyframes fadeIn {{
      from {{ opacity: 0; transform: translateY(6px); }}
      to {{ opacity: 1; transform: translateY(0); }}
    }}

    /* Document Container inside Panes */
    .doc-content {{
      background: var(--bg-surface);
      border: 1px solid var(--border);
      border-radius: 1rem;
      padding: 2.5rem;
      box-shadow: var(--shadow-sm);
      margin-bottom: 3rem;
    }}

    /* Typography & Markdown styling inside docs */
    .doc-content h1 {{ font-size: 1.85rem; font-weight: 800; margin: 1.5rem 0 1rem; color: var(--text-main); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    .doc-content h2 {{ font-size: 1.45rem; font-weight: 700; margin: 1.75rem 0 0.85rem; color: var(--text-main); }}
    .doc-content h3 {{ font-size: 1.2rem; font-weight: 700; margin: 1.5rem 0 0.75rem; color: var(--text-sub); }}
    .doc-content p {{ margin-bottom: 1.15rem; color: var(--text-sub); }}
    .doc-content ul, .doc-content ol {{ margin-left: 1.75rem; margin-bottom: 1.25rem; color: var(--text-sub); }}
    .doc-content li {{ margin-bottom: 0.4rem; }}
    .doc-content blockquote {{
      border-left: 4px solid var(--accent);
      background: var(--accent-soft);
      padding: 0.85rem 1.25rem;
      border-radius: 0 0.5rem 0.5rem 0;
      margin-bottom: 1.5rem;
      color: var(--text-sub);
    }}
    .doc-content table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      margin: 1.5rem 0;
      border: 1px solid var(--border);
      border-radius: 0.5rem;
      overflow: hidden;
      font-size: 0.92rem;
    }}
    .doc-content th, .doc-content td {{
      padding: 0.85rem 1rem;
      text-align: left;
    }}
    .doc-content th {{
      background: #f1f5f9;
      color: var(--text-muted);
      font-weight: 700;
      text-transform: uppercase;
      font-size: 0.8rem;
      letter-spacing: 0.05em;
      border-bottom: 1px solid var(--border-dark);
    }}
    .doc-content td {{
      border-bottom: 1px solid var(--border);
      color: var(--text-sub);
    }}
    .doc-content tr:last-child td {{ border-bottom: none; }}
    .doc-content tr:nth-child(even) {{ background: var(--bg-primary); }}
    .doc-content tr:hover {{ background: #f1f5f9; }}
    .doc-content img {{
      max-width: 100%;
      height: auto;
      border-radius: 0.5rem;
      border: 1px solid var(--border);
      margin: 1.5rem auto;
      display: block;
      box-shadow: var(--shadow-sm);
    }}
    .doc-content pre {{
      background: #0f172a;
      color: #f8fafc;
      padding: 1.25rem;
      border-radius: 0.5rem;
      overflow-x: auto;
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.88rem;
      line-height: 1.6;
      margin-bottom: 1.5rem;
    }}
    .doc-content code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
      font-size: 0.88em;
      background: #f1f5f9;
      color: #0369a1;
      padding: 0.2rem 0.4rem;
      border-radius: 0.25rem;
      border: 1px solid var(--border-dark);
    }}
    .doc-content pre code {{
      background: transparent;
      color: inherit;
      padding: 0;
      border: none;
    }}

    footer {{
      text-align: center;
      padding: 3rem 1rem 2rem;
      color: var(--text-muted);
      font-size: 0.9rem;
      border-top: 1px solid var(--border);
      margin-top: 3rem;
    }}
  </style>
</head>
<body>

  <!-- Top Sticky Navigation Bar -->
  <header class="top-nav">
    <a href="#" class="nav-brand">
      🏆 Kaggle S6E10 Champion Solution
    </a>
    <nav>
      <ul class="nav-links">
        <li><a href="#hld-section" class="nav-link">🏛️ High-Level Design (HLD)</a></li>
        <li><a href="javascript:void(0)" onclick="switchTab('tab-writeup')" class="nav-link">📖 Full Writeup</a></li>
        <li><a href="javascript:void(0)" onclick="switchTab('tab-arch')" class="nav-link">⚙️ Systems Architecture</a></li>
        <li><a href="javascript:void(0)" onclick="switchTab('tab-ablation')" class="nav-link">🔬 Ablation & Failures</a></li>
        <li><a href="https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction" target="_blank" class="nav-link" style="color: var(--accent);">📦 GitHub Repo ↗</a></li>
      </ul>
    </nav>
  </header>

  <div class="layout">
    
    <!-- Hero Banner -->
    <section class="hero">
      <div class="badge-gold">🏆 Top 1% Gold Medal Tier Solution</div>
      <h1>Airline Passenger Satisfaction Champion Solution</h1>
      <p class="lead">
        Unified, leak-free competitive ML engineering showcase. Discover the domain insights, mathematical stacking formulation, and modular architecture that shattered the GBDT ceiling.
      </p>
      <div class="meta-pills">
        <span class="pill">Competition: <strong>Playground Series S6E10</strong></span>
        <span class="pill">Final Metric: <strong>0.96342 ROC-AUC</strong></span>
        <span class="pill">Peak Rank: <strong>Rank 11 / 2,400+ Teams</strong></span>
        <span class="pill">Author: <strong>Vijai KM — Senior Machine Learning Engineer</strong></span>
        <span class="pill">License: <strong>MIT</strong></span>
      </div>
    </section>

    <!-- High-Level Architecture (HLD) Section -->
    <section id="hld-section" class="hld-card">
      <h2 class="section-title">High-Level Design (HLD) & Systems Topology</h2>
      <p style="color: var(--text-sub); margin-bottom: 1.25rem;">
        An end-to-end multi-tier pipeline designed for strictly leak-free cross-validation, transductive domain prior transfer, orthogonal model family diversity, and unconstrained logit-odds meta-stacking.
      </p>

      <!-- Mermaid Flowchart HLD -->
      <div style="background: #ffffff; padding: 1.5rem; border-radius: 0.75rem; border: 1px solid var(--border); margin-bottom: 1.5rem;">
        <pre class="mermaid">
flowchart LR
    subgraph Data_Layer["1. Data Ingestion &amp; Contracts"]
        Raw["Raw train.csv &amp; test.csv"] --> Polars["Polars Sub-second Columnar Ingestion"]
        Polars --> Folds["Deterministic 5-Fold Stratified Split (seed=42)"]
    end

    subgraph Feature_Layer["2. Domain &amp; Semantic Innovations"]
        Folds --> Zero["Structural Zero Flag: 0 = Structural N/A"]
        Folds --> Prior["External Super Teacher Priors (Original 129k rows)"]
        Folds --> Cross["3-Way Profile Interactions &amp; Corridor Clusters"]
        Zero --> FeatureMatrix["Gold 84-Feature Representation Matrix"]
        Prior --> FeatureMatrix
        Cross --> FeatureMatrix
    end

    subgraph Model_Layer["3. Multi-Family Orthogonal Models"]
        FeatureMatrix --> LGB["LightGBM (Hist Best-First)"]
        FeatureMatrix --> CAT["CatBoost (Ordered Categoricals)"]
        FeatureMatrix --> XGB["XGBoost (Exact Depth)"]
        FeatureMatrix --> RES["Tabular ResMLP (Quantile Normals)"]
    end

    subgraph Stacking_Layer["4. Meta-Learner &amp; Consensus"]
        LGB --> OOF["5-Fold Out-of-Fold Matrix"]
        CAT --> OOF
        XGB --> OOF
        RES --> OOF
        OOF --> LogitMap["Logit-Odds Transformation: z = ln(p / 1-p)"]
        LogitMap --> Ridge["L2 Ridge Meta-Learner (C=0.001)"]
        Ridge --> SLSQP["SLSQP Percentile Rank Optimization"]
        SLSQP --> Submission["Verified submission.csv"]
    end
        </pre>
      </div>

      <!-- High-Resolution Architectural Graphic if available -->
      {f'''
      <div class="hld-visual-wrap">
        <h3 style="font-size: 1.05rem; font-weight: 700; color: var(--text-sub); margin-bottom: 0.75rem;">Figure: Autonomous Systems Architecture &amp; Epistemic Verification Gates</h3>
        <img src="{hld_img_b64}" alt="High-Level Architecture and Autonomous Harness Architecture" />
      </div>
      ''' if hld_img_b64 else ''}
    </section>

    <!-- Deep-Dive Interactive Document Switcher -->
    <div class="tab-nav-container">
      <h3 style="font-size: 1.15rem; font-weight: 800; color: var(--text-main); margin-bottom: 0.75rem;">
        📚 Complete Technical Deep-Dive Documentation (All in One Page)
      </h3>
      <div class="tab-bar">
        <button id="btn-writeup" class="tab-btn active" onclick="switchTab('tab-writeup')">
          <div class="tab-btn-title">📖 Full Gold-Tier Writeup</div>
          <div class="tab-btn-desc">Mathematical proofs, GBDT saturation barrier, structural zero semantics, and dose-response curve geometry.</div>
        </button>
        <button id="btn-arch" class="tab-btn" onclick="switchTab('tab-arch')">
          <div class="tab-btn-title">🏛️ Systems Architecture (LLD/HLD)</div>
          <div class="tab-btn-desc">Polars schema normalization, downcasted data contracts, Level-1 diversity, and Level-2 logit mechanics.</div>
        </button>
        <button id="btn-ablation" class="tab-btn" onclick="switchTab('tab-ablation')">
          <div class="tab-btn-title">🔬 Ablation &amp; Failure Modes</div>
          <div class="tab-btn-desc">Feature ablation waterfall, proof of the Feature Dilution Trap, sensitivity sweeps, and negative results.</div>
        </button>
      </div>
    </div>

    <!-- TAB 1: FULL WRITEUP -->
    <div id="tab-writeup" class="tab-pane active">
      <article class="doc-content">
        {writeup_html}
      </article>
    </div>

    <!-- TAB 2: SYSTEMS ARCHITECTURE -->
    <div id="tab-arch" class="tab-pane">
      <article class="doc-content">
        {arch_html}
      </article>
    </div>

    <!-- TAB 3: ABLATION STUDY & NEGATIVE RESULTS -->
    <div id="tab-ablation" class="tab-pane">
      <article class="doc-content">
        {ablation_html}
      </article>
    </div>

    <footer>
      <p>
        <strong>Kaggle S6E10 Champion Portfolio Showcase</strong> | Crafted by <strong>Vijai KM</strong> (Senior Machine Learning Engineer)<br>
        Source Code &amp; Verified Pipelines on <a href="https://github.com/VijaiKMathew/kaggle-s6e10-airline-satisfaction" target="_blank" style="color: var(--accent); font-weight: 700;">GitHub</a> | Released under the <strong>MIT License</strong>
      </p>
    </footer>

  </div>

  <script>
    function switchTab(tabId) {{
      // Hide all panes
      document.querySelectorAll('.tab-pane').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

      // Show selected pane
      const targetPane = document.getElementById(tabId);
      if (targetPane) {{
        targetPane.classList.add('active');
      }}

      // Activate button
      if (tabId === 'tab-writeup') document.getElementById('btn-writeup').classList.add('active');
      if (tabId === 'tab-arch') document.getElementById('btn-arch').classList.add('active');
      if (tabId === 'tab-ablation') document.getElementById('btn-ablation').classList.add('active');

      // Scroll smoothly to content area
      targetPane.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }}
  </script>
</body>
</html>
"""
    return html


def main():
    parser = argparse.ArgumentParser(description="Generate unified interactive HTML solution showcase")
    parser.add_argument("--output", default="reports/solution_preview.html", help="Path to output HTML file")
    args = parser.parse_args()

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    html_content = build_unified_html()
    out_path.write_text(html_content, encoding="utf-8")
    print(f"✅ Unified interactive HTML report successfully created at: {out_path} ({len(html_content):,} bytes)")


if __name__ == "__main__":
    main()
