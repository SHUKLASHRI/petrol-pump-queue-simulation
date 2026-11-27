"""
Build script to package the Streamlit application for GitHub Pages using stlite (Pyodide Wasm).
Generates standalone index.html in docs/ and dist/ directories.
"""

import os
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Target files to bundle into the virtual filesystem
TARGET_DIRS = ["simulation", "visualization", "utils"]
SINGLE_FILES = ["app.py"]

files_dict = {}

for filename in SINGLE_FILES:
    file_path = BASE_DIR / filename
    if file_path.exists():
        files_dict[filename] = file_path.read_text(encoding="utf-8")

for dir_name in TARGET_DIRS:
    dir_path = BASE_DIR / dir_name
    if dir_path.exists():
        for py_file in dir_path.rglob("*.py"):
            rel_path = py_file.relative_to(BASE_DIR).as_posix()
            files_dict[rel_path] = py_file.read_text(encoding="utf-8")

# Package into HTML template
files_json_str = json.dumps(files_dict, indent=2)

HTML_TEMPLATE = f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta http-equiv="X-UA-Compatible" content="IE=edge" />
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no" />
    <title>PetroFlow • Petrol Pump Queue Simulation</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⛽</text></svg>" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.73.0/build/stlite.css" />
    <style>
      body, html {{
        margin: 0;
        padding: 0;
        height: 100%;
        width: 100%;
        background-color: #080c14;
        overflow: hidden;
      }}
      #root {{
        height: 100vh;
        width: 100vw;
      }}
      /* Loading screen */
      #loading {{
        position: fixed;
        inset: 0;
        background: #080c14;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        z-index: 9999;
        transition: opacity 0.5s ease;
      }}
      .spinner {{
        width: 48px;
        height: 48px;
        border: 4px solid rgba(16, 185, 129, 0.2);
        border-top-color: #10b981;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        margin-bottom: 20px;
      }}
      @keyframes spin {{
        to {{ transform: rotate(360deg); }}
      }}
    </style>
  </head>
  <body>
    <div id="loading">
      <div class="spinner"></div>
      <div style="font-size: 1.25rem; font-weight: 700; letter-spacing: -0.02em;">Loading PetroFlow Studio...</div>
      <div style="font-size: 0.85rem; color: #94a3b8; margin-top: 6px;">Initializing WebAssembly Python &amp; Discrete-Event Simulation Engine</div>
    </div>
    <div id="root"></div>
    <script src="https://cdn.jsdelivr.net/npm/@stlite/mountable@0.73.0/build/stlite.js"></script>
    <script>
      const files = {files_json_str};

      stlite.mount({{
        requirements: ["simpy", "numpy", "pandas", "plotly", "scipy"],
        entrypoint: "app.py",
        files: files,
        streamlitConfig: {{
          "theme.primaryColor": "#10b981",
          "theme.backgroundColor": "#080c14",
          "theme.secondaryBackgroundColor": "#0d121f",
          "theme.textColor": "#f1f5f9",
          "client.toolbarMode": "minimal"
        }}
      }}, document.getElementById("root")).then(() => {{
        const loadingEl = document.getElementById("loading");
        if (loadingEl) {{
          loadingEl.style.opacity = "0";
          setTimeout(() => loadingEl.remove(), 500);
        }}
      }});
    </script>
  </body>
</html>
"""

# Write to docs/ and dist/
for target_dir in [BASE_DIR / "docs", BASE_DIR / "dist"]:
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "index.html"
    out_file.write_text(HTML_TEMPLATE, encoding="utf-8")
    print(f"Generated: {out_file} ({out_file.stat().st_size:,} bytes)")

print(f"Bundled {len(files_dict)} source files successfully.")
