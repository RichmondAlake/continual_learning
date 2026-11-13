"""Render the weight-space notebooks' Mermaid diagrams and embed them as PNG attachments.

Each diagram's source is a file in beyond_the_context_window/images/diagrams/, named after the
notebook it belongs to: 02_reference_architecture.mmd belongs to 02_finetune_embeddings.ipynb. A
markdown cell shows it with ![alt](attachment:02_reference_architecture.png). VS Code does not draw
Mermaid, so the notebook carries the picture and the repository carries the source.

    python scripts/diagrams/render.py beyond_the_context_window/02_finetune_embeddings.ipynb
    python scripts/diagrams/render.py beyond_the_context_window/*.ipynb

Rendering uses mermaid-cli through npx (run `npx -y @mermaid-js/mermaid-cli --version` once), the
shared dark theme in theme.json and the neon glow in neon.css, on a black background. Images are
rendered at 2x and scaled to at most 960 px wide, so they stay readable in a notebook.
"""
import base64
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
MAX_WIDTH = 960
ATTACHMENT = re.compile(r"\(attachment:([\w-]+)\.png\)")


def mmdc() -> list[str]:
    found = next(Path.home().glob(".npm/_npx/*/node_modules/.bin/mmdc"), None)
    return [str(found)] if found else ["npx", "-y", "@mermaid-js/mermaid-cli"]


def render_png(source: Path) -> bytes:
    """One .mmd file to PNG bytes: 2x render, then scaled to at most MAX_WIDTH pixels wide."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "diagram.png"
        cmd = mmdc() + ["-i", str(source), "-o", str(out), "-c", str(HERE / "theme.json"),
                        "-C", str(HERE / "neon.css"), "-b", "#000000", "-s", "2", "-w", "900", "-q"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"{source.name}: {result.stderr[-800:]}")
        image = Image.open(out).convert("RGB")
    natural = image.width // 2                      # the width at 1x
    width = min(MAX_WIDTH, int(natural * 1.4))      # a little larger than 1x, never wider than MAX_WIDTH
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def embed(notebook: Path) -> None:
    """Re-render every diagram a notebook references and store it as a cell attachment."""
    sources = notebook.parent / "images" / "diagrams"
    nb = json.loads(notebook.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        names = ATTACHMENT.findall("".join(cell["source"]))
        for name in names:
            mmd = sources / f"{name}.mmd"
            if not mmd.exists():
                continue
            png = render_png(mmd)
            cell.setdefault("attachments", {})[f"{name}.png"] = {"image/png": base64.b64encode(png).decode()}
            print(f"{notebook.name}: {name}.png ({Image.open(io.BytesIO(png)).width} px wide)")
    notebook.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        embed(Path(arg).resolve())
