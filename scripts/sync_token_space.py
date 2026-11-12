"""Copy the token-space notebooks from a MemoRizz checkout into self_improving_agents/.

    python scripts/sync_token_space.py ~/Desktop/memorizz

The five notebooks are written in the MemoRizz repository, in examples/continual_learning.
This copies them with their images and calderwood_world.py, then makes two changes so they
run outside that repository:

1. The setup cell looked for MemoRizz's source tree above the notebook. Here MemoRizz is the
   installed package, or a source checkout named by the MEMORIZZ_SRC environment variable.
2. Links to files elsewhere in the MemoRizz repository point at its GitHub pages instead.

The store the notebooks write (calderwood_store/) is not copied.
"""
import json
import shutil
import sys
from pathlib import Path

NOTEBOOKS = [
    "01_workflows_to_skills.ipynb",
    "02_persona_evolution.ipynb",
    "03_entity_evolution.ipynb",
    "04_toolbox_evolution.ipynb",
    "05_refine_cycle.ipynb",
]
MEMORIZZ_GITHUB = "https://github.com/RichmondAlake/memorizz/blob/main"
LINKS = {
    "](../../docs/guides/continual-learning.md)": f"]({MEMORIZZ_GITHUB}/docs/guides/continual-learning.md)",
    "](continual_learning_guide.ipynb)": f"]({MEMORIZZ_GITHUB}/examples/continual_learning/continual_learning_guide.ipynb)",
}
SOURCE_TREE = """# 1. MemoRizz from this repository's source tree (pip install after the release).
REPO = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "memorizz").is_dir()),
    None,
)
if REPO and str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
"""
INSTALLED_OR_SRC = """# 1. MemoRizz: the installed package, or a source checkout named by MEMORIZZ_SRC
#    (for example MEMORIZZ_SRC=~/memorizz/src while a feature is not yet released).
SRC = os.path.expanduser(os.getenv("MEMORIZZ_SRC", ""))
if SRC and SRC not in sys.path:
    sys.path.insert(0, SRC)
"""


def patch(notebook: dict, name: str) -> dict:
    setup_cells = 0
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and SOURCE_TREE in source:
            source = source.replace(SOURCE_TREE, INSTALLED_OR_SRC)
            setup_cells += 1
        if cell["cell_type"] == "markdown":
            for old, new in LINKS.items():
                source = source.replace(old, new)
        cell["source"] = source.splitlines(keepends=True)
    if setup_cells != 1:
        sys.exit(f"{name}: expected one setup cell that loads MemoRizz from source, found {setup_cells}")
    return notebook


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    source = Path(sys.argv[1]).expanduser() / "examples" / "continual_learning"
    target = Path(__file__).resolve().parents[1] / "self_improving_agents"
    target.mkdir(exist_ok=True)
    for name in NOTEBOOKS:
        notebook = patch(json.loads((source / name).read_text()), name)
        (target / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    shutil.copy2(source / "calderwood_world.py", target / "calderwood_world.py")
    shutil.copytree(source / "images", target / "images", dirs_exist_ok=True)
    print(f"copied {len(NOTEBOOKS)} notebooks, calderwood_world.py and images/ into {target}")


if __name__ == "__main__":
    main()
