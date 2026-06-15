"""Generate static PNG figures of the HfO2 crystal-structure polymorphs.

Renders the four HfO2 phases (monoclinic / tetragonal / polar orthorhombic / cubic)
individually plus a 2x2 comparison grid, using the Materials Project (live fetch,
cached to CIF under data/structures/) and ASE + matplotlib. Pure script, no notebook
(see scripts/render_structures.sh).

Usage:
    export MP_API_KEY=...        # free key from https://materialsproject.org
    pip install -r requirements-viz.txt
    python src/run_structure_visualization.py [--out DIR] [--phase KEY]
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG, no display needed

sys.path.append(str(Path(__file__).resolve().parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "predictions" / "structures"


def main() -> int:
    """Render the requested HfO2 structure figures to PNG."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT), help="Output directory for PNG figures"
    )
    parser.add_argument(
        "--api-key", default=None, help="Materials Project API key (else MP_API_KEY env)"
    )
    parser.add_argument(
        "--phase", default=None, help="Render only this phase key (default: all + grid)"
    )
    args = parser.parse_args()

    try:
        from visualization.structures import (
            CachedStructureProvider,
            CrystalStructureVisualizer,
            MaterialsProjectProvider,
            available_phase_keys,
        )

        provider = CachedStructureProvider(MaterialsProjectProvider(api_key=args.api_key))
        viz = CrystalStructureVisualizer(provider=provider)
    except (ImportError, ValueError) as exc:
        print(f"\n[error] {exc}\n")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.phase:
        keys = [args.phase]
    else:
        keys = available_phase_keys()

    print(f"Rendering HfO2 structures -> {out_dir}")
    for key in keys:
        save_path = out_dir / f"hfo2_{key}.png"
        viz.render_phase(key, save_path=str(save_path))
        print(f"  - {save_path.name}")

    if not args.phase:
        grid_path = out_dir / "hfo2_comparison.png"
        viz.render_comparison_grid(save_path=str(grid_path))
        print(f"  - {grid_path.name} (2x2 comparison)")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
