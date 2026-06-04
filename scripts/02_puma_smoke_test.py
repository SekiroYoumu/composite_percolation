from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pipeline_common import (
    available_pumapy_functions,
    center_crop_slices,
    compute_transport,
    field_shape,
    load_config,
    read_volume_as_zyx,
    sample_label_path,
    sample_results_dir,
    save_flux_slice_png,
    um_size_to_zyx_voxels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small PuMA smoke test on center-cropped label volumes.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--voxels", nargs=3, type=int, metavar=("NX", "NY", "NZ"), help="Override smoke crop size in voxels.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    direction = cfg["through_plane_axis"].lower()

    try:
        import pumapy  # noqa: F401
    except Exception as exc:
        print(f"Could not import pumapy: {exc}")
        print("Available PuMA-related functions:")
        for name in available_pumapy_functions():
            print(f"  {name}")
        return 2

    for sample_key in cfg["samples"]:
        out_dir = sample_results_dir(cfg, sample_key)
        label_path = sample_label_path(cfg, sample_key)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing {label_path}; run 01_validate_and_merge_masks.py first.")
        label = read_volume_as_zyx(label_path, "z_y_x")

        if args.voxels:
            size_zyx = (args.voxels[2], args.voxels[1], args.voxels[0])
        elif cfg.get("smoke_test_voxels") is not None:
            nx, ny, nz = cfg["smoke_test_voxels"]
            size_zyx = (int(nz), int(ny), int(nx))
        else:
            size_zyx = um_size_to_zyx_voxels(cfg.get("smoke_test_um", [5, 5, 5]), cfg)

        crop = center_crop_slices(label.shape, size_zyx)
        block = label[crop]
        print(f"\n[{sample_key}] smoke crop shape_zyx={block.shape}, direction={direction}, side_bc={cfg.get('side_bc')}")
        try:
            result = compute_transport(block, cfg, prefer_electrical=True)
        except Exception as exc:
            print(f"PuMA smoke test failed for {sample_key}: {type(exc).__name__}: {exc}")
            print("Available conductivity/workspace functions in pumapy:")
            for name in available_pumapy_functions():
                print(f"  {name}")
            return 3

        print(f"  solver: {result['solver_name']}")
        print(f"  normalized effective transport coefficient: {result['keff_norm']:.8g}")
        print(f"  potential field shape: {field_shape(result.get('potential'))}")
        print(f"  flux field shape: {field_shape(result.get('flux'))}")
        print(f"  runtime_s: {result.get('runtime_s', np.nan):.3f}")
        png = out_dir / "smoke_flux_magnitude_center_slice.png"
        save_flux_slice_png(result.get("flux_magnitude"), png, f"{sample_key} smoke flux magnitude", direction)
        if png.exists():
            print(f"  saved flux magnitude center slice: {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
