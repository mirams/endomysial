"""Batch-run CZI annotation analysis across a Data folder tree.

Expected layout:
    Data/
      <top_level_folder>/
        Slice<x>/
          *.czi

For each .czi file found, this script computes measurements using the same
analysis functions as czi_annotation_analysis.py and writes a spreadsheet
(CSV) with columns:
    1) top_level_folder
    2) slice_number
    3) annotation_number
    4) full_line_length
    5) boundary_boundary_length
    6) cell_a_area
    7) cell_b_area

By default, lengths are written in micrometers and areas in µm².
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from czi_annotation_analysis import (
    load_czi,
    measure,
    pair_line_to_blobs,
    parse_annotations,
)


SLICE_RE = re.compile(r"^Slice(\d+)$", re.IGNORECASE)


def iter_czi_files(data_dir: Path):
    """Yield (top_level_name, slice_number, czi_path) entries.

    Entries are returned in sorted folder/file order.
    """
    for top_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        slice_dirs = [
            p for p in top_dir.iterdir()
            if p.is_dir() and SLICE_RE.match(p.name)
        ]
        for slice_dir in sorted(slice_dirs):
            match = SLICE_RE.match(slice_dir.name)
            if match is None:
                continue
            slice_num = int(match.group(1))
            for czi_path in sorted(slice_dir.glob("*.czi")):
                yield top_dir.name, slice_num, czi_path


def analyse_czi(czi_path: Path, channel: int, scene: int):
    """Return list of (annotation_index, full_len, boundary_len, area_a_um2, area_b_um2) tuples."""
    _, meta_xml, pixel_um, origin = load_czi(
        str(czi_path), channel=channel, scene=scene)
    blobs, lines = parse_annotations(
        meta_xml, dump=False, origin=origin, scene=scene)

    rows = []
    for idx, line in enumerate(lines, start=1):
        blob_a, blob_b = pair_line_to_blobs(line, blobs)
        meas = measure(line, blob_a, blob_b, pixel_um)
        if meas is None:
            continue
        rows.append((idx, meas["c2c_um"], meas["b2b_um"],
                     meas["area_a_um2"], meas["area_b_um2"]))
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Batch-run czi_annotation_analysis over Data/*/Slice*/.czi"
    )
    parser.add_argument(
        "--data-dir",
        default="Data",
        help="Path to Data folder (default: Data)",
    )
    parser.add_argument(
        "--output",
        default="czi_annotation_measurements.csv",
        help="Output CSV path (default: czi_annotation_measurements.csv)",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=0,
        help="Display/analysis channel passed to loader (default: 0)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    output_path = Path(args.output)
    written_rows = 0
    analysed_files = 0

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "top_level_folder",
            "slice_number",
            "annotation_number",
            "full_line_length",
            "boundary_boundary_length",
            "cell_a_area",
            "cell_b_area",
        ])

        for top_name, slice_num, czi_path in iter_czi_files(data_dir):
            print(f"[batch] analysing {czi_path}")
            try:
                scene_idx = max(0, slice_num - 1)
                measurements = analyse_czi(
                    czi_path, channel=args.channel, scene=scene_idx)
            except Exception as exc:
                print(f"[batch] failed {czi_path}: {exc}")
                continue

            analysed_files += 1
            for annotation_num, full_len, boundary_len, area_a, area_b in measurements:
                writer.writerow([
                    top_name,
                    slice_num,
                    annotation_num,
                    f"{full_len:.6f}",
                    f"{boundary_len:.6f}",
                    f"{area_a:.6f}" if area_a is not None else "",
                    f"{area_b:.6f}" if area_b is not None else "",
                ])
                written_rows += 1

    print(f"[batch] analysed files: {analysed_files}")
    print(f"[batch] rows written:   {written_rows}")
    print(f"[batch] output:         {output_path}")


if __name__ == "__main__":
    main()
