"""
czi_annotation_analysis.py
===========================
Loads a .czi file annotated in Zeiss Zen Lite with:
  - Blob shapes (ellipses, circles, or similar) drawn in pairs
  - Lines connecting each pair's centres

Tasks performed:
  1. Plot the image + all annotations so you can verify the load.
  2. Calculate centre-to-centre line lengths (in µm and pixels).
  3. Calculate boundary-to-boundary lengths (subtracting each blob's radius
     along the line direction from each end).
  4. Draw the boundary-to-boundary segment on top of the full line in a
     contrasting colour for visual verification.
  5. Print a summary table of all measurements.
  6. Optionally save the figure to disk.

Usage
-----
    python czi_annotation_analysis.py --file path/to/your.czi

Optional flags:
    --channel   INT   Which channel to display as background (default: 0)
    --save      STR   Save figure to this path (e.g. result.png).
                      If omitted the figure is shown interactively.
    --no-plot         Skip plotting (just print the table).
    --dump-xml        Print the raw annotation XML for debugging.
"""

import argparse
import math
import sys
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import czifile


# ── Colour scheme ────────────────────────────────────────────────────────────
BLOB_COLOUR = "#00d4ff"   # cyan  – blob outlines + centres
LINE_COLOUR = "#ffaa00"   # amber – full centre-to-centre line
BDY_COLOUR = "#ff2255"   # red   – boundary-to-boundary segment


# ═════════════════════════════════════════════════════════════════════════════
# 1.  LOAD CZI
# ═════════════════════════════════════════════════════════════════════════════

def _image_origin_px(czi):
    """
    Return the (X, Y) pixel-space origin of the image — i.e. where in
    Zen's global coordinate space the top-left of this acquisition sits.
    Annotation coordinates must have this subtracted to become image-
    relative pixels.

    Reads from czi.subblock_directory, which is present in all czifile
    versions and is the most direct source of truth.  Returns None if
    the origin cannot be determined, so the caller can fall back to the
    XML heuristic.
    """
    # Primary: iterate SubBlock directory entries directly.
    # Each entry has dimension_entries with .dimension ('X','Y',...) and
    # .start (integer pixel position in Zen's global space).
    try:
        min_x, min_y = float('inf'), float('inf')
        for entry in czi.subblock_directory:
            for de in entry.dimension_entries:
                dim = de.dimension.strip('\x00')
                if dim == 'X':
                    min_x = min(min_x, float(de.start))
                elif dim == 'Y':
                    min_y = min(min_y, float(de.start))
        if math.isfinite(min_x) and math.isfinite(min_y):
            return min_x, min_y
    except Exception:
        pass

    # Secondary: czi.start + czi.axes (available in older czifile builds).
    try:
        axes = czi.axes
        start = czi.start
        if 'X' in axes and 'Y' in axes:
            return float(start[axes.index('X')]), float(start[axes.index('Y')])
    except Exception:
        pass

    # Signal to the caller to use the XML fallback instead.
    return None


def load_czi(path, channel=0):
    """
    Returns
    -------
    image        : 2-D float32 ndarray, normalised to [0, 1], ready for display
    meta_xml     : str  – raw XML metadata
    pixel_um     : float – physical pixel size in µm (X axis)
    origin       : (float, float) – (X, Y) pixel offset of the image within
                   Zen's global coordinate system, for annotation alignment

    Compatible with czifile >= 2024 (no .axes / .scale() API).
    Uses asxarray() to get a labelled DataArray, then falls back to
    asarray() + manual squeeze if xarray is unavailable.
    """
    with czifile.CziFile(path) as czi:
        meta_xml = czi.metadata()          # always works – returns XML str

        # ── Pixel size from XML (works in all versions) ───────────────────
        pixel_um = _pixel_size_from_xml(meta_xml)

        # ── Image origin from subblock bounding box (reliable) ───────────
        origin = _image_origin_px(czi)

        # ── Image data ───────────────────────────────────────────────────
        # Prefer asxarray: gives named dims so we can select channel safely.
        try:
            xarr = czi.asxarray()          # xarray.DataArray with dim names
            print(f"[load] dims  : {dict(xarr.sizes)}")
            image = _squeeze_xarray(xarr, channel)
        except Exception as e:
            print(f"[load] asxarray failed ({e}); falling back to asarray …")
            data = czi.asarray()
            print(f"[load] shape (raw): {data.shape}")
            image = _squeeze_ndarray(data, channel)

    lo, hi = image.min(), image.max()
    image = (image.astype(np.float32) - lo) / (hi - lo + 1e-12)
    print(f"[load] display shape: {image.shape}")
    print(f"[load] pixel size   : {pixel_um:.6f} µm/px")
    print(f"[load] image origin : x={origin[0]:.0f} px, y={origin[1]:.0f} px")
    return image, meta_xml, pixel_um, origin


def _pixel_size_from_xml(meta_xml):
    """
    Read X pixel size (metres → µm) from Zeiss CZI metadata XML.
    Looks for  <Scaling><Items><Distance Id="X"><Value>…</Value>
    """
    try:
        root = ET.fromstring(meta_xml)
        for dist in root.findall('.//Scaling/Items/Distance'):
            if dist.get('Id') == 'X':
                val = dist.findtext('Value')
                if val:
                    return abs(float(val)) * 1e6
    except Exception:
        pass
    print("[warn] Could not read pixel size from XML; distances in pixels only.")
    return 1.0


def _squeeze_xarray(xarr, channel):
    """Reduce an xarray DataArray to a 2-D (Y, X) numpy array."""
    import numpy as np
    arr = xarr

    # Drop any singleton or unwanted dims one by one
    for dim in list(arr.dims):
        if dim in ('Y', 'X'):
            continue
        size = arr.sizes[dim]
        if size == 1:
            arr = arr.isel({dim: 0})
        elif dim == 'C':
            idx = min(channel, size - 1)
            arr = arr.isel({dim: idx})
        else:
            arr = arr.isel({dim: 0})   # T, Z, S, etc. → take first

    data = np.array(arr).squeeze()
    if data.ndim != 2:
        raise ValueError(
            f"Still not 2-D after xarray squeeze: shape={data.shape}")

    # Handle (X, Y) vs (Y, X) – xarray usually gives (Y, X) but check
    if arr.dims[-2:] == ('X', 'Y'):
        data = data.T
    return data


def _squeeze_ndarray(data, channel):
    """
    Last-resort squeeze for a raw ndarray when we have no dim labels.
    Assumes common Zeiss ordering and squeezes everything but the two
    largest spatial axes.
    """
    arr = data.squeeze()
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # Could be (C, Y, X) or (Y, X, C) or (Z, Y, X)
        # Pick the channel/slice with the highest variance as a heuristic
        # if channel index is valid, use it; otherwise take axis=0, idx=0
        if arr.shape[0] > arr.shape[2]:  # likely (Y, X, C)
            idx = min(channel, arr.shape[2] - 1)
            return arr[:, :, idx]
        else:                             # likely (C, Y, X) or (Z, Y, X)
            idx = min(channel, arr.shape[0] - 1)
            return arr[idx]
    # 4-D+: just keep peeling axis 0
    while arr.ndim > 2:
        idx = min(channel, arr.shape[0] - 1) if arr.ndim == 3 else 0
        arr = arr[idx]
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Cannot reduce array to 2-D: shape={data.shape}")
    return arr


# ═════════════════════════════════════════════════════════════════════════════
# 2.  PARSE ANNOTATIONS FROM XML
# ═════════════════════════════════════════════════════════════════════════════

def _coord_offset(root):
    """
    Zen stores annotation coordinates in the global stage/pixel coordinate
    system, which has an origin offset from the image top-left corner.
    The <MetadataNode> element gives us StartX / StartY (in pixels) that
    we must subtract to get image-relative pixel coordinates.
    """
    mn = root.find('.//MetadataNodes/MetadataNode')
    if mn is not None:
        try:
            return float(mn.get('StartX', 0)), float(mn.get('StartY', 0))
        except (TypeError, ValueError):
            pass
    return 0.0, 0.0


def parse_annotations(meta_xml, dump=False, origin=None):
    """
    Extract blob shapes and lines from Zen Lite XML metadata.

    Your file uses:
      - <Bezier> for blob outlines  (points as space-separated "x,y" pairs)
      - <Line>   for the connecting lines  (child elements X1/Y1/X2/Y2)

    All raw coordinates are in the global Zen pixel space; we subtract
    the image origin offset so everything is image-relative pixels.

    Parameters
    ----------
    origin : (float, float) or None
        (X, Y) pixel-space origin from czifile's subblock bounding box.
        When supplied this is used directly; otherwise the code falls
        back to parsing StartX/StartY from the XML (less reliable).

    Returns
    -------
    blobs : list of dict  {cx, cy, points, name}  – image-relative pixels
                cx/cy = centroid of the bezier polygon
                points = Nx2 numpy array of all boundary points
    lines : list of dict  {x1, y1, x2, y2, name} – image-relative pixels
    """
    if dump:
        print("\n── RAW ANNOTATION XML (first 6000 chars) ──────────────────")
        print(meta_xml[:6000])
        print("────────────────────────────────────────────────────────────\n")

    root = ET.fromstring(meta_xml)
    if origin is not None:
        off_x, off_y = origin
    else:
        off_x, off_y = _coord_offset(root)
    print(
        f"[parse] coordinate offset: x={off_x:.0f} px, y={off_y:.0f} px")

    blobs, lines = [], []

    for el in root.iter():
        tag = el.tag.split('}')[-1]   # strip any XML namespace

        # ── Blob shapes: Bezier, Ellipse, Circle, Polygon, Polyline ──────
        if tag.lower() in ('bezier', 'ellipse', 'circle', 'polygon',
                           'polyline', 'closedpolyline', 'openpolyline'):
            b = _parse_bezier_blob(el, off_x, off_y)
            if b:
                blobs.append(b)

        # ── Lines ─────────────────────────────────────────────────────────
        elif tag.lower() == 'line':
            ln = _parse_line(el, off_x, off_y)
            if ln:
                lines.append(ln)

    print(f"[parse] {len(blobs)} blob(s)  |  {len(lines)} line(s)")
    return blobs, lines


# ── XML element parsers ───────────────────────────────────────────────────────

def _parse_bezier_blob(el, off_x, off_y):
    """
    Parse a <Bezier> (or similar closed-polygon) element.

    Zen stores the outline as a space-separated string of "x,y" pairs
    inside <Geometry><Points>…</Points></Geometry>.

    Returns a dict with:
        points : np.ndarray  shape (N, 2)  image-pixel coordinates
        cx, cy : centroid in image pixels
        name   : label string
    """
    pts_el = el.find('.//Points')
    if pts_el is None or not pts_el.text:
        return None

    try:
        pairs = pts_el.text.strip().split()
        pts = np.array([list(map(float, p.split(','))) for p in pairs])
    except Exception as exc:
        print(
            f"[warn] Could not parse Points for {el.tag} id={el.get('Id')}: {exc}")
        return None

    if len(pts) < 3:
        return None

    # Apply coordinate offset
    pts[:, 0] -= off_x
    pts[:, 1] -= off_y

    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()

    # Extract a human-readable name / measurement label if present
    mt = el.findtext('.//MeasurementText') or el.findtext('.//Text') or \
        el.findtext('.//Name') or f"blob-{el.get('Id', '?')}"
    name = mt.strip().replace('\n', ' ')

    return dict(cx=cx, cy=cy, points=pts, name=name)


def _parse_line(el, off_x, off_y):
    """
    Parse a <Line> element.  Zen stores endpoints as child text elements:
        <X1>…</X1>  <Y1>…</Y1>  <X2>…</X2>  <Y2>…</Y2>
    """
    def _txt(tag):
        node = el.find(f'.//{tag}')
        return float(node.text.strip()) if node is not None and node.text else None

    x1, y1, x2, y2 = _txt('X1'), _txt('Y1'), _txt('X2'), _txt('Y2')
    if None in (x1, y1, x2, y2):
        return None

    mt = el.findtext('.//MeasurementText') or el.findtext('.//Text') or \
        f"line-{el.get('Id', '?')}"

    return dict(
        x1=x1 - off_x, y1=y1 - off_y,
        x2=x2 - off_x, y2=y2 - off_y,
        name=mt.strip().replace('\n', ' '),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PAIR LINES WITH BLOBS
# ═════════════════════════════════════════════════════════════════════════════

def pair_line_to_blobs(line, blobs, max_dist_px=200):
    """
    Match a line's two endpoints to the nearest blob centroids.

    max_dist_px is generous because the line endpoint sits ON the blob
    centroid (Zen draws lines between centres), so the distance should be
    very small — but we allow slack for imprecise manual annotation.

    Returns (blob_a, blob_b) – either may be None if no blob is close enough.
    """
    def nearest(px, py):
        best, best_d = None, float("inf")
        for b in blobs:
            d = math.hypot(b["cx"] - px, b["cy"] - py)
            if d < best_d:
                best_d, best = d, b
        return best if best_d <= max_dist_px else None

    return nearest(line["x1"], line["y1"]), nearest(line["x2"], line["y2"])


# ═════════════════════════════════════════════════════════════════════════════
# 4.  MEASURE
# ═════════════════════════════════════════════════════════════════════════════

def _polygon_boundary_trim(blob, line_start_xy, ux, uy):
    """
    Find how far along the line direction (ux, uy) we must travel from
    the line endpoint before hitting the blob's polygon boundary.

    Strategy: cast a ray from line_start_xy in direction (ux, uy) and find
    the first intersection with any edge of the polygon.

    Returns the trim distance in pixels. If no valid intersection is found,
    returns 0 so the endpoint is left untrimmed.
    """
    if blob is None:
        return 0.0

    pts = blob["points"]   # (N, 2)
    sx, sy = line_start_xy
    n = len(pts)

    # Ray: P(t) = (sx, sy) + t*(ux, uy),  t >= 0
    min_t = float("inf")
    for i in range(n):
        ax, ay = pts[i]
        bx, by = pts[(i + 1) % n]
        ex, ey = bx - ax, by - ay   # edge direction

        # Solve: (sx + t*ux, sy + t*uy) = (ax + s*ex, ay + s*ey)
        denom = ux * ey - uy * ex
        if abs(denom) < 1e-12:
            continue   # ray and edge are parallel

        dx, dy = ax - sx, ay - sy
        t = (dx * ey - dy * ex) / denom
        s = (dx * uy - dy * ux) / denom

        if t > 1e-6 and -1e-6 <= s <= 1.0 + 1e-6:
            min_t = min(min_t, t)

    if math.isinf(min_t):
        return 0.0

    return min_t


def measure(line, blob_a, blob_b, pixel_um):
    """
    Calculate centre-to-centre and boundary-to-boundary distances.

    For polygon blobs we ray-cast from each centroid along the line direction
    to find the exact boundary intersection point.

    Returns a dict, or None if the line has zero length.
    """
    dx = line["x2"] - line["x1"]
    dy = line["y2"] - line["y1"]
    c2c_px = math.hypot(dx, dy)
    if c2c_px < 1e-9:
        return None

    ux, uy = dx / c2c_px, dy / c2c_px   # unit vector from endpoint 1 → 2

    trim_a = _polygon_boundary_trim(blob_a, (line["x1"], line["y1"]),  ux,  uy)
    trim_b = _polygon_boundary_trim(blob_b, (line["x2"], line["y2"]), -ux, -uy)
    b2b_px = max(0.0, c2c_px - trim_a - trim_b)

    return dict(
        c2c_px=c2c_px,
        c2c_um=c2c_px * pixel_um,
        b2b_px=b2b_px,
        b2b_um=b2b_px * pixel_um,
        trim_a=trim_a,
        trim_b=trim_b,
        # Boundary-segment endpoints (pixels)
        bx1=line["x1"] + ux * trim_a,
        by1=line["y1"] + uy * trim_a,
        bx2=line["x2"] - ux * trim_b,
        by2=line["y2"] - uy * trim_b,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 5.  PRINT TABLE
# ═════════════════════════════════════════════════════════════════════════════

def print_table(results, pixel_um):
    w = 74
    print("\n" + "═" * w)
    print(f"  {'#':>3}  {'C→C (px)':>10}  {'C→C (µm)':>10}  "
          f"{'B→B (px)':>10}  {'B→B (µm)':>10}")
    print("─" * w)
    for i, (line, meas) in enumerate(results):
        if meas is None:
            print(f"  {i+1:>3}  (line {i+1} — could not pair blobs to endpoints)")
        else:
            print(f"  {i+1:>3}  "
                  f"{meas['c2c_px']:>10.2f}  "
                  f"{meas['c2c_um']:>10.2f}  "
                  f"{meas['b2b_px']:>10.2f}  "
                  f"{meas['b2b_um']:>10.2f}")
    print("═" * w)
    print("  C→C = centre-to-centre   |   B→B = boundary-to-boundary\n")


# ═════════════════════════════════════════════════════════════════════════════
# 6.  PLOT
# ═════════════════════════════════════════════════════════════════════════════

def plot(image, blobs, results, pixel_um, save_path=None):
    fig, ax = plt.subplots(figsize=(13, 10))
    ax.imshow(image, cmap="gray", origin="upper", interpolation="nearest")
    ax.set_title("CZI Annotation Analysis",
                 fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("x  (pixels)")
    ax.set_ylabel("y  (pixels)")

    # ── Blobs (Bezier polygons) ───────────────────────────────────────────
    for blob in blobs:
        pts = blob["points"]
        # Close the polygon by appending the first point
        closed = np.vstack([pts, pts[0]])
        ax.plot(closed[:, 0], closed[:, 1],
                color=BLOB_COLOUR, linewidth=1.5, linestyle="--", zorder=3)

    # ── Lines and measurements ────────────────────────────────────────────
    for i, (line, meas) in enumerate(results):
        ax.plot([line["x1"], line["x2"]], [line["y1"], line["y2"]],
                color=LINE_COLOUR, linewidth=2.2, zorder=5,
                label="Centre–centre" if i == 0 else "_")

        if meas is None:
            continue

        ax.plot([meas["bx1"], meas["bx2"]], [meas["by1"], meas["by2"]],
                color=BDY_COLOUR, linewidth=3.5, zorder=6,
                label="Boundary–boundary" if i == 0 else "_")

        mx = (meas["bx1"] + meas["bx2"]) / 2
        my = (meas["by1"] + meas["by2"]) / 2
        label = (f"#{i+1}\n"
                 f"b→b  {meas['b2b_um']:.2f} µm\n"
                 f"c→c  {meas['c2c_um']:.2f} µm")
        ax.text(mx + 6, my - 6, label,
                color="white", fontsize=7.5, zorder=7,
                bbox=dict(boxstyle="round,pad=0.25", fc="#111111", alpha=0.65))

    # ── Legend ────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(edgecolor=BLOB_COLOUR, facecolor="none",
                       linestyle="--", linewidth=1.5, label="Blob boundary"),
        Line2D([0], [0], color=LINE_COLOUR, linewidth=2.2,
               label="Centre–centre line"),
        Line2D([0], [0], color=BDY_COLOUR, linewidth=3.5,
               label="Boundary–boundary segment"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
              facecolor="#111111", labelcolor="white", framealpha=0.75)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] Figure saved → {save_path}")

    plt.show()   # always show interactively as well (unless --no-plot)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Analyse Zen Lite blob+line annotations in a .czi file."
    )
    ap.add_argument("--file",     required=True,
                    help="Path to .czi file")
    ap.add_argument("--channel",  type=int, default=0,
                    help="Display channel (default 0)")
    ap.add_argument("--save",     default=None,
                    help="Save figure to this path")
    ap.add_argument("--no-plot",  action="store_true",  help="Skip the plot")
    ap.add_argument("--dump-xml", action="store_true",
                    help="Print raw annotation XML")
    args = ap.parse_args()

    # 1. Load
    image, meta_xml, pixel_um, origin = load_czi(
        args.file, channel=args.channel)

    # 2. Parse annotations
    blobs, lines = parse_annotations(
        meta_xml, dump=args.dump_xml, origin=origin)

    if not lines:
        print("\n[warn] No lines found in the metadata.\n"
              "       Try running with --dump-xml to inspect the raw XML,\n"
              "       then open an issue with the output so the parser can be updated.")
        sys.exit(1)

    if not blobs:
        print(
            "\n[warn] No blob shapes found — line lengths will be centre-to-centre only.")

    # 3. Pair each line to its two blobs and measure
    results = []
    for line in lines:
        blob_a, blob_b = pair_line_to_blobs(line, blobs)
        meas = measure(line, blob_a, blob_b, pixel_um)
        results.append((line, meas))

    # 4. Print table
    print_table(results, pixel_um)

    # 5. Plot
    if not args.no_plot:
        plot(image, blobs, results, pixel_um, save_path=args.save)


if __name__ == "__main__":
    main()
