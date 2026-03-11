#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html import escape
from itertools import combinations
from pathlib import Path

try:
    from PIL import Image, ImageStat
except ImportError as exc:  # pragma: no cover - exercised manually in this repo
    raise SystemExit(
        "Pillow is required to analyze the sample photos. "
        "Install it with `python3 -m pip install Pillow` and rerun the script."
    ) from exc


PALETTE = {
    "black": (30, 30, 30),
    "white": (235, 235, 235),
    "gray": (128, 128, 128),
    "red": (200, 65, 65),
    "orange": (220, 140, 70),
    "yellow": (215, 200, 80),
    "green": (90, 160, 95),
    "blue": (80, 120, 200),
    "purple": (145, 105, 185),
    "pink": (220, 155, 185),
    "brown": (135, 95, 60),
    "beige": (210, 190, 160),
}

ITEM_CODE_PATTERN = re.compile(r"(?:(?:GOOD|ERR)-)?([A-Z]+-[A-Z]+-\d+)")


@dataclass
class ImageSummary:
    file_name: str
    width: int
    height: int
    aspect_ratio: float
    orientation: str
    item_code: str
    variant_label: str
    average_rgb: tuple[int, int, int]
    brightness: float
    brightness_band: str
    contrast: float
    contrast_band: str
    dominant_colors: list[str]
    average_hash: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze local input photos and create a lightweight FashionGraph-style graph."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("input"),
        help="Directory containing the sample photos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory where graph artifacts should be written.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=int,
        default=10,
        help="Maximum average-hash Hamming distance for creating visual-similarity edges.",
    )
    return parser.parse_args()


def extract_item_code(stem: str) -> str:
    match = ITEM_CODE_PATTERN.match(stem)
    return match.group(1) if match else stem


def extract_variant_label(stem: str, item_code: str) -> str:
    if stem.startswith("GOOD-"):
        return "good"
    if stem.startswith("ERR-"):
        return "error_sample"
    if stem == item_code:
        return "source"

    suffix = stem[len(item_code) :].strip("-_")
    return suffix or "variant"


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    grayscale = image.convert("L").resize((hash_size, hash_size))
    pixels = list(grayscale.tobytes())
    avg = sum(pixels) / len(pixels)
    return "".join("1" if pixel >= avg else "0" for pixel in pixels)


def hamming_distance(left: str, right: str) -> int:
    return sum(left_bit != right_bit for left_bit, right_bit in zip(left, right))


def classify_brightness(value: float) -> str:
    if value < 100:
        return "dark"
    if value < 180:
        return "balanced"
    return "bright"


def classify_contrast(value: float) -> str:
    if value < 35:
        return "soft"
    if value < 65:
        return "moderate"
    return "high"


def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    def distance(palette_rgb: tuple[int, int, int]) -> float:
        return math.sqrt(sum((component - reference) ** 2 for component, reference in zip(rgb, palette_rgb)))

    return min(PALETTE, key=lambda name: distance(PALETTE[name]))


def dominant_colors(image: Image.Image, top_n: int = 3) -> list[str]:
    resized = image.convert("RGB").resize((64, 64))
    raw = resized.tobytes()
    counts: Counter[str] = Counter()
    for red, green, blue in zip(raw[0::3], raw[1::3], raw[2::3]):
        rgb = (red, green, blue)
        counts[nearest_color_name(rgb)] += 1
    return [name for name, _count in counts.most_common(top_n)]


def analyze_image(path: Path) -> ImageSummary:
    with Image.open(path) as image:
        rgb_image = image.convert("RGB")
        grayscale = rgb_image.convert("L")

        rgb_stats = ImageStat.Stat(rgb_image)
        gray_stats = ImageStat.Stat(grayscale)
        width, height = rgb_image.size
        stem = path.stem
        item_code = extract_item_code(stem)
        brightness = round(gray_stats.mean[0], 2)
        contrast = round(gray_stats.stddev[0], 2)
        average_rgb = (
            round(rgb_stats.mean[0]),
            round(rgb_stats.mean[1]),
            round(rgb_stats.mean[2]),
        )

        return ImageSummary(
            file_name=path.name,
            width=width,
            height=height,
            aspect_ratio=round(width / height, 3),
            orientation="portrait" if height >= width else "landscape",
            item_code=item_code,
            variant_label=extract_variant_label(stem, item_code),
            average_rgb=average_rgb,
            brightness=brightness,
            brightness_band=classify_brightness(brightness),
            contrast=contrast,
            contrast_band=classify_contrast(contrast),
            dominant_colors=dominant_colors(rgb_image),
            average_hash=average_hash(rgb_image),
        )


def build_graph(images: list[ImageSummary], similarity_threshold: int) -> dict:
    nodes: list[dict] = []
    edges: list[dict] = []
    product_to_images: defaultdict[str, list[ImageSummary]] = defaultdict(list)

    for image in images:
        product_to_images[image.item_code].append(image)

    for item_code, grouped_images in sorted(product_to_images.items()):
        nodes.append(
            {
                "id": f"product:{item_code}",
                "type": "product",
                "label": item_code,
                "image_count": len(grouped_images),
            }
        )

    attribute_nodes: dict[str, dict] = {}

    for image in images:
        image_node_id = f"image:{image.file_name}"
        nodes.append(
            {
                "id": image_node_id,
                "type": "image",
                "label": image.file_name,
                "item_code": image.item_code,
                "variant_label": image.variant_label,
                "orientation": image.orientation,
                "brightness_band": image.brightness_band,
                "contrast_band": image.contrast_band,
                "dominant_colors": image.dominant_colors,
                "average_rgb": image.average_rgb,
                "width": image.width,
                "height": image.height,
                "aspect_ratio": image.aspect_ratio,
            }
        )
        edges.append(
            {
                "source": f"product:{image.item_code}",
                "target": image_node_id,
                "relation": "has_photo",
            }
        )

        attributes = {
            "orientation": [image.orientation],
            "brightness": [image.brightness_band],
            "contrast": [image.contrast_band],
            "color": image.dominant_colors[:2],
            "variant": [image.variant_label],
        }

        for attribute_type, values in attributes.items():
            for value in values:
                attribute_id = f"{attribute_type}:{value}"
                if attribute_id not in attribute_nodes:
                    attribute_nodes[attribute_id] = {
                        "id": attribute_id,
                        "type": "attribute",
                        "attribute_type": attribute_type,
                        "label": value.replace("_", " "),
                    }
                edges.append(
                    {
                        "source": image_node_id,
                        "target": attribute_id,
                        "relation": attribute_type,
                    }
                )

    nodes.extend(sorted(attribute_nodes.values(), key=lambda node: (node["attribute_type"], node["label"])))

    for left, right in combinations(images, 2):
        distance = hamming_distance(left.average_hash, right.average_hash)
        same_item = left.item_code == right.item_code
        if distance <= similarity_threshold:
            edges.append(
                {
                    "source": f"image:{left.file_name}",
                    "target": f"image:{right.file_name}",
                    "relation": "visually_similar",
                    "hash_distance": distance,
                    "same_item_code": same_item,
                }
            )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "similarity_threshold": similarity_threshold,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def layout_nodes(graph: dict) -> dict[str, tuple[int, int]]:
    positions: dict[str, tuple[int, int]] = {}
    grouped: defaultdict[str, list[dict]] = defaultdict(list)
    for node in graph["nodes"]:
        grouped[node["type"]].append(node)

    column_x = {"product": 140, "image": 500, "attribute": 980}
    base_y = 120
    gap_y = 150

    for node_type, nodes in grouped.items():
        for index, node in enumerate(sorted(nodes, key=lambda entry: entry["id"])):
            positions[node["id"]] = (column_x[node_type], base_y + index * gap_y)

    return positions


def render_svg(graph: dict) -> str:
    positions = layout_nodes(graph)
    height = max(y for _x, y in positions.values()) + 120
    width = 1220

    node_colors = {"product": "#ffd166", "image": "#8ecae6", "attribute": "#d9ed92"}
    edge_colors = {
        "has_photo": "#555555",
        "orientation": "#5b8e7d",
        "brightness": "#b56576",
        "contrast": "#6d597a",
        "color": "#3a86ff",
        "variant": "#ff006e",
        "visually_similar": "#fb8500",
    }

    def multiline_label(node: dict) -> list[str]:
        if node["type"] == "product":
            return [node["label"], f'{node["image_count"]} photo(s)']
        if node["type"] == "attribute":
            return [node["attribute_type"], node["label"]]
        return [
            node["label"],
            f'{node["variant_label"]} · {node["orientation"]}',
            f'{node["brightness_band"]} · {node["contrast_band"]}',
            ", ".join(node["dominant_colors"][:2]),
        ]

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        'text { font-family: Arial, sans-serif; fill: #1f2933; }',
        '.edge-label { font-size: 12px; fill: #374151; }',
        '.title { font-size: 24px; font-weight: bold; }',
        '.subtitle { font-size: 14px; fill: #4b5563; }',
        '</style>',
        '<rect width="100%" height="100%" fill="#f8fafc" />',
        '<text x="40" y="50" class="title">FashionGraph input photo graph</text>',
        '<text x="40" y="76" class="subtitle">Products, image descriptors, and visual similarity links generated from ./input photos</text>',
    ]

    for edge in graph["edges"]:
        start = positions[edge["source"]]
        end = positions[edge["target"]]
        dashed = ' stroke-dasharray="8 6"' if edge["relation"] == "visually_similar" else ""
        svg_parts.append(
            f'<line x1="{start[0]}" y1="{start[1]}" x2="{end[0]}" y2="{end[1]}" '
            f'stroke="{edge_colors[edge["relation"]]}" stroke-width="2"{dashed} />'
        )
        label_x = round((start[0] + end[0]) / 2)
        label_y = round((start[1] + end[1]) / 2) - 8
        label = edge["relation"].replace("_", " ")
        if edge["relation"] == "visually_similar":
            label = f'similar ({edge["hash_distance"]})'
        svg_parts.append(f'<text x="{label_x}" y="{label_y}" text-anchor="middle" class="edge-label">{escape(label)}</text>')

    for node in graph["nodes"]:
        x, y = positions[node["id"]]
        width_box = 240 if node["type"] == "image" else 180
        height_box = 92 if node["type"] == "image" else 64
        labels = multiline_label(node)
        svg_parts.append(
            f'<rect x="{x - width_box / 2}" y="{y - height_box / 2}" width="{width_box}" height="{height_box}" '
            f'rx="18" fill="{node_colors[node["type"]]}" stroke="#1f2933" stroke-width="2" />'
        )
        for line_index, line in enumerate(labels):
            text_y = y - (len(labels) - 1) * 12 + line_index * 22
            svg_parts.append(f'<text x="{x}" y="{text_y}" text-anchor="middle" font-size="14">{escape(line)}</text>')

    svg_parts.append("</svg>")
    return "\n".join(svg_parts)


def render_html(graph: dict, svg: str) -> str:
    rows = []
    for node in sorted((node for node in graph["nodes"] if node["type"] == "image"), key=lambda item: item["label"]):
        rows.append(
            "<tr>"
            f"<td>{escape(node['label'])}</td>"
            f"<td>{escape(node['item_code'])}</td>"
            f"<td>{escape(node['variant_label'])}</td>"
            f"<td>{node['width']}×{node['height']}</td>"
            f"<td>{escape(node['orientation'])}</td>"
            f"<td>{escape(node['brightness_band'])}</td>"
            f"<td>{escape(node['contrast_band'])}</td>"
            f"<td>{escape(', '.join(node['dominant_colors']))}</td>"
            "</tr>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>FashionGraph input photo graph</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f8fafc; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    p {{ margin-top: 0; }}
    .panel {{ background: white; border-radius: 16px; padding: 20px; box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08); margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; padding: 10px; text-align: left; }}
    th {{ background: #f1f5f9; }}
    svg {{ width: 100%; height: auto; }}
    code {{ background: #e2e8f0; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>FashionGraph input photo graph</h1>
    <p>This lightweight graph is generated directly from the local <code>./input/</code> photos, using filename grouping plus image descriptors (orientation, brightness, contrast, dominant colors, and perceptual-hash similarity).</p>
    {svg}
  </div>
  <div class="panel">
    <h2>Image summaries</h2>
    <table>
      <thead>
        <tr>
          <th>File</th>
          <th>Item code</th>
          <th>Variant</th>
          <th>Size</th>
          <th>Orientation</th>
          <th>Brightness</th>
          <th>Contrast</th>
          <th>Dominant colors</th>
        </tr>
      </thead>
      <tbody>
        {'\n        '.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise SystemExit(f"No supported images were found in {input_dir}")

    images = [analyze_image(path) for path in image_paths]
    graph = build_graph(images, similarity_threshold=args.similarity_threshold)
    graph["images"] = [asdict(image) for image in images]

    graph_path = output_dir / "sample_photo_graph.json"
    svg_path = output_dir / "sample_photo_graph.svg"
    html_path = output_dir / "sample_photo_graph.html"

    svg = render_svg(graph)
    html = render_html(graph, svg)

    graph_path.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    svg_path.write_text(svg, encoding="utf-8")
    html_path.write_text(html, encoding="utf-8")

    print(f"Analyzed {len(images)} image(s) from {input_dir}")
    print(f"Wrote graph JSON to {graph_path}")
    print(f"Wrote graph SVG to {svg_path}")
    print(f"Wrote graph HTML to {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
