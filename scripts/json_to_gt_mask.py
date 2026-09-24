import json
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

JSON_PATH = Path("data/test_images/sample3.json")
OUTPUT_PATH = Path("data/test_images/sample3_chair_gt.png")

TARGET_LABEL = "chair"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

width = data["imageWidth"]
height = data["imageHeight"]

# 검정색 배경 마스크
mask = Image.new("L", (width, height), 0)
draw = ImageDraw.Draw(mask)

count = 0

for shape in data["shapes"]:
    if shape["label"] != TARGET_LABEL:
        continue

    if shape["shape_type"] != "polygon":
        continue

    points = [
        (round(x), round(y))
        for x, y in shape["points"]
    ]

    # chair 영역 = 흰색(255)
    draw.polygon(points, fill=255)

    count += 1

mask.save(OUTPUT_PATH)

print(f"label: {TARGET_LABEL}")
print(f"polygons: {count}")
print(f"mask size: {width} x {height}")
print(f"saved: {OUTPUT_PATH}")