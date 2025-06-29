import os
import json
import csv
import numpy as np
from PIL import Image
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.world_object import Wall, Floor, Door, Key, Ball, Box, Lava, Goal

# Size of each tile in pixels
TILE_SIZE = 32

# List of (name, WorldObj instance) to be rendered
tile_defs = []

# Add single-variant objects
tile_defs.append(("wall", Wall()))
tile_defs.append(("floor", Floor()))
tile_defs.append(("lava", Lava()))   # Default is red, defined in Lava class
tile_defs.append(("goal", Goal()))

# Add color variants for doors, keys, balls, boxes
for color in COLOR_NAMES:
    tile_defs.append((f"door_{color}_closed", Door(color=color, is_open=False, is_locked=False)))
    tile_defs.append((f"door_{color}_open", Door(color=color, is_open=True, is_locked=False)))
    tile_defs.append((f"door_{color}_locked", Door(color=color, is_open=False, is_locked=True)))
    tile_defs.append((f"key_{color}", Key(color)))
    tile_defs.append((f"ball_{color}", Ball(color)))
    tile_defs.append((f"box_{color}", Box(color)))

# Render tiles into individual RGBA images
tiles = []
labels = []

for name, obj in tile_defs:
    # Create a float32 canvas with 4 channels: R, G, B, A
    img = np.zeros((TILE_SIZE, TILE_SIZE, 4), dtype=np.float32)

    # Render RGB channels using the object's render function
    obj.render(img[..., :3])

    # Compute alpha channel: set to 1.0 where any RGB value is non-zero
    img[..., 3] = (img[..., :3].sum(axis=2) > 0).astype(np.float32)

    # Clamp pixel values to the range [0, 1] to prevent overflow
    img = np.clip(img, 0.0, 1.0)

    # Convert to uint8 and create a PIL image in RGBA format
    pil_img = Image.fromarray((img * 255).astype(np.uint8), mode='RGBA')

    tiles.append(pil_img)
    labels.append(name)

# Create a blank tileset atlas image
tileset_width = TILE_SIZE * len(tiles)
tileset_height = TILE_SIZE
tileset = Image.new("RGBA", (tileset_width, tileset_height), (0, 0, 0, 0))

# Paste each tile into the atlas horizontally
for i, tile in enumerate(tiles):
    tileset.paste(tile, (i * TILE_SIZE, 0))

# Create output directory
os.makedirs("output", exist_ok=True)

# Save the tileset image
tileset.save("output/minigrid_tileset.png")

# Save tile index mapping to CSV
with open("output/minigrid_tileset_mapping.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for i, name in enumerate(labels):
        writer.writerow([i, name])

# Save tile index mapping to JSON
with open("output/minigrid_tileset_mapping.json", "w") as f:
    json.dump({name: i for i, name in enumerate(labels)}, f, indent=2)

print("Tileset saved to: output/minigrid_tileset.png")
print("Mapping saved to: output/minigrid_tileset_mapping.csv and .json")
