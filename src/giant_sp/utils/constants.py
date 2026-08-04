"""Constants used throughout the codebase.

Defines named geographic regions (as longitude/latitude bounding boxes) and
the corresponding plot extents derived from them.
"""

# Region limits as [[lonmin, lonmax], [latmin, latmax]]
region_limits: dict[str, list[list[int]]] = {
    "greenland": [[-73, -12], [59, 85]],  # Greenland ice sheet
}

# Region extents for plots [xmin, xmax, ymin, ymax], flattened from region_limits
region_extents: dict[str, list[int]] = {
    key: [val for pair in region_limits[key] for val in pair] for key in region_limits
}
