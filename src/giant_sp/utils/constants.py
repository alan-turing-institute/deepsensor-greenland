# Constants
# Use this file for definitions used throughout the codebase

# Region limits as lonmin,lonmax , latmin,latmax
region_limits = {
    "greenland": [[-73, -12], [59, 85]],  # Greenland ice sheet
}

# Region extents for plots [xmin, xmax, ymin, ymax]
region_extents = {
    key: [val for pair in region_limits[key] for val in pair] for key in region_limits
}
