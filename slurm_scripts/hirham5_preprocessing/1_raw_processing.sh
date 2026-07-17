#!/bin/bash

# Modified on 01.07.2026 by Alejandro Coca-Castro
# Created on 25.06.2025
# @author: eschlager

# Initial preprocessing of Hirham data: 
# 1. loop through years
# 2. Reformat time axis
# 3. Remap all variables to regional grid
# 4. set dimension names to (x,y) for all files
# 5. save data as annual files

set -e  # Exit on error

module load jaspy

RAW_DIR="/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/raw_annual"
BASE_DIR="/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/raw"
OUTPUT_DIR="$BASE_DIR/firnpack"
AUX_DIR="$BASE_DIR/AuxFiles"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$AUX_DIR"

for year in {1980..2021}; do
    echo "Merging files for year $year..."

    files=$(ls ${RAW_DIR}/${year}/Daily2D_${year}*.nc)
    echo "$files"
    
    GRID="$AUX_DIR/curvilinear_grid.nc"
    echo $GRID
    if [ ! -f "$GRID" ]; then
        ncfile=$(ls ${RAW_DIR}/${year}/Daily2D_${year}*.nc | head -n 1)
        echo "make curvilinear grid file from $ncfile"
        cdo -selname,snmel -seltimestep,1 $ncfile $GRID
    fi


    if [ -n "$files" ]; then
        output_file="$OUTPUT_DIR/Daily2D_Merged_${year}.nc"
        cdo -r -f nc mergetime \
            $(for f in $files; do
            echo "-setmissval,nan -settime,12:00:00 -setgrid,$GRID $f"
            done) \
            "$output_file"
        
        if ncdump -h f"$output_file" | grep -q 'dimensions: rlat'; then
            echo "Rename dimensions (rlat,rlon) to (y,x)..."
            ncrename -D 0 -d rlat,y -d rlon,x "$output_file"
        fi

        echo "Saved merged file to $output_file"
    
    else
        echo "No files found for year $year"
    fi

done

echo "All data files merged."

echo "Process AuxFiles..."

MASK="$AUX_DIR/original/GL2offline_CountryGlacierMasks.nc"
AREA="$AUX_DIR/original/HHgrid_maskall_ISMIP6.nc"
GRID="$AUX_DIR/curvilinear_grid.nc"

# create file with elevation of HIRHAM5 grid cells
cdo setgrid,${GRID} -selvar,elev ${AUX_DIR}/original/SMBmodel_LatLonElev.nc ${AUX_DIR}/h5_elevation.nc

# create file with Greenland mask and basins masks
cdo setgrid,${GRID} -setmissval,nan -merge -setctomiss,0. -selvar,glacGRL ${MASK} -setctomiss,0. -selvar,maskbas ${AREA} ${AUX_DIR}/GRLmask.nc

# set same coordinates for runoff timescale
cdo setgrid,${GRID} ${AUX_DIR}/original/SMBmodel_ElevGrad.nc ${AUX_DIR}/runoff_timescale.nc