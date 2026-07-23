#!/bin/bash

# Created on 27.06.2025
# @author: eschlager

# Preprocessing of Hirham firnpack data; run after raw_processing.sh
# 1. chosse variables
# 2. apply mask to HIRHAM5 data
# 3. correct for negative rainfall and snfall

BASE_DIR="/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/raw"
RAW_DIR="$BASE_DIR/firnpack"
AUX_DIR="$BASE_DIR/AuxFiles"

OUT_DIR="/gws/pw/j07/aria_giant/wip/draft/acocac/data/hirham5/interim"

module load jaspy

mkdir -p "$OUT_DIR"

start_year=1990
stop_year=2021

# apply GL mask to HIRHAM5 data, and correct rainfall and snfall to >=0
# only select variables ahfl,ahfs,dlwrad,dswrad,evspsbl,rainfall,snfall,tas,snmel,gld,rogl,albedom,sn
cdo -r -f nc -settaxis,1990-01-01,12:00:00,1day -duplicate,366 ${AUX_DIR}/GRLmask.nc ${AUX_DIR}/GRLmask_366.nc
for ((i=start_year;i<=stop_year;i+=1))
	do
		echo "Processing $RAW_DIR/Daily2D_Merged_$i.nc..."
		# choose variables and apply GrIS mask
		cdo -r -f nc -div \
		 -selvar,ahfl,ahfs,dlwrad,dswrad,evspsbl,rainfall,snfall,tas,snmel,gld,rogl,albedom,sn,tsl ${RAW_DIR}/Daily2D_Merged_$i.nc \
		 -selvar,glacGRL ${AUX_DIR}/GRLmask_366.nc ${OUT_DIR}/Daily2D_GRL_tmp_$i.nc

		# make files with corrected rainfall and snfall
		cdo -setrtoc,-inf,0,0 -selvar,rainfall,snfall ${OUT_DIR}/Daily2D_GRL_tmp_$i.nc ${OUT_DIR}/Daily2D_GRL_precip_$i.nc
		# replace rainfall and snfall in file
		cdo replace ${OUT_DIR}/Daily2D_GRL_tmp_$i.nc ${OUT_DIR}/Daily2D_GRL_precip_$i.nc ${OUT_DIR}/Daily2D_GRL_$i.nc
		# remove the helper files
		rm ${OUT_DIR}/Daily2D_GRL_precip_$i.nc
		rm ${OUT_DIR}/Daily2D_GRL_tmp_$i.nc
	done
