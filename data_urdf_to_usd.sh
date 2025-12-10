#!/bin/bash

CONVERT_SCRIPT="$HOME/IsaacLab/scripts/tools/convert_urdf.py"

if [ "$1" == "gapartnet" ]; then
    GAPARTNET_DIR="data/gapartnet"
    for obj_dir in "$GAPARTNET_DIR"/*; do
        if [ -d "$obj_dir" ]; then
            urdf_file="$obj_dir/mobility_relabel_gapartnet.urdf"
            usd_file="$obj_dir/mobility_relabel_gapartnet.usd"
            if [ -f "$urdf_file" ]; then
                echo "Converting $urdf_file to $usd_file"
                ~/IsaacLab/isaaclab.sh -p "$CONVERT_SCRIPT" "$urdf_file" "$usd_file" --fix-base --joint-stiffness 0.0 --joint-damping 50.0 --headless
            fi
        fi
    done
# note: had to remove trailing space in dae filename in playboard.urdf and playboard_inv.urdf
# find $HOME/Tac-Man-Simulation/data/playboards -name 'playboard*.urdf' -exec sed -i 's/playboard\.dae /playboard.dae/g' {} +
elif [ "$1" == "playboards" ]; then
    PLAYBOARD_DIR="$HOME/Tac-Man-Simulation/data/playboards"
    for obj_dir in "$PLAYBOARD_DIR"/*/*; do
        if [ -d "$obj_dir" ]; then
            cd "$obj_dir" || continue
            urdf_file="playboard.urdf"
            usd_file="playboard.usd"
            if [ -f "$urdf_file" ]; then
                echo "Converting $urdf_file to $usd_file"
                ~/IsaacLab/isaaclab.sh -p "$CONVERT_SCRIPT" "$urdf_file" "$usd_file" --fix-base --joint-stiffness 0.0 --joint-damping 50.0 --headless
            fi
            urdf_file="playboard_inv.urdf"
            usd_file="playboard_inv.usd"
            if [ -f "$urdf_file" ]; then
                echo "Converting $urdf_file to $usd_file"
                ~/IsaacLab/isaaclab.sh -p "$CONVERT_SCRIPT" "$urdf_file" "$usd_file" --fix-base --joint-stiffness 0.0 --joint-damping 50.0 --headless
            fi
            cd - > /dev/null
        fi
    done
else
    echo "Usage: $0 {gapartnet|playboards}"
    exit 1
fi