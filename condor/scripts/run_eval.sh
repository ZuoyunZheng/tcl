#!/bin/bash
cd /work/scratch/zzheng/github/tcl
args=()
for i in "$@"; do
	args+=" $i"
done
source /work/scratch/zzheng/bin/python/openmm/bin/activate
source /work/scratch/zzheng/github/tcl/envvars.sh
python3 /work/scratch/zzheng/github/tcl/main.py --resume /work/scratch/zzheng/github/tcl/output/tcl.pth --cfg $args;
