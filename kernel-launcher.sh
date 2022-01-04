#!/bin/bash
module load python/3.6
module load cuda/10.0
module load xpdf

##  source activate ntf

export KERNELLAUNCHER="kernel-launcher.sh"

exec /global/home/users/mjwilson/.conda/envs/ntf/bin/python \
    -m ipykernel_launcher "$@"
