#!/bin/sh
### ------------- specify queue name ----------------
### -q gpua100: for GPU queue
### -q hpc: for faster queue
#BSUB -q gpua100

### ------------- specify gpu request----------------
#BSUB -gpu "num=1:mode=exclusive_process"

### ------------- specify job name ----------------
#BSUB -J dl-group-64

### ------------- specify number of cores ----------------
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### ------------- specify CPU memory requirements ----------------
#BSUB -R "rusage[mem=20GB]"

### ------------- specify wall-clock time (max allowed is 12:00)---------------- 
#BSUB -W 12:00

#BSUB -o YOUR_PATH/hpc_logs/OUTPUT_FILE%J.out
#BSUB -e YOUR_PATH/hpc_logs/OUTPUT_FILE%J.err

module load python3/3.11.9

source YOUR_PATH .venv/bin/activate
python YOUR_PATH/src/sample.py