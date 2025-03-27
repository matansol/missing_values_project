#!/bin/bash

#SBATCH -o ./out/%j.txt
#SBATCH -e ./err/%j.txt

# Activate conda environment
source /home/lotan.amit/miniconda3/etc/profile.d/conda.sh
conda activate /home/lotan.amit/MSC/.conda

job_id="--j=$SLURM_JOB_ID"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --e) exp_id="--e=$2"; shift ;;
        --ds) dataset_name="--ds=$2"; shift ;;
        --im) imputer="--im=$2"; shift ;;
        --s) strategy="--s=$2"; shift ;;
        --mr) missing_rate="--mr=$2"; shift ;;
        --rs) random_state="--rs=$2"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

echo "job_id: $job_id"
echo "exp_id: $exp_id"
echo "dataset_name: $dataset_name"
echo "imputer: $imputer"
echo "strategy: $strategy"
echo "missing_rate: $missing_rate"
echo "random_state: $random_state"

python3 exp_imputations.py \
 $job_id $exp_id $dataset_name $imputer $strategy $missing_rate $random_state