#!/bin/bash
#SBATCH --job-name=ecir23-unjudged-cross-validation
#SBATCH --output=./logs-cross-validation/output.%a.out
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30g
#SBATCH --cpus-per-task=2

./src/main/python/run_evaluation_task.py --taskDefinititionFile src/main/resources/cross-validation-tasks.jsonl --taskNumber  $SLURM_ARRAY_TASK_ID

