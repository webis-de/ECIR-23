#!/bin/bash
#SBATCH --job-name=ecir23-unjudged-documents
#SBATCH --output=./logs/output.%a.out
#
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=5g
#SBATCH --cpus-per-task=1

./src/main/python/run_evaluation_task.py --taskDefinititionFile src/main/resources/all-tasks.jsonl --taskNumber  $SLURM_ARRAY_TASK_ID

