#!/bin/bash -e
#SBATCH --job-name colbert
#SBATCH -c 16
#SBATCH --gres=gpu:ampere:4
#SBATCH --mem=400g
#SBATCH --output=log-%j.txt

CONTAINER_NAME=pyxis_ir
WORKSPACE="/mnt/ceph/storage/data-tmp/current/fschlatt/ecir23-incomplete-judgments/beir/"
COLLECTION_PATH="/mnt/ceph/storage/data-tmp/current/kibi9872/beir-ColBERT/colbert-datasets/trec-covid-beir/collection.tsv"
ROOT_DIR="/mnt/ceph/storage/data-tmp/current/fschlatt/ecir23-incomplete-judgments/beir/colbert-results/"
EXPERIMENT_NAME="trec-covid-beir"
CHECKPOINT_PATH="/mnt/ceph/storage/data-tmp/current/kibi9872/beir-ColBERT/msmarco.psg.l2/huggingface-checkpoint/"

enroot start \
	-w \
	-m ${WORKSPACE}:/workspace \
	-e CUDA_HOME=/opt/conda/pkgs/cuda-toolkit \
	${CONTAINER_NAME} \
	python dense_retrieval/index_colbert.py \
		--index_name trec-covid-beir \
		--collection_path ${COLLECTION_PATH} \
		--experiment_name ${EXPERIMENT_NAME} \
		--experiment_root_dir ${ROOT_DIR} \
		--checkpoint_path ${CHECKPOINT_PATH} \
		--num_procs 4
