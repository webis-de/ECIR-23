#!/bin/bash -e
#SBATCH --job-name colbert
#SBATCH -c 16
#SBATCH --gres=gpu:ampere
#SBATCH --mem=300g
#SBATCH --output=log-%j.txt

CONTAINER_NAME=pyxis_ir
WORKSPACE="/mnt/ceph/storage/data-tmp/current/fschlatt/health-question-answering/retrieval"
TRIP_CLICK_DIR="/mnt/ceph/storage/corpora/corpora-thirdparty/corpus-trip-click/"
#QUERIES_PATH=$TRIP_CLICK_DIR"2_TripClick_IR_Benchmark/benchmark_tsv/topics/topics.head.val.tsv"
#QUERIES_PATH=$TRIP_CLICK_DIR"2_TripClick_IR_Benchmark/benchmark_tsv/topics/topics.torso.val.tsv"
#QUERIES_PATH=$TRIP_CLICK_DIR"2_TripClick_IR_Benchmark/benchmark_tsv/topics/topics.tail.val.tsv"
#QUERIES_PATH=$TRIP_CLICK_DIR"2_TripClick_IR_Benchmark/benchmark_tsv/topics/topics.head.test.tsv"

ROOT_DIR="/mnt/ceph/storage/data-in-progress/data-research/web-search/health-question-answering/retrieval/colbert"
EXPERIMENT_NAME="colbert-trip-click"
#INDEX_NAME="trip-click.nbits=2"
INDEX_NAME="pubmed.nbits=2"
OUT_DIR="/mnt/ceph/storage/data-in-progress/data-research/web-search/health-question-answering/trec-health-misinfo"

enroot start \
	-w \
	-m ${WORKSPACE}:/workspace \
	-e CUDA_HOME=/opt/conda/pkgs/cuda-toolkit \
	${CONTAINER_NAME} \
	python -m dense_retrieval.retrieve_colbert \
		--topic_xml_paths ./data/topics*.xml \
		--doc_id_tsv_file /mnt/ceph/storage/corpora/corpora-thirdparty/corpus-pubmed/tsv/pubmed-abstracts.tsv \
		--index_name ${INDEX_NAME} \
		--experiment_name ${EXPERIMENT_NAME} \
		--experiment_root_dir ${ROOT_DIR} \
		--out_dir ${OUT_DIR} \
		--num_procs 1
