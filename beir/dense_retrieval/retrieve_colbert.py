import argparse
import pathlib

import pandas as pd
from colbert import Searcher
from colbert.data import Queries
from colbert.infra import Run, RunConfig
from util.read_xml_topics import read_xml_file


def main(args=None):

    parser = argparse.ArgumentParser()

    parser.add_argument("--query_tsv_path", type=pathlib.Path)
    parser.add_argument("--topic_xml_paths", type=pathlib.Path, nargs="+")
    parser.add_argument("--index_name", type=pathlib.Path, required=True)
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--experiment_root_dir", type=pathlib.Path, required=True)
    parser.add_argument("--out_dir", type=pathlib.Path, required=True)
    parser.add_argument("--doc_id_tsv_file", type=pathlib.Path, default=None)

    parser.add_argument("--num_procs", type=int, default=1)
    parser.add_argument("--k", type=int, default=100)

    args = parser.parse_args(args)

    doc_ids = []
    if args.doc_id_tsv_file:
        with args.doc_id_tsv_file.open("r") as file:
            for line in file:
                doc_id = int(line.split("\t")[0])
                doc_ids.append(doc_id)

    with Run().context(
        RunConfig(
            nranks=args.num_procs,
            experiment=args.experiment_name,
            root=str(args.experiment_root_dir),
        )
    ):

        if args.topic_xml_paths:
            query_data = []
            for xml_path in args.topic_xml_paths:
                query_data.extend(read_xml_file(xml_path))
            query_data = sorted(query_data, key=lambda x: x["topic_id"])
            query_data = {
                query_dict["topic_id"]: query_dict["cause"] + " " + query_dict["effect"]
                for query_dict in query_data
            }
            queries = Queries(data=query_data)
            queries_name = "trec-health-misinfo"
        elif args.query_tsv_path:
            queries = Queries(str(args.query_tsv_path))
            queries_name = args.queries_path.stem
        else:
            raise ValueError()

        searcher = Searcher(index=str(args.index_name))
        ranking = searcher.search_all(queries, k=args.k)

        data = []
        for topic_id, hits in ranking.data.items():
            for hit in hits:
                data.append([topic_id, *hit])

        df = pd.DataFrame(data, columns=["qid", "doc_id", "rank", "score"])
        if doc_ids:
            df["doc_id"] = df["doc_id"].map(lambda x: doc_ids[x])

        df.to_csv(
            args.out_dir / f"{args.index_name}.{queries_name}.{args.k}.tsv",
            sep="\t",
            index=False,
            header=False,
        )


if __name__ == "__main__":
    main()
