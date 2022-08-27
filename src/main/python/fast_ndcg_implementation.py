import numpy as np
import pandas as pd
import json
from trectools import TrecRun


class MemorizedNdcg:
    def __init__(self, depth, trec_eval=True, labels=(0, 1, 2, 3)):
        self.__depth = depth
        self.__labels = labels
        self.__trec_eval = trec_eval
        self.__gains = self.__calculate_gains()

    def __calculate_gains(self):
        """
        :return: A map of type rank -> label -> gain
        """
        ret = {i: {} for i in range(1, self.__depth + 1)}

        for label in self.__labels:
            for rank in ret.keys():
                discount = 1. / np.log2(rank+1)

                if self.__trec_eval:
                    ret[rank][label] = label * discount
                else:
                    ret[rank][label] = (2 ** label - 1.0) * discount

        return ret

    def __get_gain(self, rank, label):
        return self.__gains[rank][label]

    def get_ndcg(self, run, qrels, depth, trec_eval=True):
        queries = qrels.qrels_data['query'].unique()
        ret = []

        if self.__depth != depth:
            raise ValueError(f'Depth is wrong {depth}')

        if self.__trec_eval != trec_eval:
            raise ValueError(f'Config for trec_eval is wrong {trec_eval}')

        for query in queries:
            qrels_for_query = qrels.qrels_data[qrels.qrels_data['query'] == query]
            run_for_query = self.__run_for_query(run, query, qrels_for_query)
            idcg, free_gains = self.__sum_dcg_(self.__perfect_ranking(qrels_for_query))
            if len(free_gains) != 0:
                raise ValueError('IDCG calculation has free parameters')

            dcg_incomplete, free_gains = self.__sum_dcg_(run_for_query)

            ret += [ImmediateNdcgResultPerQuery(
                idcg=idcg,
                dcg_incomplete=dcg_incomplete,
                free_gains=free_gains,
                query=query
            )]

        tmp_ret = {}
        for i in ret:
            if i.query in tmp_ret:
                raise ValueError('')

            tmp_ret[i.query] = i

        return tmp_ret

    def __run_for_query(self, run, query, qrels_for_query):
        if type(run) == TrecRun:
            return self.__run_for_query(run.run_data, query, qrels_for_query)
        if 'query' not in run.columns:
            return self.__run_for_query(
                pd.DataFrame(columns=["query", "q0", "docid", "rank", "score", "system"]),
                query, qrels_for_query
            )

        return self.__ranking_with_labels(run[run['query'] == query], qrels_for_query)

    def __sum_dcg_(self, run_with_rel):
        free_gains = {}
        dcg = 0

        for _, i in run_with_rel.iterrows():
            rank = i['rank']

            if np.isnan(i['rel']):
                if i['docid'] in free_gains:
                    raise ValueError('Can not happen')

                free_gains[i['docid']] = {rel: self.__gains[rank][rel] for rel in self.__labels}
            else:
                dcg += self.__gains[rank][i['rel']]

        return dcg, free_gains

    def __ranking_with_labels(self, run_df, qrels_df):
        top_x = run_df.groupby("query")[["query", "docid", "score"]].head(self.__depth)

        # Make sure that rank position starts by 1
        top_x["rank"] = 1
        top_x["rank"] = top_x.groupby("query")["rank"].cumsum()

        qrels_df = qrels_df.copy()
        qrels_df['rel'] = qrels_df['rel'].apply(lambda i: max(0, i))
        relevant_docs = qrels_df[qrels_df.rel >= 0]
        selection = pd.merge(top_x, relevant_docs[["query", "docid", "rel"]], how="left")

        if len(selection['query'].unique()) > 1:
            raise ValueError('I expect exactly one query')

        return selection

    def __perfect_ranking(self, qrels_df):
        relevant_docs = qrels_df[qrels_df.rel > 0]

        if len(relevant_docs['query'].unique()) != 1:
            raise ValueError('I expect the qrels per query!')

        ret = relevant_docs.sort_values(["query", "rel"], ascending=[True, False]).reset_index(drop=True)
        ret = ret.groupby("query").head(self.__depth)

        ret["rank"] = 1
        ret["rank"] = ret.groupby("query")["rank"].cumsum()

        return ret


class ImmediateNdcgResultPerQuery:
    def __init__(self, idcg, dcg_incomplete, free_gains, query):
        self.__idcg = idcg
        self.__dcg_incomplete = dcg_incomplete
        self.__free_gains = free_gains
        self.query = query
        self.__qrels_for_topic = None

    def set_doc_to_qrel(self, qrels_for_topic):
        if self.__qrels_for_topic:
            raise ValueError('Can not be reused')

        self.__qrels_for_topic = qrels_for_topic

    def calculate(self, doc_rels_or_topic_str, qrels_for_topic=None):
        if type(doc_rels_or_topic_str) == str:
            return self.calculate({k: qrels_for_topic[v] for k, v in json.loads(doc_rels_or_topic_str).items()})
        elif (doc_rels_or_topic_str and type(doc_rels_or_topic_str) != dict) or qrels_for_topic is not None:
            raise ValueError(f'Invalid input type(doc_rels_or_topic_str)={type(doc_rels_or_topic_str)} ' +
                             f'and qrels_for_topic = {qrels_for_topic}')

        doc_rels = doc_rels_or_topic_str
        dcg = self.__dcg_incomplete

        for unjudged_document in self.__free_gains:
            rel_of_doc = doc_rels[unjudged_document]
            dcg += self.__free_gains[unjudged_document][rel_of_doc]

        return dcg / self.__idcg

    def __getitem__(self, i):
        return self.calculate(i, self.__qrels_for_topic)
