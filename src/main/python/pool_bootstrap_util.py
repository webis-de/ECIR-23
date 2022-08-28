from bootstrap_util import BootstrappingStrategey
import json


class PoolAndRunIndependentBootstrappingStrategey(BootstrappingStrategey):
    def join_run_and_qrels(self, run):
        topic_id = run['query'].unique()
        if len(topic_id) > 1:
            raise ValueError('Ambiguous topics')
        elif len(topic_id) == 0:
            return []

        topic_id = topic_id[0]
        unjudged = self.unjudged_documents(run)
        qrels = self.available_qrels_for_topic(run, topic_id)
        qrels = {k: v[:len(unjudged)] for k, v in qrels.items()}

        ret = []
        for v in qrels.values():
            for i in v:
                ret += [v]

        return sorted(ret)

    def single_bootstrap(self, qrels_and_run, unjudged_docs, rand):
        ret = {}
        representative_documents = rand.sample(qrels_and_run, len(unjudged_docs))

        for doc in sorted(unjudged_docs):
            for i in representative_documents.pop():
                if i not in ret.values():
                    ret[doc] = i
                    break

        return ret

    def available_qrels_for_topic(self, run_for_topic, topic_id):
        qrels_for_topic = self.qrels[self.qrels['query'] == topic_id]
        qrels_for_topic = qrels_for_topic[~qrels_for_topic['docid'].isin(run_for_topic['docid'].unique())]
        qrels_for_topic = qrels_for_topic.groupby('rel').aggregate({'docid': lambda x: sorted(list(x))}).reset_index()
        qrels = {i['rel']: i['docid'] for _, i in qrels_for_topic.iterrows()}

        return {k: qrels[k] for k in sorted(qrels.keys())}
