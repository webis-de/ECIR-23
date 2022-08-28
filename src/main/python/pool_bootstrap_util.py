from bootstrap_util import BootstrappingStrategey


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


class ProbabilityEstimatedBootstrappingStrategey(PoolAndRunIndependentBootstrappingStrategey):
    def __init__(self, qrels, probability_estimator):
        super().__init__(qrels)
        self.__probability_estimator = probability_estimator

    def join_run_and_qrels(self, run):
        topic_id = run['query'].unique()
        if len(topic_id) > 1:
            raise ValueError('Ambiguous topics')
        elif len(topic_id) == 0:
            return []

        topic_id = topic_id[0]
        unjudged = self.unjudged_documents(run)
        qrels = self.available_qrels_for_topic(run, topic_id)
        probabilities = self.__probability_estimator.estimate_probabilities(run, self.qrels)

        return {
            'qrels': {k: v[:len(unjudged)] for k, v in qrels.items()},
            'probabilities': {k: probabilities[k] for k in sorted(probabilities.keys(), reverse=True)}
        }

    def single_bootstrap(self, qrels_and_probabilities, unjudged_docs, rand):
        ret = {}

        for doc in sorted(unjudged_docs):
            ret[doc] = self.select_next_random_doc(ret.values(), rand, qrels_and_probabilities)

        return ret

    @staticmethod
    def select_next_random_doc(already_selected, rand, qrels_and_probabilities):
        random_number = rand.random()
        last_qrels = 10000
        cumulated_probability = 1.0

        for qrel, probability in qrels_and_probabilities['probabilities'].items():
            if qrel > last_qrels:
                raise ValueError('TBD.')
            cumulated_probability -= probability

            if random_number >= cumulated_probability:
                if qrel not in qrels_and_probabilities['qrels']:
                    continue
                for i in qrels_and_probabilities['qrels'][qrel]:
                    if i not in already_selected:
                        return i

        raise ValueError('I cant find a suitable candidate for probabilities ' +
                         f'{qrels_and_probabilities["probabilities"]} on probability {random_number}.')