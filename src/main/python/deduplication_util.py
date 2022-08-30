import json
from trectools import TrecQrel, TrecRun
import pandas as pd
from run_file_processing import normalize_run


class CanonicalGroupDeduplicationDeduplication:
    def __init__(self, canonical_group_files):
        translation_map = {}

        for canonical_group_file in canonical_group_files:
            for k, v in self.__all_translations_in_file(canonical_group_file):
                if not k or not v or k == '' or v == '' or k in translation_map:
                    raise ValueError(f'Error in the data: Got k={k}, v={v}')
                translation_map[k] = v

        self.__translation_map = translation_map

    @staticmethod
    def __all_translations_in_file(canonical_group_file):
        lines = [json.loads(i) for i in open(canonical_group_file, 'r')]

        for line in lines:
            ids = sorted([i['id'] for i in line['ids']])
            canonical_id = ids[0]

            for doc_id in ids:
                yield (doc_id, canonical_id)

    def document_id_to_canonical_id(self, doc_id):
        ret = self.__translation_map.get(doc_id, doc_id)
        if not ret:
            raise ValueError('Error in the data')
        return ret


class QrelDeduplication(CanonicalGroupDeduplicationDeduplication):
    def deduplicated(self, qrels):
        topic_to_doc_to_max_score = {}

        for _, i in qrels.qrels_data.iterrows():
            doc_id = self.document_id_to_canonical_id(i['docid'])

            if i['query'] not in topic_to_doc_to_max_score:
                topic_to_doc_to_max_score[i['query']] = {}

            rels = [i['rel']]

            if doc_id in topic_to_doc_to_max_score[i['query']]:
                rels += [topic_to_doc_to_max_score[i['query']][doc_id]]

            topic_to_doc_to_max_score[i['query']][doc_id] = max(rels)

        ret_df = []

        for query, rels in topic_to_doc_to_max_score.items():
            for doc_id, rel in rels.items():
                ret_df += [{'query': query, 'q0': 0, 'docid': doc_id, 'rel': rel}]

        ret = TrecQrel()
        ret.qrels_data = pd.DataFrame(ret_df)

        return ret


class ClueWebQrelDeduplication(QrelDeduplication):
    def __init__(self):
        super().__init__(['src/main/resources/unprocessed/clueweb12-fingerprint-groups-canonical.jsonl',
                          'src/main/resources/unprocessed/clueweb09-fingerprint-groups-canonical.jsonl'])


class RunDeduplication(CanonicalGroupDeduplicationDeduplication):
    def deduplicated(self, run):
        ret_df = []
        run = normalize_run(run, depth=1000)
        qid_to_covered_doc_ids = {}

        for _, i in run.run_data.iterrows():
            if i['query'] not in qid_to_covered_doc_ids:
                qid_to_covered_doc_ids[i['query']] = set()

            doc_id = self.document_id_to_canonical_id(i['docid'])

            if doc_id in qid_to_covered_doc_ids[i['query']]:
                continue

            qid_to_covered_doc_ids[i['query']].add(doc_id)
            ret_df += [{'query': i['query'], 'q0': 'Q0', 'docid': doc_id,
                        'rank': len(qid_to_covered_doc_ids[i['query']]), 'score': i['score'], 'system': i['system']
                        }]

        ret = TrecRun
        ret.run_data = pd.DataFrame(ret_df)

        return ret


class ClueWebRunDeduplication(RunDeduplication):
    def __init__(self):
        super().__init__(['src/main/resources/unprocessed/clueweb12-fingerprint-groups-canonical.jsonl',
                          'src/main/resources/unprocessed/clueweb09-fingerprint-groups-canonical.jsonl'])
