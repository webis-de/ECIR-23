import json
from trectools import TrecQrel
import pandas as pd


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

            for id in ids:
                yield (id, canonical_id)

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
