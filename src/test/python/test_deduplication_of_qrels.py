from unittest import TestCase
from error_analysis_approaches import create_qrels, ql
from deduplication_util import ClueWebQrelDeduplication


class TestDeduplicationOfQrels(TestCase):
    def test_deduplicate_qrels_without_duplicates(self):
        qrels = create_qrels([
            ql('1-1', 1), ql('r-2', 1),
            ql('nr-1', 0), ql('nr-2', 0),
        ])

        expected_qrels = create_qrels([
            ql('1-1', 1), ql('r-2', 1),
            ql('nr-1', 0), ql('nr-2', 0),
        ])

        actual = ClueWebQrelDeduplication().deduplicated(qrels)

        self.assertEquals(actual.qrels_data.to_json(), expected_qrels.qrels_data.to_json())

    def test_deduplicate_qrels_with_duplicates_order_01(self):
        qrels = create_qrels([
            ql('1-1', 1), ql('r-2', 1),
            ql('nr-1', 0), ql('nr-2', 0),
            ql('clueweb12-0808wb-19-10672', 0),
            ql('clueweb12-0808wb-28-15054', 0),
            ql('clueweb12-0808wb-28-15057', 1),
            ql('clueweb12-0809wb-32-13618', 0),
        ])

        expected_qrels = create_qrels([
            ql('1-1', 1), ql('r-2', 1),
            ql('nr-1', 0), ql('nr-2', 0),
            ql('clueweb12-0808wb-19-10672', 1),
        ])

        actual = ClueWebQrelDeduplication().deduplicated(qrels)

        print(actual.qrels_data.to_json())
        self.assertEquals(actual.qrels_data.to_json(), expected_qrels.qrels_data.to_json())

    def test_deduplicate_qrels_with_duplicates_order_02(self):
        qrels = create_qrels([
            ql('1-1', 1), ql('r-2', 1),
            ql('nr-1', 0), ql('nr-2', 0),
            ql('clueweb12-0808wb-19-10672', 2),
            ql('clueweb12-0808wb-28-15054', 0),
            ql('clueweb12-0808wb-28-15057', 0),
            ql('clueweb12-0809wb-32-13618', 0),
        ])

        expected_qrels = create_qrels([
            ql('1-1', 1), ql('r-2', 1),
            ql('nr-1', 0), ql('nr-2', 0),
            ql('clueweb12-0808wb-19-10672', 2),
        ])

        actual = ClueWebQrelDeduplication().deduplicated(qrels)

        print(actual.qrels_data.to_json())
        self.assertEquals(actual.qrels_data.to_json(), expected_qrels.qrels_data.to_json())
