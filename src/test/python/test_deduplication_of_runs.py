from unittest import TestCase
from error_analysis_approaches import create_run, rl
from deduplication_util import ClueWebRunDeduplication


class TestDeduplicationOfRuns(TestCase):
    def test_deduplicate_qrels_without_duplicates(self):
        run = create_run([
            rl('1-1', 1), rl('r-2', 2),
            rl('nr-1', 3), rl('nr-2', 4),
        ])

        expected_run = create_run([
            rl('1-1', 1), rl('r-2', 2),
            rl('nr-1', 3), rl('nr-2', 4),
        ])

        actual = ClueWebRunDeduplication().deduplicated(run)

        self.assertEquals(actual.run_data.to_json(), expected_run.run_data.to_json())

    def test_deduplicate_qrels_with_duplicates_order_01(self):
        run = create_run([
            rl('1-1', 1), rl('r-2', 2),
            rl('clueweb12-0808wb-28-15054', 3), rl('clueweb12-0808wb-28-15057', 4),
            rl('nr-1', 5), rl('clueweb12-0809wb-32-13618', 6), rl('nr-2', 7),
        ])

        expected_run = create_run([
            rl('1-1', 1), rl('r-2', 2),
            rl('clueweb12-0808wb-19-10672', 3),
            rl('nr-1', 4, 3000-5), rl('nr-2', 5, 3000-7),
        ])

        actual = ClueWebRunDeduplication().deduplicated(run)

        self.assertEquals(actual.run_data.to_json(), expected_run.run_data.to_json())

    def test_deduplicate_qrels_with_duplicates_order_02(self):
        run = create_run([
            rl('nr-2', 7),
            rl('clueweb12-0808wb-28-15054', 3), rl('clueweb12-0808wb-28-15057', 4),
            rl('1-1', 1), rl('r-2', 2),
            rl('nr-1', 5), rl('clueweb12-0809wb-32-13618', 6),
        ])

        expected_run = create_run([
            rl('1-1', 1), rl('r-2', 2),
            rl('clueweb12-0808wb-19-10672', 3),
            rl('nr-1', 4, 3000-5), rl('nr-2', 5, 3000-7),
        ])

        actual = ClueWebRunDeduplication().deduplicated(run)

        self.assertEquals(actual.run_data.to_json(), expected_run.run_data.to_json())

    def test_deduplicate_qrels_with_duplicates_order_03(self):
        run = create_run([
            rl('1-1', 1), rl('r-2', 2),
            rl('clueweb12-0809wb-32-13618', 3), rl('clueweb12-0808wb-28-15054', 4),
            rl('nr-1', 5), rl('clueweb12-0808wb-28-15057', 6), rl('nr-2', 7),
        ])

        expected_run = create_run([
            rl('1-1', 1), rl('r-2', 2),
            rl('clueweb12-0808wb-19-10672', 3),
            rl('nr-1', 4, 3000-5), rl('nr-2', 5, 3000-7),
        ])

        actual = ClueWebRunDeduplication().deduplicated(run)

        self.assertEquals(actual.run_data.to_json(), expected_run.run_data.to_json())
