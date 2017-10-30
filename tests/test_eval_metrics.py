import unittest
from citeomatic.eval_metrics import precision_recall_f1_at_ks, average_results


class TestEvalMetrics(unittest.TestCase):
    def test_precision_recall_f1_at_ks(self):
        gold_y = ['1', '2', '3']
        pred_y = ['1', '4', '3']
        scores_y = [1.0, 0.1, 0.5]
        k = [1, 2, 3]
        results = precision_recall_f1_at_ks(gold_y, pred_y, scores=None, k_list=k)

        assert results['precision'] == [1.0, 0.5, 2/3]
        assert results['recall'] == [1/3, 1/3, 2/3]
        assert results['f1'] == [1/2, 2/5, 2/3]
        assert results['mrr'] == 1.0

        results_2 = precision_recall_f1_at_ks(gold_y, pred_y, scores_y, k)

        assert results_2['precision'] == [1.0, 1.0, 2/3]
        assert results_2['recall'] == [1/3, 2/3, 2/3]
        assert results_2['f1'] == [1/2, 4/5, 2/3]
        assert results_2['mrr'] == 1.0

    def test_average_results(self):
        r1 = {
            'precision': [1.0, 0.5, 2/3],
            'recall': [1.0, 0.5, 2/3],
            'f1': [1.0, 0.5, 2/3],
            'mrr': 1.0,
        }

        r2 = {
            'precision': [3.0, 1.0, 4/3],
            'recall': [3.0, 1.0, 4/3],
            'f1': [3.0, 1.0, 4/3],
            'mrr': 0.5,
        }

        averaged_results = average_results([r1, r2])
        assert averaged_results['precision'] == [2.0, 0.75, 1.0]
        assert averaged_results['mrr'] == 0.75
        

if __name__ == '__main__':
    unittest.main()