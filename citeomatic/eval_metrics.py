import numpy as np


def precision_recall_f1_at_ks(gold_y, predictions, scores=None, k_list=None):

    def _mrr(ranked_list):
        try:
            idx = ranked_list.index(True)
            return 1. / (idx + 1)
        except ValueError:
            return 0.0

    if k_list is None:
        k_list = [1, 5, 10]
    if scores is not None:
        sorted_predictions = [p for p, _ in
                              sorted(zip(predictions, scores), key=lambda x : x[1], reverse=True)]
    else:
        sorted_predictions = predictions

    gold_set = set(gold_y)

    sorted_correct = [y_pred in gold_set for y_pred in sorted_predictions]

    results = {
        'precision': [],
        'recall': [],
        'f1': [],
        'mrr': _mrr(sorted_correct),
        'k': k_list
    }
    num_gold = len(gold_y)

    for k in k_list:
        num_correct = np.sum(sorted_correct[:k])
        p = num_correct / k
        r = num_correct / num_gold
        if num_correct == 0:
            f = 0.0
        else:
            f = 2 * p * r / (p + r)
        results['precision'].append(p)
        results['recall'].append(r)
        results['f1'].append(f)

    return results


def average_results(results: list):
    p_matrix = []
    r_matrix = []
    f_matrix = []
    mrr_list = []

    for r in results:
        p_matrix.append(r['precision'])
        r_matrix.append(r['recall'])
        f_matrix.append(r['f1'])
        mrr_list.append(r['mrr'])

    return {
        'precision': list(np.mean(p_matrix, axis=0)),
        'recall': list(np.mean(r_matrix, axis=0)),
        'f1': list(np.mean(f_matrix, axis=0)),
        'mrr': np.mean(mrr_list),
    }


def f1(p, r):
    if p + r == 0.0:
        return 0.0
    else:
        return 2 * p * r / (p + r)
