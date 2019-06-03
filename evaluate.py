from load import Document


def get_intersection_size(cluster1, cluster2):
    cluster1 = set(tuple(element) for element in cluster1)
    cluster2 = set(tuple(element) for element in cluster2)
    return len(cluster1.intersection(cluster2))


def evaluate_b3(doc, response):
    assert isinstance(doc, Document)
    numerator_sum_recall = 0
    denominator_sum_recall = 0
    numerator_sum_precision = 0
    denominator_sum_precision = sum(len(cluster2) for cluster2 in response)
    for cluster1 in doc.real_coreferences:
        for cluster2 in response:
            intersection_size = get_intersection_size(cluster1, cluster2)
            numerator_sum_recall += (intersection_size ** 2)/len(cluster1)
            numerator_sum_precision += (intersection_size ** 2)/len(cluster2)
        denominator_sum_recall += len(cluster1)

    recall = float(numerator_sum_recall)/float(denominator_sum_recall)
    precision = float(numerator_sum_precision)/float(denominator_sum_precision)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def get_avg_metrics(data, method, config, limit=100):
    precisions = []
    recalls = []
    f1s = []
    for doc in data[0:limit]:
        precision, recall, f1 = evaluate_b3(doc, method(doc, **config))
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    precision_avg = sum(precisions) / len(precisions)
    recall_avg = sum(recalls) / len(recalls)
    f1s_avg = sum(f1s) / len(f1s)
    return precision_avg, recall_avg, f1s_avg