def fill_confusion_matrix(pas_actual, pas_prediction, arg):
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for idx in range(len(pas_actual)):
        if pas_prediction[idx][1] == arg:
            if pas_actual[idx][1] == arg:
                tp += 1
            else:
                fp += 1
        else:
            if pas_actual[idx][1] == arg:
                fn += 1
            else:
                tn += 1

    precision, recall, exist = 0.0, 0.0, False
    if (tp + fp > 0 and tp + fn > 0):
        precision = float(tp)/float(tp + fp)
        recall = float(tp)/float(tp + fn)
        exist = True

    return precision, recall, exist

def calculate_pas_performance(pas_actual, pas_prediction):
    # subj, verb, obj, tmp, loc, adj
    # 0, 1, 2, 3, 4, 5
    precisions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    recalls = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    num_exists = [0, 0, 0, 0, 0, 0]
    args = ["SUBJ", "VERB", "OBJ", "TMP", "LOC", "ADJ"]
    # print(len(pas_actual))
    # print(len(pas_prediction))
    if len(pas_actual) == len(pas_prediction):
        for idx in range(len(pas_actual)):
            for idx_arg, arg in enumerate(args):
                precision, recall, exist = fill_confusion_matrix(pas_actual[idx], pas_prediction[idx], arg)
                if exist:
                    num_exists[idx_arg] += 1
                    precisions[idx_arg] += precision
                    recalls[idx_arg] += recall
    
        for idx in range(len(args)):
            precisions[idx] /= num_exists[idx]
            recalls[idx] /= num_exists[idx]

    return precisions, recalls
