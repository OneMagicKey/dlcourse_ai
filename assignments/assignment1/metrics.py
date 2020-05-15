def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    tp,tn,fp,fn = 0, 0, 0, 0
    for i in range(prediction.shape[0]):
        if prediction[i] == 1 and ground_truth[i] == 1:
            tp += 1
        elif prediction[i] == 0 and ground_truth[i] == 1:
            fn += 1
        elif prediction[i] == 1 and ground_truth[i] == 0:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / prediction.shape[0]
    f1 = 2 * precision * recall / (precision + recall)
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    return sum(prediction == ground_truth) / prediction.shape[0]
