def binary_classification_metrics(prediction, ground_truth):
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp = np.sum(prediction[ground_truth == True] == True)
    fn = np.sum(prediction[ground_truth == True] == False)
    fp = np.sum(prediction[ground_truth == False] == True)
    tn = np.sum(prediction[ground_truth == False] == False)
    
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    accuracy = (tp + tn) / prediction.shape[0]
    f1 = 2 * precision * recall / (precision + recall)
    
    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    return sum(prediction == ground_truth) / prediction.shape[0]
