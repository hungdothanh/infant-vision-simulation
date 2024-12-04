
from torchmetrics.functional import confusion_matrix


def precision_recall(predictions, labels, num_classes):
    conf_matrix = confusion_matrix(predictions, labels, task="binary", num_classes=num_classes)

    true_positives = conf_matrix.diag()
    total_predicted = conf_matrix.sum(dim=0)
    total_actual = conf_matrix.sum(dim=1)

    # Class-wise precision and recall
    precisions = (true_positives / total_predicted).nan_to_num(0)
    recalls = (true_positives / total_actual).nan_to_num(0)

    return precisions, recalls, conf_matrix
