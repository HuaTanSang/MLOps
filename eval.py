from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
def evaluate(
    y_true, y_pred, prefix=''
) :
    """
    Get the classification metrics.

    :param y_true: The true target values.
    :param y_pred: The predicted target values.
    :param prefix: The prefix of the metric names.
    :return: The classification metrics.
    """

    return {
        f"{prefix}_accuracy": accuracy_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}_precision": precision_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}_recall": recall_score(y_true=y_true, y_pred=y_pred),
        f"{prefix}_f1": f1_score(y_true=y_true, y_pred=y_pred),
    }
