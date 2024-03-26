import torch.nn as nn


class Metric:

    def __init__(self, acc, pr, re, f1, auc):
        """
        Constructor method to initialize the Metric object.

        Args:
            acc (float): Accuracy metric value.
            pr (float): Precision metric value.
            re (float): Recall metric value.
            f1 (float): F1-score metric value.
            auc (float): Area Under the Curve metric value.
        """
        self.acc = acc
        self.pr = pr
        self.re = re
        self.f1 = f1
        self.auc = auc





