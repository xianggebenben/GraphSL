import torch.nn as nn
class Metric(nn.Module):


    def __init__(self,acc,pr,re,fs,auc):

        super().__init__()
        self.acc=acc
        self.pr=pr
        self.re=re
        self.fs=fs
        self.auc=auc





