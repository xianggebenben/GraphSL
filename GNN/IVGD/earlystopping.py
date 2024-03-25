from typing import List
import copy
import operator
from enum import Enum, auto
import numpy as np
from torch.nn import Module


class StopVariable(Enum):
    """
    Enum class representing stopping criteria variables.
    """
    LOSS = auto()
    ACCURACY = auto()
    NONE = auto()


class Best(Enum):
    """
    Enum class representing best stopping criteria.
    """
    RANKED = auto()
    ALL = auto()


stopping_args = dict(
    stop_varnames=[StopVariable.LOSS],
    patience=30, max_epochs=1000, remember=Best.RANKED)


class EarlyStopping:
    """
    Class for implementing early stopping in model training.
    """

    def __init__(
            self, model: Module, stop_varnames: List[StopVariable],
            patience: int = 30, max_epochs: int = 1000, remember: Best = Best.ALL):
        """
        Initializes the EarlyStopping object.

        Args:
        - model (Module): The neural network model.
        - stop_varnames (List[StopVariable]): List of stopping criteria variables.
        - patience (int): Number of epochs to wait after reaching the best model before stopping.
        - max_epochs (int): Maximum number of epochs for training.
        - remember (Best): Specifies how to remember the best model.
        """
        self.model = model
        self.comp_ops = []
        self.stop_vars = []
        self.best_vals = []
        for stop_varname in stop_varnames:
            if stop_varname is StopVariable.LOSS:
                self.stop_vars.append('loss')
                self.comp_ops.append(operator.le)
                self.best_vals.append(np.inf)
            elif stop_varname is StopVariable.ACCURACY:
                self.stop_vars.append('acc')
                self.comp_ops.append(operator.ge)
                self.best_vals.append(-np.inf)
        self.remember = remember
        self.remembered_vals = copy.copy(self.best_vals)
        self.max_patience = patience
        self.patience = self.max_patience
        self.max_epochs = max_epochs
        self.best_epoch = None
        self.best_state = None

    def check(self, values: List[np.floating], epoch: int) -> bool:
        """
        Checks if early stopping criteria are met.

        Args:
        - values (List[np.floating]): List of evaluation metric values.
        - epoch (int): Current epoch number.

        Returns:
        - bool: True if early stopping criteria are met, False otherwise.
        """
        checks = [self.comp_ops[i](val, self.best_vals[i])
                  for i, val in enumerate(values)]
        if any(checks):
            self.best_vals = np.choose(checks, [self.best_vals, values])
            self.patience = self.max_patience

            comp_remembered = [
                    self.comp_ops[i](val, self.remembered_vals[i])
                    for i, val in enumerate(values)]
            if self.remember is Best.ALL:
                if all(comp_remembered):
                    self.best_epoch = epoch
                    self.remembered_vals = copy.copy(values)
                    self.best_state = {
                            key: value.cpu() for key, value
                            in self.model.state_dict().items()}
            elif self.remember is Best.RANKED:
                for i, comp in enumerate(comp_remembered):
                    if comp:
                        if not(self.remembered_vals[i] == values[i]):
                            self.best_epoch = epoch
                            self.remembered_vals = copy.copy(values)
                            self.best_state = {
                                    key: value.cpu() for key, value
                                    in self.model.state_dict().items()}
                            break
                    else:
                        break
        else:
            self.patience -= 1
        return self.patience == 0