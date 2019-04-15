import time
import numpy as np
import torch
from torch import nn
from abc import abstractmethod, ABCMeta


class DetectNet(nn.Module, metaclass=ABCMeta):
    """Base class for Convolutional Neural Networks for detection."""

    @abstractmethod
    def _prepare_module(self, **kwargs):
        """
        Prepare model.
        This should be implemented.
        """
        pass

    def predict(self, dataset, **kwargs):

        batch_size = kwargs.pop('batch_size', 16)
        pred_size = dataset.num_examples
        num_steps = pred_size // batch_size
        flag = int(bool(pred_size % batch_size))
        # Start prediction loop
        _y_pred = []
        self.eval()
        for i in range(num_steps + flag):
            if i == num_steps and flag:
                _batch_size = pred_size - num_steps * batch_size
            else:
                _batch_size = batch_size
            X_true, _ = dataset.next_batch(_batch_size, shuffle=False)
            X_true = torch.tensor(X_true).cuda()
            # Compute predictions
            y_pred = self.output(X_true).cpu().detach().numpy()

            _y_pred.append(y_pred)
        _y_pred = np.concatenate(_y_pred, axis=0)
        return _y_pred

    def _init_normal(self, m):
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)

    def save(self, model_path):
        torch.save({'model_state_dict': self.state_dict()}, model_path)

    def restore(self, model_path):
        ckpt = torch.load(model_path)
        self.load_state_dict(ckpt['model_state_dict'])
