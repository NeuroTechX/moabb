import numpy as np
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet
import torch.nn as nn
import torch.nn.functional as F
import torch
from skorch.dataset import ValidSplit
from skorch.classifier import NeuralNetClassifier

class BrainDecodeShallowConvNet(EEGClassifier):
    def __init__(
            self,
            module = ShallowFBCSPNet(
                in_chans=30,
                n_classes=2,
                input_window_samples=300,
                final_conv_length="auto",
            ),
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            batch_size=64,
            max_epochs=2,
            train_split=ValidSplit(0.2),
            device="cpu",
            verbose=1,
            **kwargs
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.train_split = train_split
        self.device = device
        self.verbose = verbose
        super().__init__(module=module, **kwargs)


    def check_data(self, X, y):

        self.set_params(module__in_chans=X.shape[1])
        self.set_params(module__n_classes=len(np.unique(y)))
        self.set_params(module__input_window_samples=X.shape[2])
        self.initialize()
        super().check_data(X, y)


class BrainDecodeShallowConvNet2(NeuralNetClassifier):
    def __init__(
            self,
            *args,
            custom=ShallowFBCSPNet,
            custom__in_chans=30,
            custom__n_classes=2,
            custom__input_window_samples=300,
            custom__final_conv_length="auto",
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            batch_size=64,
            max_epochs=2,
            train_split=ValidSplit(0.2),
            device="cpu",
            verbose=1,
            **kwargs
    ):
        self.custom = custom
        super().__init__(*args, **kwargs)
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.train_split = train_split
        self.device = device
        self.verbose = verbose
        self.custom__in_chans = custom__in_chans
        self.custom__n_classes = custom__n_classes
        self.custom__input_window_samples = custom__input_window_samples
        self.custom__final_conv_length = custom__final_conv_length

    def initialize_module(self, *args, **kwargs):

        params = self.get_params_for('custom')
        print(params)
        self.custom_ = self.custom(**params)

        return super().initialize_module()


    def check_data(self, X, y):
        super().check_data(X, y)

        self.set_params(module__in_chans=X.shape[1])
        self.set_params(module__n_classes=len(np.unique(y)))
        self.set_params(module__input_window_samples=X.shape[2])
        self.initialize()



class BrainDecodeShallowConvNet3(EEGClassifier):
    def __init__(
            self,
            module = ShallowFBCSPNet(
                in_chans=30,
                n_classes=2,
                input_window_samples=300,
                final_conv_length="auto",
            ),
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            batch_size=64,
            max_epochs=2,
            train_split=ValidSplit(0.2),
            device="cpu",
            verbose=1,
            **kwargs
    ):
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.train_split = train_split
        self.device = device
        self.verbose = verbose
        super().__init__(module=module, **kwargs)


import numpy as np
from functools import partial
from sklearn.base import BaseEstimator, ClassifierMixin
from braindecode import EEGClassifier
from braindecode.models import ShallowFBCSPNet

class braindecodeObject(BaseEstimator, ClassifierMixin):

    def __init__(self, fun_model, criterion, optimizer,
                 train_split, optimizer__lr
                 , batch_size, max_epochs, verbose,
                 callbacks, device):
        self.clf = None
        self.fun_model = fun_model
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.optimizer__lr = optimizer__lr
        self.train_split = train_split
        self.device = device
        self.verbose = verbose
        self.callbacks = callbacks

    def build_torch_model(self, fun_module, in_chans, n_classes,
                          input_window_samples, **kargs):
        self.module = fun_module(in_chans=in_chans, n_classes=n_classes,
                                 input_window_samples=input_window_samples, **kargs)
        return self

    def _build_EEGClassifier(self):
        return EEGClassifier(module=self.module,
                             criterion=self.criterion,
                             optimizer=self.optimizer,
                             train_split=self.train_split,  # using valid_set for validation
                             optimizer__lr=self.optimizer__lr,
                             batch_size=self.batch_size,
                             callbacks=self.callbacks,
                             device=self.device, )

    def fit(self, X, y):
        in_chans = X.shape[1]
        n_classes = len(np.unique(y))
        input_window_samples = X.shape[2]

        _ = self._build_torch_model(self.fun_model,
                                    in_chans=in_chans,
                                    n_classes=n_classes,
                                    input_window_samples=input_window_samples)

        self.clf = self._build_EEGClassifier()

        self.clf.initialize()

        return self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)
