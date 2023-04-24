from collections import Counter
from functools import partial
from inspect import getmembers, isclass, isroutine

import mne
from braindecode.datasets.base import BaseConcatDataset
from braindecode.datasets.xy import create_from_X_y
from numpy import unique
from sklearn.base import BaseEstimator, TransformerMixin
from skorch.callbacks import Callback
from torch.nn import Module


# check if the data format is numpy or mne epoch
def _check_data_format(X):
    """
    Check if the data format is compatible with braindecode.
    Expect values in the format of MNE objects.
    Parameters
    ----------
    X: BaseConcatDataset

    Returns
    -------

    """
    if not isinstance(X, mne.EpochsArray):
        raise ValueError(
            "The data format is not supported. "
            "Please use the option return_epochs=True"
            "inside the Evaluations module."
        )


class BraindecodeDatasetLoader(BaseEstimator, TransformerMixin):
    """
    Class to Load the data from MOABB in a format compatible with braindecode
    """

    def __init__(self, drop_last_window=False, kw_args=None):
        self.drop_last_window = drop_last_window
        self.kw_args = kw_args

    def fit(self, X, y=None):
        _check_data_format(X)
        self.y = y
        return self

    def transform(self, X, y=None):
        _check_data_format(X)
        dataset = create_from_X_y(
            X=X.get_data(),
            y=self.y,
            window_size_samples=X.get_data().shape[2],
            window_stride_samples=X.get_data().shape[2],
            drop_last_window=self.drop_last_window,
            ch_names=X.info["ch_names"],
            sfreq=X.info["sfreq"],
        )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


def get_shape_from_baseconcat(X, param_name):
    """Get the shape of the data after BaseConcatDataset is applied"""
    if isinstance(X, BaseConcatDataset):
        in_channel = X[0][0].shape[0]
        input_window_samples = X[0][0].shape[1]
        return {param_name[0]: in_channel, param_name[1]: input_window_samples}
    else:
        return X.shape


def _find_model_from_braindecode(model_name):
    # soft dependency on braindecode
    model_list = []
    import braindecode.models as models

    for ds in getmembers(models, isclass):
        if issubclass(ds[1], Module):
            model_list.append(ds[1])

    for model in model_list:
        if model_name == model.__name__:
            # return an instance of the found model not initialized
            if model_name == "ShallowFBCSPNet":
                model = partial(model, final_conv_length="auto")

            return model
    raise ValueError(f"{model_name} not found in braindecode models")


class InputShapeSetterEEG(Callback):
    """Sets the input dimension of the PyTorch module to the input dimension
    of the training data.
    This can be of use when the shape of X is not known beforehand,
    e.g. when using a skorch model within an sklearn pipeline and
    grid-searching feature transformers, or using feature selection
    methods.InputShapeSetterEEG
    Basic usage:

    Parameters
    ----------
    params_list : list
      The list of parameters that define the
      input dimension in its ``__init__`` method.
      Usually the mandatory parameters from the model.
    input_dim_fn : callable, None (default=None)
      In case your ``X`` value is more complex and deriving the input
      dimension is not as easy as ``X.shape[-1]`` you can pass a callable
      to this parameter which takes ``X`` and returns the input dimension.
    module_name : str (default='module')
      Only needs change when you are using more than one module in your
      skorch model (e.g., in case of GANs).
    """

    def __init__(
        self,
        params_list=None,
        input_dim_fn=get_shape_from_baseconcat,
        module_name="module",
    ):
        self.module_name = module_name
        self.params_list = params_list
        self.input_dim_fn = input_dim_fn

    def get_input_dim(self, X):
        if self.input_dim_fn is not None:
            return self.input_dim_fn(X, self.params_list)
        if len(X.shape) < 2:
            raise ValueError(
                "Expected at least two-dimensional input data for X. "
                "If your data is one-dimensional, please use the "
                "`input_dim_fn` parameter to infer the correct "
                "input shape."
            )
        return X.shape[-1]

    def on_train_begin(self, net, X, y, **kwargs):
        # Get the parameters of the neural network
        params = net.get_params()
        # Get the input dimensions from the BaseConcat dataset
        params_get_from_dataset: dict = self.get_input_dim(X)
        # Get the number of classes in the output labels
        params_get_from_dataset["n_classes"] = len(unique(y))

        # Get all the parameters of the neural network module
        all_params_module = getmembers(params["module"], lambda x: not (isroutine(x)))
        # Filter the parameters to only include the selected ones
        selected_params_module = {
            sub[0]: sub[1] for sub in all_params_module if sub[0] in self.params_list
        }

        # Check if the selected parameters are inside the model parameters
        if Counter(params_get_from_dataset.keys()) != Counter(
            selected_params_module.keys()
        ):
            raise ValueError("Set the correct input name for the model from BrainDecode.")
        else:
            # Find the new module based on the current module's class name
            new_module = _find_model_from_braindecode(net.module.__class__.__name__)
            # Initialize the new module with the dataset parameters
            module_initilized = new_module(**params_get_from_dataset)
            # Set the neural network module to the new initialized module
            net.set_params(module=module_initilized)
            # Initialize the new module
            net.initialize_module()
