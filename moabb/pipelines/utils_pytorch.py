import inspect

from braindecode.datasets import BaseConcatDataset, create_from_X_y
from numpy import unique
from sklearn.base import BaseEstimator, TransformerMixin
from skorch.callbacks import Callback
from torch.nn import Module


class Transformer(BaseEstimator, TransformerMixin):
    """
    Class to Load the data from MOABB in a format compatible with braindecode
    """

    def __init__(self, kw_args=None):
        self.kw_args = kw_args

    def fit(self, X, y=None):
        self.y = y
        return self

    def transform(self, X, y=None):
        dataset = create_from_X_y(
            X.get_data(),
            y=self.y,
            window_size_samples=X.get_data().shape[2],
            window_stride_samples=X.get_data().shape[2],
            drop_last_window=False,
            sfreq=X.info["sfreq"],
        )

        return dataset

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True


def get_shape_from_baseconcat(X):
    """Get the shape of the data after BaseConcatDataset is applied"""
    if isinstance(X, BaseConcatDataset):
        in_channel = X[0][0].shape[0]
        input_window_samples = X[0][0].shape[1]
        return in_channel, input_window_samples
    else:
        return X.shape


def _find_model_from_braindecode(model_name):
    # soft dependency on braindecode
    model_list = []
    import braindecode.models as models

    for ds in inspect.getmembers(models, inspect.isclass):
        if issubclass(ds[1], Module):
            model_list.append(ds[1])

    for model in model_list:
        if model_name == model.__name__:
            # return an instance of the found model not initialized
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
        input_dim_fn=None,
        module_name="module",
    ):
        self.module_name = module_name
        self.params_list = params_list
        self.input_dim_fn = input_dim_fn

    def get_input_dim(self, X):
        if self.input_dim_fn is not None:
            return self.input_dim_fn(X)
        if len(X.shape) < 2:
            raise ValueError(
                "Expected at least two-dimensional input data for X. "
                "If your data is one-dimensional, please use the "
                "`input_dim_fn` parameter to infer the correct "
                "input shape."
            )
        return X.shape[-1]

    def on_train_begin(self, net, X, y, **kwargs):
        params = net.get_params()
        in_chans, input_window_samples = self.get_input_dim(X)
        n_classes = len(unique(y))

        if (
            params["module"].in_chans
            == in_chans & params["module"].input_window_samples
            == input_window_samples & params["module"].n_classes
            == n_classes
        ):
            return
        kwargs = {
            "input_window_samples": input_window_samples,
            "n_classes": n_classes,
            "in_chans": in_chans,
        }

        new_module = _find_model_from_braindecode(net.module.__class__.__name__)
        module_initilized = new_module(**kwargs)
        net.set_params(**{"module": module_initilized})
        net.initialize_module()
