from __future__ import annotations

from pathlib import Path
from pickle import HIGHEST_PROTOCOL, dump
from typing import Sequence

from numpy import argmax
from sklearn.pipeline import Pipeline


try:
    from optuna.distributions import CategoricalDistribution

    optuna_available = True
except ImportError:
    optuna_available = False


def _check_if_is_keras_model(model):
    """Check if the model is a Keras model.

    Parameters
    ----------
    model: object
        Model to check
    Returns
    -------
    is_keras_model: bool
        True if the model is a Keras model
    """
    try:
        from scikeras.wrappers import KerasClassifier

        is_keras_model = isinstance(model, KerasClassifier)
        return is_keras_model
    except ImportError:
        return False


def _check_if_is_pytorch_model(model):
    """Check if the model is a Keras model.

    Parameters
    ----------
    model: object
        Model to check
    Returns
    -------
    is_keras_model: bool
        True if the model is a Keras model
    """
    try:
        from skorch import NeuralNetClassifier

        is_pytorch_model = isinstance(model, NeuralNetClassifier)
        return is_pytorch_model
    except ImportError:
        return False


def _check_if_is_pytorch_steps(model):
    skorch_valid = False
    try:
        skorch_valid = any(
            _check_if_is_pytorch_model(j) for j in model.named_steps.values()
        )
        return skorch_valid
    except Exception:
        return skorch_valid


def _check_if_is_keras_steps(model):
    keras_valid = False
    try:
        keras_valid = any(_check_if_is_keras_model(j) for j in model.named_steps.values())
        return keras_valid
    except Exception:
        return keras_valid


def save_model_cv(model: object, save_path: str | Path, cv_index: str | int):
    """Save a model fitted to a given fold from cross-validation.

    Parameters
    ----------
    model: object
        Model (pipeline) fitted
    save_path: str
        Path to save the model, will create if it does not exist
        based on the parameter hdf5_path from the evaluation object.
    cv_index: str
        Index of the cross-validation fold used to fit the model
        or 'best' if the model is the best fitted

    Returns
    -------
    """
    if save_path is None:
        raise IOError("No path to save the model")
    else:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    if _check_if_is_pytorch_steps(model):
        for step_name in model.named_steps:
            step = model.named_steps[step_name]
            file_step = f"{step_name}_fitted_{cv_index}"

            if _check_if_is_pytorch_model(step):
                step.save_params(
                    f_params=Path(save_path) / f"{file_step}_model.pkl",
                    f_optimizer=Path(save_path) / f"{file_step}_optim.pkl",
                    f_history=Path(save_path) / f"{file_step}_history.json",
                    f_criterion=Path(save_path) / f"{file_step}_criterion.pkl",
                )
            else:
                with open((Path(save_path) / f"{file_step}.pkl"), "wb") as file:
                    dump(step, file, protocol=HIGHEST_PROTOCOL)

    elif _check_if_is_keras_steps(model):
        for step_name in model.named_steps:
            file_step = f"{step_name}_fitted_model_{cv_index}"
            step = model.named_steps[step_name]
            if _check_if_is_keras_model(step):
                step.model_.save(Path(save_path) / f"{file_step}.h5")
            else:
                with open((Path(save_path) / f"{file_step}.pkl"), "wb") as file:
                    dump(step, file, protocol=HIGHEST_PROTOCOL)
    else:
        with open((Path(save_path) / f"fitted_model_{cv_index}.pkl"), "wb") as file:
            dump(model, file, protocol=HIGHEST_PROTOCOL)


def save_model_list(model_list: list | Pipeline, score_list: Sequence, save_path: str):
    """Save a list of models fitted to a folder.

    Parameters
    ----------
    model_list: list | Pipeline
        List of models or model (pipelines) fitted
    score_list: Sequence
        List of scores for each model in model_list
    save_path: str
        Path to save the models, will create if it does not exist
        based on the parameter hdf5_path from the evaluation object.
    Returns
    -------
    """
    if model_list is None:
        return

    Path(save_path).mkdir(parents=True, exist_ok=True)

    if not isinstance(model_list, list):
        model_list = [model_list]

    for cv_index, model in enumerate(model_list):
        save_model_cv(model, save_path, str(cv_index))

    best_model = model_list[argmax(score_list)]

    save_model_cv(best_model, save_path, "best")


def create_save_path(
    hdf5_path,
    code: str,
    subject: int | str,
    session: str,
    name: str,
    grid=False,
    eval_type="WithinSession",
):
    """Create a save path based on evaluation parameters.

    Parameters
    ----------
    hdf5_path : str
       The base path where the models will be saved.
    code : str
       The code for the evaluation.
    subject : int
       The subject ID for the evaluation.
    session : str
       The session ID for the evaluation.
    name : str
       The name for the evaluation.
    grid : bool, optional
       Whether the evaluation is a grid search or not. Defaults to False.
    eval_type : str, optional
       The type of evaluation, either 'WithinSession', 'CrossSession' or 'CrossSubject'.
       Defaults to WithinSession.
    Returns
    -------
    path_save: str
       The created save path.
    """
    if hdf5_path is not None:
        if eval_type != "WithinSession":
            session = ""

        if grid:
            path_save = (
                Path(hdf5_path)
                / f"GridSearch_{eval_type}"
                / code
                / f"{str(subject)}"
                / str(session)
                / str(name)
            )
        else:
            path_save = (
                Path(hdf5_path)
                / f"Models_{eval_type}"
                / code
                / f"{str(subject)}"
                / str(session)
                / str(name)
            )

        return str(path_save)
    else:
        print("No hdf5_path provided, models will not be saved.")


def _convert_sklearn_params_to_optuna(param_grid: dict) -> dict:
    """
    Function to convert the parameter in Optuna format. This function will
    create a categorical distribution of values from the list of values
    provided in the parameter grid.

    Parameters
    ----------
    param_grid:
        Dictionary with the parameters to be converted.

    Returns
    -------
    optuna_params: dict
        Dictionary with the parameters converted to Optuna format.
    """
    if not optuna_available:
        raise ImportError(
            "Optuna is not available. Please install it optuna " "and optuna-integration."
        )
    else:
        optuna_params = {}
        for key, value in param_grid.items():
            try:
                if isinstance(value, list):
                    optuna_params[key] = CategoricalDistribution(value)
                else:
                    optuna_params[key] = value
            except Exception as e:
                raise ValueError(f"Conversion failed for parameter {key}: {e}")
        return optuna_params
