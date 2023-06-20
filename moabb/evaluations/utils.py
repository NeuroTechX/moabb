from os import makedirs
from pathlib import Path
from typing import Sequence

from joblib import dump
from numpy import argmax


def save_model(model, save_path: str, cv_index: int):
    """
    Save a model fitted to a folder
    Parameters
    ----------
    model: object
        Model (pipeline) fitted
    save_path: str
        Path to save the model, will create if it does not exist
        based on the parameter hdf5_path from the evaluation object.
    cv_index: int
        Index of the cross-validation fold used to fit the model
    Returns
    -------
    filenames: list
        List of filenames where the model is saved
    """
    # Save the model
    makedirs(save_path, exist_ok=True)
    return dump(model, Path(save_path) / f"fitted_model_{cv_index}.pkl")


def save_model_list(model_list: list, score_list: Sequence, save_path: str):
    """
    Save a list of models fitted to a folder
    Parameters
    ----------
    model_list: list
        List of models (pipelines) fitted
    save_path: str
        Path to save the models, will create if it does not exist
        based on the parameter hdf5_path from the evaluation object.
    Returns
    -------
    """
    # Save the result
    makedirs(save_path, exist_ok=True)
    for i, model in enumerate(model_list):
        dump(
            model,
            Path(save_path) / f"fitted_model_cv_{str(i)}.pkl",
        )
    # Saving the best model
    best_model = model_list[argmax(score_list)]
    dump(
        best_model,
        Path(save_path) / "best_model.pkl",
    )


def create_save_path(
    hdf5_path,
    code: str,
    subject: int,
    session: str,
    name: str,
    grid=False,
    eval_type="WithinSession",
):
    """
    Create a save path based on evaluation parameters.

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
