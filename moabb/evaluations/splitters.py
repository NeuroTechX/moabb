import inspect
import logging

from sklearn.model_selection import (
    BaseCrossValidator,
    LeaveOneGroupOut,
    StratifiedKFold,
)
from sklearn.utils import check_random_state


log = logging.getLogger(__name__)


class WithinSessionSplitter(BaseCrossValidator):
    """Data splitter for within session evaluation.

    Within-session evaluation uses k-fold cross_validation to determine train
    and test sets for each subject in each session. This splitter
    assumes that all data from all subjects is already known and loaded.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/withinsess.png
        :alt: The schematic diagram of the WithinSession split
        :align: center

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses StratifiedKFold.

    Parameters
    ----------
    n_folds : int, default=5
        Number of folds. Must be at least 2. If
    shuffle : bool, default=True
        Whether to shuffle each class's samples before splitting into batches.
        Note that the samples within each split will not be shuffled.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of splits. Only used when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
    cv_class: cros-validation class, default=StratifiedKFold
        Inner cross-validation strategy for splitting the sessions.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    """

    def __init__(
        self,
        n_folds: int = 5,
        shuffle: bool = True,
        random_state: int = None,
        cv_class: type[BaseCrossValidator] = StratifiedKFold,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        self.random_state = random_state
        self._rng = check_random_state(random_state) if shuffle else None

        if not shuffle and random_state is not None:
            raise ValueError("random_state should be None when shuffle is False")

        # Create a dictionary of parameters by adding arguments only if they
        # are part of the inner cross-validation strategy's signature
        params = inspect.signature(self.cv_class).parameters
        for p, v in [
            ("n_splits", n_folds),
            ("shuffle", shuffle),
            ("random_state", self._rng),
        ]:
            if p in params:
                self._cv_kwargs[p] = v

    def get_n_splits(self, metadata):
        num_sessions_subjects = metadata.groupby(["subject", "session"]).ngroups
        return self.n_folds * num_sessions_subjects

    def split(self, y, metadata):
        all_index = metadata.index.values

        # Shuffle subjects if required
        subjects = metadata["subject"].unique()
        if self.shuffle:
            self._rng.shuffle(subjects)

        for subject in subjects:
            subject_mask = metadata["subject"] == subject
            subject_indices = all_index[subject_mask]
            subject_metadata = metadata[subject_mask]
            y_subject = y[subject_mask]

            # Shuffle sessions if required
            sessions = subject_metadata["session"].unique()

            if self.shuffle:
                self._rng.shuffle(sessions)

            for session in sessions:
                session_mask = subject_metadata["session"] == session
                indices = subject_indices[session_mask]
                y_session = y_subject[session_mask]

                # Instantiate a new internal splitter for each session
                splitter = self.cv_class(**self._cv_kwargs)

                # Split using the current instance of StratifiedKFold by default
                for train_ix, test_ix in splitter.split(indices, y_session):

                    yield indices[train_ix], indices[test_ix]


class CrossSessionSplitter(BaseCrossValidator):
    """Data splitter for cross session evaluation.

    This splitter enables cross-session evaluation by performing a Leave-One-Session-Out (LOSO)
    cross-validation on data from each subject.

    It assumes that the entire metainformation across all subjects is already loaded.

    Unlike the `CrossSessionEvaluation` class from `moabb.evaluation`, which manages
    the complete evaluation process end-to-end, this splitter is solely responsible
    for dividing the data into training and testing sets based on sessions.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/crosssess.jpg
        :alt: The schematic diagram of the CrossSession split
        :align: center

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses LeaveOneGroupOut,
    which performs Leave-One-Session-Out cross-validation.

    Parameters
    ----------
    cv_class: cross-validation class, default=LeaveOneGroupOut
        Inner cross-validation strategy for splitting the sessions of one subject.
        LeaveOneGroupOut is the most common default.
    shuffle: bool, default=False
        Whether to shuffle the session order for each subject. It can only be
        used when changing the `cv_class` to a class compatible with `shuffle`.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the inner cross-validation when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
        For `cv_class` accepting `random_state`, they are provided with a shared rng.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    Yields
    ------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """

    def __init__(
        self,
        cv_class: type[BaseCrossValidator] = LeaveOneGroupOut,
        shuffle: bool = False,
        random_state: int = None,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        self.shuffle = shuffle
        self.random_state = random_state

        params = inspect.signature(self.cv_class).parameters
        if shuffle and ("shuffle" not in params and "random_state" not in params):
            raise ValueError(
                f"Shuffling is not supported for {cv_class.__name__}. "
                "Choose a different `cv_class` or use `shuffle=False`."
                "Example of `cv_class`: `GroupShuffleSplit`: "
                "CrossSessionSplitter(shuffle=True, random_state=42, cv_class=GroupShuffleSplit)"
            )

        if not shuffle and "shuffle" in params and random_state is not None:
            raise ValueError(
                "`random_state` should be None when `shuffle` is False for {cv_class.__name__}"
            )

        self._need_rng = "random_state" in params and (shuffle or "shuffle" not in params)

        if "shuffle" in params:
            self._cv_kwargs["shuffle"] = shuffle

    def get_n_splits(self, metadata):
        """
        Return the number of splits for the cross-validation.

        The number of splits is the number of subjects times the number of splits
        of the inner cross-validation strategy.

        We try to keep the same behaviour as the sklearn cross-validation classes.

        Parameters
        ----------
        metadata: pd.DataFrame
            The metadata containing the subject and session information.

        Returns
        -------
        n_splits: int
            The number of splits for the cross-validation
        """
        subjects = metadata["subject"].unique()
        n_splits = 0
        for subject in subjects:
            subject_metadata = metadata.query("subject == @subject")
            sessions = subject_metadata["session"].unique()

            if len(sessions) <= 1:
                continue  # Skip subjects with only one session

            splitter = self.cv_class(**self._cv_kwargs)
            n_splits += splitter.get_n_splits(
                subject_metadata, groups=subject_metadata["session"]
            )
        return n_splits

    def split(self, y, metadata):
        # here, I am getting the index across all the subject
        all_index = metadata.index.values
        # I check how many subjects are here:
        subjects = metadata["subject"].unique()

        # To make sure that when I shuffle the subject, I shuffle the same way
        # the session when the object is created
        cv_kwargs = {**self._cv_kwargs}  # Copy the original kwargs
        if self._need_rng:
            cv_kwargs["random_state"] = check_random_state(self.random_state)

        # For each subject I am creating the mask to select the subject metainformation.
        for subject in subjects:
            # Creating the subject_mask
            subject_mask = metadata["subject"] == subject
            # from all the index, I am getting the trial index
            subject_indices = all_index[subject_mask]
            # Here, I am getting the metainformation to use the column session
            subject_metadata = metadata[subject_mask]
            # getting the label at subject level
            y_subject = y[subject_mask]
            # check the number of sessions and check how many sessions we
            # have!
            sessions = subject_metadata["session"].unique()

            if len(sessions) <= 1:
                log.info(
                    f"Skipping subject {subject}: Only one session available"
                    f"Cross-session evaluation requires at least two sessions."
                )
                continue  # Skip subjects with only one session

            # by default, I am using LeaveOneGroupOut
            splitter = self.cv_class(**cv_kwargs)

            # Yield the splits for a given subject
            for train_session_idx, test_session_idx in splitter.split(
                X=subject_indices, y=y_subject, groups=subject_metadata["session"]
            ):
                # returning the index
                yield subject_indices[train_session_idx], subject_indices[
                    test_session_idx
                ]


class CrossSubjectSplitter(BaseCrossValidator):
    """Data splitter for cross subject evaluation.

    This splitter enables cross-subject evaluation by performing a Leave-One-Session-Out (LOSO)
    cross-validation on the dataset.

    It assumes that the entire metainformation across all subjects is already loaded.

    Unlike the `CrossSubjectEvaluation` class from `moabb.evaluation`, which manages
    the complete evaluation process end-to-end, this splitter is solely responsible
    for dividing the data into training and testing sets based on subjects.

    .. image:: https://raw.githubusercontent.com/NeuroTechX/moabb/refs/heads/develop/docs/source/images/crosssubj.png
        :alt: The schematic diagram of the CrossSubject split
        :align: center

    The splitting strategy for the subjects can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses LeaveOneGroupOut,
    which performs Leave-One-Subject-Out cross-validation.

    Parameters
    ----------
    cv_class: cross-validation class, default=LeaveOneGroupOut
        Cross-validation strategy for splitting the subjects between train and test sets.
        By default, use LeaveOneGroupOut, which keeps one subject as a test.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness of the cross-validation.
        Pass an int for reproducible output across multiple calls.
    cv_kwargs: dict
        Additional arguments to pass to the inner cross-validation strategy.

    Yields
    ------
    train : ndarray
        The training set indices for that split.

    test : ndarray
        The testing set indices for that split.
    """

    def __init__(
        self,
        cv_class: type[BaseCrossValidator] = LeaveOneGroupOut,
        random_state: int = None,
        **cv_kwargs,
    ):
        self.cv_class = cv_class
        self.cv_kwargs = cv_kwargs
        self._cv_kwargs = dict(**cv_kwargs)

        params = inspect.signature(self.cv_class).parameters
        if "random_state" in params:
            self._cv_kwargs["random_state"] = random_state

    def get_n_splits(self, metadata):
        """
        Return the number of splits for the cross-validation.

        The number of splits is the number of subjects times the number of splits
        of the inner cross-validation strategy.

        We try to keep the same behaviour as the sklearn cross-validation classes.

        Parameters
        ----------
        metadata: pd.DataFrame
            The metadata containing the subject and session information.

        Returns
        -------
        n_splits: int
            The number of splits for the cross-validation
        """

        splitter = self.cv_class(**self._cv_kwargs)
        n_splits = splitter.get_n_splits(metadata.index, groups=metadata["subject"])
        return n_splits

    def split(self, y, metadata):
        # here, I am getting the index across all the subject
        all_index = metadata.index.values

        splitter = self.cv_class(**self._cv_kwargs)

        # Yield the splits for the entire dataset
        for train_session_idx, test_session_idx in splitter.split(
            X=all_index, y=y, groups=metadata["subject"]
        ):
            # returning the index
            yield all_index[train_session_idx], all_index[test_session_idx]
