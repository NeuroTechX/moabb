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

    .. image:: images/withinsess.png
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

    .. image:: images/crosssess.jpg
        :alt: The schematic diagram of the CrossSession split
        :align: center

    The inner cross-validation strategy can be changed by passing the
    `cv_class` and `cv_kwargs` arguments. By default, it uses LeaveOneGroupOut,
    which effectively performs Leave-One-Session-Out cross-validation when
    sessions are passed as groups.

    Parameters
    ----------
    cv_class: cross-validation class, default=LeaveOneGroupOut
        Inner cross-validation strategy for splitting the sessions.
        For cross-session splitting, LeaveOneGroupOut is the most suitable as default.
    shuffle: bool, default=False
        Whether to shuffle the session order for each subject. By default, it is not
        used because of LeaveOneGroupOut's determinist behaviour.
    random_state: int, RandomState instance or None, default=None
        Controls the randomness when `shuffle` is True.
        Pass an int for reproducible output across multiple function calls.
        Be default, it is not used because of the determinist behaviour of
        LeaveOneGroupOut.
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
        self._cv_kwargs = cv_kwargs

        self.shuffle = shuffle
        self.random_state = random_state

        if not shuffle and random_state is not None:
            raise ValueError("`random_state` should be None when `shuffle` is False")

        if shuffle and len(self._cv_kwargs) == 0 and cv_class is LeaveOneGroupOut:
            raise ValueError(
                "Shuffling is not implemented for LeaveOneGroupOut. "
                "The `shuffle` parameter change the behaviour of the splitter."
                "Use GroupShuffleSplit instead."
            )

    @property
    def cv_kwargs(self):
        params = inspect.signature(self.cv_class).parameters
        rng = None
        if self.shuffle:
            rng = check_random_state(self.random_state)
        for p, v in [
            ("shuffle", self.shuffle),
            ("random_state", rng),
        ]:
            if p in params:
                self._cv_kwargs[p] = v

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
            subject_metadata = metadata.filter("subject == @subject)
            sessions = subject_metadata["session"].unique()

            if len(sessions) <= 1:
                continue  # Skip subjects with only one session

            self.cv_kwargs
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
        self.cv_kwargs

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
            splitter = self.cv_class(**self._cv_kwargs)

            # Yield the splits for a given subject
            for train_session_idx, test_session_idx in splitter.split(
                X=subject_indices, y=y_subject, groups=subject_metadata["session"]
            ):
                # returning the index
                yield subject_indices[train_session_idx], subject_indices[
                    test_session_idx
                ]
