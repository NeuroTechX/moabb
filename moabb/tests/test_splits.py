import numpy as np
import pytest
from sklearn.model_selection import (
    GroupKFold,
    GroupShuffleSplit,
    KFold,
    LeaveOneGroupOut,
    LeaveOneOut,
    LeavePGroupsOut,
    LeavePOut,
    PredefinedSplit,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.utils import check_random_state

import moabb.evaluations.splitters as splitters_module
from moabb.datasets.fake import FakeDataset
from moabb.evaluations.splitters import (
    CrossDatasetSplitter,
    CrossSessionSplitter,
    CrossSubjectSplitter,
    LearningCurveSplitter,
    WithinSessionSplitter,
    WithinSubjectSplitter,
)
from moabb.paradigms.motor_imagery import FakeImageryParadigm


@pytest.fixture
def data():
    dataset = FakeDataset(
        ["left_hand", "right_hand"], n_subjects=5, seed=12, n_sessions=5
    )
    paradigm = FakeImageryParadigm()
    return paradigm.get_data(dataset=dataset)


@pytest.fixture(scope="module")
def small_data():
    dataset = FakeDataset(
        ["left_hand", "right_hand"], n_subjects=2, n_sessions=2, n_runs=2, seed=12
    )
    return FakeImageryParadigm().get_data(dataset=dataset)


class OpaqueLeaveOneGroupOut(LeaveOneGroupOut):
    """Real group CV whose constructor signature cannot be inspected."""

    __signature__ = object()


class OpaqueGroupKFold(GroupKFold):
    """Real shuffled group CV whose constructor signature is opaque."""

    __signature__ = object()


def _group_run(metadata):
    return metadata["run"].to_numpy()


def _group_session(metadata):
    return metadata["session"].to_numpy()


def _group_subject(metadata):
    return metadata["subject"].to_numpy()


def _group_dataset(metadata):
    return metadata["dataset"].to_numpy()


def _assert_same_split_arrays(first, second):
    assert len(first) == len(second)
    for (train_first, test_first), (train_second, test_second) in zip(first, second):
        assert np.array_equal(train_first, train_second)
        assert np.array_equal(test_first, test_second)


def _assert_disjoint_split_groups(folds, metadata, group_callable):
    for train, test in folds:
        train_groups = set(group_callable(metadata.loc[train]).tolist())
        test_groups = set(group_callable(metadata.loc[test]).tolist())
        assert train_groups.isdisjoint(test_groups)


def _canonical_dataset_assignments(folds, metadata):
    return tuple(
        sorted(
            (tuple(sorted(set(metadata.loc[test, "dataset"]))) for _train, test in folds),
            key=repr,
        )
    )


def _direct_dataset_group_kfold_assignments(metadata, seed):
    groups = metadata["dataset"].to_numpy()
    splitter = GroupKFold(n_splits=3, shuffle=True, random_state=seed)
    return tuple(
        sorted(
            (
                tuple(sorted(set(groups[test])))
                for _train, test in splitter.split(np.zeros(len(metadata)), groups=groups)
            ),
            key=repr,
        )
    )


def _four_equal_dataset_groups(data):
    _, y, metadata = data
    selected_subjects = metadata["subject"].unique()[:4]
    mask = metadata["subject"].isin(selected_subjects)
    metadata = _metadata_with_dataset_column(metadata.loc[mask], n_datasets=4)
    return y[mask.to_numpy()], metadata


# Split done for the Within Session evaluation
def eval_split_within_session(shuffle, random_state, data):
    _, y, metadata = data

    all_index = metadata.index.values
    # Convert to numpy array to avoid ArrowStringArray shuffle warning
    subjects = np.array(metadata["subject"].unique())
    if shuffle:
        shuffle_rng = check_random_state(random_state)
        shuffle_rng.shuffle(subjects)

    for _i, subject in enumerate(subjects):
        subject_mask = metadata["subject"] == subject

        subject_indices = all_index[subject_mask]
        subject_metadata = metadata[subject_mask]
        # Convert to numpy array to avoid ArrowStringArray shuffle warning
        sessions = np.array(subject_metadata["session"].unique())
        y_subject = y[subject_mask]

        if shuffle:
            shuffle_rng.shuffle(sessions)

        for session in sessions:
            session_mask = subject_metadata["session"] == session
            indices = subject_indices[session_mask]
            metadata_ = subject_metadata[session_mask]
            y_ = y_subject[session_mask]

            cv_rng = check_random_state(random_state) if shuffle else None
            cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=cv_rng)

            for idx_train, idx_test in cv.split(metadata_, y_):
                yield indices[idx_train], indices[idx_test]


def eval_split_within_subject(shuffle, random_state, data):
    _, y, metadata = data
    rng = check_random_state(random_state) if shuffle else None

    all_index = metadata.index.values
    # Convert to numpy array to avoid ArrowStringArray shuffle warning
    subjects = np.array(metadata["subject"].unique())
    if shuffle:
        rng.shuffle(subjects)

    for subject in subjects:
        subject_mask = metadata["subject"] == subject
        subject_indices = all_index[subject_mask]
        y_subject = y[subject_mask]

        cv = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=rng)

        for idx_train, idx_test in cv.split(subject_indices, y_subject):
            yield subject_indices[idx_train], subject_indices[idx_test]


def eval_split_cross_subject(shuffle, random_state, data):
    rng = check_random_state(random_state) if shuffle else None

    _, y, metadata = data
    subjects = metadata["subject"].unique()

    if shuffle:
        splitter = GroupShuffleSplit(random_state=rng)
    else:
        splitter = LeaveOneGroupOut()

    for train_subj_idx, test_subj_idx in splitter.split(
        X=np.zeros(len(subjects)), y=None, groups=subjects
    ):
        train_mask = metadata["subject"].isin(subjects[train_subj_idx])
        test_mask = metadata["subject"].isin(subjects[test_subj_idx])

        yield metadata.index[train_mask].values, metadata.index[test_mask].values


def _metadata_with_dataset_column(metadata, n_datasets=3):
    """Attach a synthetic dataset-group column while preserving subjects."""
    metadata = metadata.copy()
    subjects = np.array(metadata["subject"].unique())
    n_datasets = max(1, min(n_datasets, len(subjects)))
    dataset_labels = np.array([f"ds_{i + 1}" for i in range(n_datasets)])
    subject_to_dataset = {
        subject: dataset_labels[i % len(dataset_labels)]
        for i, subject in enumerate(subjects)
    }
    metadata["dataset"] = metadata["subject"].map(subject_to_dataset)
    return metadata


def eval_split_cross_dataset(shuffle, random_state, data):
    rng = check_random_state(random_state) if shuffle else None

    _, y, metadata = data
    metadata = _metadata_with_dataset_column(metadata)
    datasets = metadata["dataset"].unique()

    if shuffle:
        splitter = GroupShuffleSplit(random_state=rng)
    else:
        splitter = LeaveOneGroupOut()

    for train_dataset_idx, test_dataset_idx in splitter.split(
        X=np.zeros(len(datasets)), y=None, groups=datasets
    ):
        train_mask = metadata["dataset"].isin(datasets[train_dataset_idx])
        test_mask = metadata["dataset"].isin(datasets[test_dataset_idx])

        yield metadata.index[train_mask].values, metadata.index[test_mask].values


def eval_split_cross_session(shuffle, random_state, data):
    _, y, metadata = data

    rng = check_random_state(random_state) if shuffle else None

    subjects = metadata["subject"].unique()

    for subject in subjects:
        subject_mask = metadata["subject"] == subject
        subject_metadata = metadata[subject_mask]
        subject_sessions = subject_metadata["session"].unique()

        if shuffle:
            splitter = GroupShuffleSplit(n_splits=len(subject_sessions), random_state=rng)
        else:
            splitter = LeaveOneGroupOut()

        for train_ix, test_ix in splitter.split(
            X=subject_metadata, y=y[subject_mask], groups=subject_metadata["session"]
        ):
            yield subject_metadata.index[train_ix], subject_metadata.index[test_ix]


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_within_session_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_within_session(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        # Check if the output is the same as the input
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_within_subject_compatibility(shuffle, random_state, data):
    _, y, metadata = data

    split = WithinSubjectSplitter(n_folds=5, shuffle=shuffle, random_state=random_state)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        eval_split_within_subject(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        # Check if the output is the same as the input
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_is_shuffling(data):
    X, y, metadata = data

    split = WithinSessionSplitter(n_folds=5, shuffle=False)
    split_shuffle = WithinSessionSplitter(n_folds=5, shuffle=True, random_state=3)

    for (train, test), (train_shuffle, test_shuffle) in zip(
        split.split(y, metadata), split_shuffle.split(y, metadata)
    ):
        # Check if the output is the same as the input
        assert not np.array_equal(train, train_shuffle)
        assert not np.array_equal(test, test_shuffle)


@pytest.mark.parametrize(
    "splitter",
    [
        WithinSessionSplitter,
        WithinSubjectSplitter,
        CrossSessionSplitter,
        CrossSubjectSplitter,
        CrossDatasetSplitter,
    ],
)
def test_custom_inner_cv(splitter, data):
    X, y, metadata = data
    if splitter == CrossDatasetSplitter:
        metadata = _metadata_with_dataset_column(metadata)
    # Use a custom inner cv
    split = splitter(cv_class=TimeSeriesSplit, max_train_size=2)

    for train, test in split.split(y, metadata):
        # Check if the output is the same as the input
        assert len(train) <= 2  # Due to TimeSeriesSplit constraints
        assert len(test) >= 20


def test_custom_shuffle_group(data):
    _, y, metadata = data

    n_splits = 5
    splitter = CrossSubjectSplitter(
        random_state=42, cv_class=GroupShuffleSplit, n_splits=n_splits
    )

    splits = list(splitter.split(y, metadata))

    assert len(splits) == n_splits, f"Expected {n_splits} splits, got {len(splits)}"

    for train, test in splits:
        train_subjects = metadata.iloc[train]["subject"].unique()
        test_subjects = metadata.iloc[test]["subject"].unique()

        # Assert no overlap between train and test subjects
        assert len(set(train_subjects) & set(test_subjects)) == 0

    # Check if shuffling produces different splits
    splitter_different_seed = CrossSubjectSplitter(
        cv_class=GroupShuffleSplit, n_splits=n_splits
    )
    splits_different_seed = list(splitter_different_seed.split(y, metadata))

    assert not all(
        np.array_equal(train, train_alt) and np.array_equal(test, test_alt)
        for (train, test), (train_alt, test_alt) in zip(splits, splits_different_seed)
    )


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session(shuffle, random_state, data):
    _, y, metadata = data

    params = {"random_state": random_state}
    params["shuffle"] = shuffle
    if shuffle:
        params["cv_class"] = GroupShuffleSplit

    split = CrossSessionSplitter(**params)

    for idx_train_splitter, idx_test_splitter in split.split(y, metadata):
        # Check if the output is the same as the input
        session_train = metadata.iloc[idx_train_splitter]["session"].unique()
        session_test = metadata.iloc[idx_test_splitter]["session"].unique()
        assert np.intersect1d(session_train, session_test).size == 0
        assert (
            np.union1d(session_train, session_test).size
            == metadata["session"].unique().size
        )


@pytest.mark.parametrize(
    "splitter", [CrossSessionSplitter, CrossSubjectSplitter, CrossDatasetSplitter]
)
@pytest.mark.parametrize("shuffle, random_state", [(False, None), (True, 0), (True, 42)])
def test_cross_compatibility(splitter, shuffle, random_state, data):
    _, y, metadata = data

    if splitter == CrossSessionSplitter:
        function_split = eval_split_cross_session
    elif splitter == CrossSubjectSplitter:
        function_split = eval_split_cross_subject
    else:
        function_split = eval_split_cross_dataset
        metadata = _metadata_with_dataset_column(metadata)

    params = {"random_state": random_state}
    if splitter == CrossSessionSplitter:
        params["shuffle"] = shuffle
    if shuffle:
        params["cv_class"] = GroupShuffleSplit

    split = splitter(**params)

    for (idx_train, idx_test), (idx_train_splitter, idx_test_splitter) in zip(
        function_split(shuffle=shuffle, random_state=random_state, data=data),
        split.split(y, metadata),
    ):
        assert np.array_equal(idx_train, idx_train_splitter)
        assert np.array_equal(idx_test, idx_test_splitter)


def test_cross_session_is_shuffling_and_order(data):
    _, y, metadata = data

    splitter_no_shuffle = CrossSessionSplitter(shuffle=False)
    splitter_shuffle = CrossSessionSplitter(
        shuffle=True, random_state=3, cv_class=GroupShuffleSplit
    )

    splits_no_shuffle = list(splitter_no_shuffle.split(y, metadata))
    splits_shuffle = list(splitter_shuffle.split(y, metadata))

    train_diff = []
    test_diff = []

    # For tracking session order differences
    session_orders_no_shuffle = []
    session_orders_shuffle = []

    for i, ((train_ns, test_ns), (train_s, test_s)) in enumerate(
        zip(splits_no_shuffle, splits_shuffle)
    ):
        print(f"\nFold {i}:")

        # Get session ordering for non-shuffled and shuffled
        train_ns_sessions = metadata.iloc[train_ns]["session"].unique()
        test_ns_sessions = metadata.iloc[test_ns]["session"].unique()
        train_s_sessions = metadata.iloc[train_s]["session"].unique()
        test_s_sessions = metadata.iloc[test_s]["session"].unique()

        print(f"Train no shuffle sessions: {train_ns_sessions}")
        print(f"Test no shuffle sessions : {test_ns_sessions}")
        print(f"Train shuffled sessions  : {train_s_sessions}")
        print(f"Test shuffle sessions    : {test_s_sessions}")

        # Track if indices are the same
        train_diff.append(np.array_equal(train_ns, train_s))
        test_diff.append(np.array_equal(test_ns, test_s))

        # Track session orders
        session_orders_no_shuffle.append(
            (list(train_ns_sessions), list(test_ns_sessions))
        )
        session_orders_shuffle.append((list(train_s_sessions), list(test_s_sessions)))

    # Check if indices are different in at least some folds
    assert not all(train_diff), "All train indices are identical despite shuffle"
    assert not all(test_diff), "All test indices are identical despite shuffle"

    # Check if session ordering is different
    session_order_differences = [
        not (
            np.array_equal(no_shuffle[0], shuffle[0])
            and np.array_equal(no_shuffle[1], shuffle[1])
        )
        for no_shuffle, shuffle in zip(session_orders_no_shuffle, session_orders_shuffle)
    ]

    assert any(session_order_differences), (
        "Session ordering is identical in all folds despite shuffle. "
        "When shuffle=True, we expect some difference in the session ordering."
    )


def test_cross_session_unique_subjects(data):
    _, y, metadata = data

    splitter_shuffle = CrossSessionSplitter(
        shuffle=True, random_state=3, cv_class=GroupShuffleSplit
    )
    splits_shuffle = list(splitter_shuffle.split(y, metadata))

    # Check if session splits are different across subjects
    subject_session_patterns = {}
    for _i, (train_idx, test_idx) in enumerate(splits_shuffle):
        subject = metadata.iloc[train_idx]["subject"].iloc[
            0
        ]  # Get the subject for this fold
        if subject not in subject_session_patterns:
            subject_session_patterns[subject] = []

        train_sessions = set(metadata.iloc[train_idx]["session"].unique())
        test_sessions = set(metadata.iloc[test_idx]["session"].unique())
        subject_session_patterns[subject].append((train_sessions, test_sessions))

    # Verify that at least some subjects have different session splitting patterns
    pattern_differences = []
    subjects = list(subject_session_patterns.keys())
    for sub1, sub2 in zip(subjects, subjects[1:]):
        # Compare patterns for each subject pair
        patterns_differ = False
        for (train1, test1), (train2, test2) in zip(
            subject_session_patterns[sub1], subject_session_patterns[sub2]
        ):
            if train1 != train2 or test1 != test2:
                patterns_differ = True
                break
        pattern_differences.append(patterns_differ)

    assert any(pattern_differences), (
        "Session splitting patterns are identical across all subjects"
    )


@pytest.mark.parametrize("shuffle, random_state", [(True, 0), (True, 42), (False, None)])
def test_cross_session_unique_sessions(shuffle, random_state, data):
    _, y, metadata = data
    if shuffle:
        split = CrossSessionSplitter(
            shuffle=shuffle, random_state=random_state, cv_class=GroupShuffleSplit
        )
    else:
        split = CrossSessionSplitter(shuffle=shuffle, random_state=random_state)

    splits = list(split.split(y, metadata))

    for i, (train, test) in enumerate(splits):
        train_sessions = metadata.iloc[train]["session"].unique()
        test_sessions = metadata.iloc[test]["session"].unique()
        assert not np.intersect1d(train_sessions, test_sessions).size, (
            f"Fold {i} train and test sessions overlap"
        )


@pytest.mark.parametrize("shuffle", [True, False])
def test_cross_session_get_n_splits(data, shuffle):
    _, y, metadata = data
    if shuffle:
        split = CrossSessionSplitter(shuffle=shuffle, cv_class=GroupShuffleSplit)
    else:
        split = CrossSessionSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == 5 * 5  # 5 subjects, 5 sessions each


def test_cross_subject_get_n_splits(data):
    _, y, metadata = data

    split = CrossSubjectSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == 5  # 5 subjects


def test_cross_dataset_get_n_splits(data):
    _, y, metadata = data
    metadata = _metadata_with_dataset_column(metadata)

    split = CrossDatasetSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == metadata["dataset"].nunique()


def test_within_subject_get_n_splits(data):
    _, y, metadata = data

    split = WithinSubjectSplitter()

    n_splits = split.get_n_splits(metadata)
    assert n_splits == 5 * 5  # 5 subjects, 5 folds each


@pytest.mark.parametrize("splitter", [WithinSessionSplitter, WithinSubjectSplitter])
def test_cv_kwargs_n_splits_not_overwritten(data, splitter):
    """Explicit n_splits in cv_kwargs must not be overwritten by n_folds."""
    _, y, metadata = data

    split = splitter(
        cv_class=StratifiedShuffleSplit,
        n_splits=1,
        test_size=0.25,
        shuffle=True,
        random_state=42,
    )

    # The inner cv should keep the explicitly requested single split.
    assert split._cv_kwargs["n_splits"] == 1

    if splitter == WithinSessionSplitter:
        num_groups = metadata.groupby(["subject", "session"]).ngroups
    else:
        num_groups = metadata["subject"].nunique()

    splits = list(split.split(y, metadata))
    assert len(splits) == num_groups  # one split per group, not n_folds per group


@pytest.mark.parametrize("splitter", [WithinSessionSplitter, WithinSubjectSplitter])
def test_within_split_is_reproducible(data, splitter):
    """Repeated split() calls with a fixed seed must yield identical folds."""
    _, y, metadata = data
    split = splitter(shuffle=True, random_state=42)
    first = list(split.split(y, metadata))
    second = list(split.split(y, metadata))
    assert len(first) == len(second)
    for (train, test), (train_2, test_2) in zip(first, second):
        assert np.array_equal(train, train_2)
        assert np.array_equal(test, test_2)


@pytest.mark.parametrize(
    "splitter", [CrossSessionSplitter, CrossSubjectSplitter, CrossDatasetSplitter]
)
def test_if_split_is_not_random(data, splitter):
    _, y, metadata = data
    if splitter == CrossDatasetSplitter:
        metadata = _metadata_with_dataset_column(metadata)

    if splitter == CrossSessionSplitter:
        split = splitter(shuffle=True, random_state=42, cv_class=GroupShuffleSplit)
    else:
        split = splitter(random_state=42, cv_class=GroupShuffleSplit)

    splits = list(split.split(y, metadata))
    splits_2 = list(split.split(y, metadata))

    for (train, test), (train_2, test_2) in zip(splits, splits_2):
        print(f"Train: {train}")
        print(f"Test: {test}")
        print(f"Train 2: {train_2}")
        print(f"Test 2: {test_2}")
        assert np.array_equal(train, train_2)
        assert np.array_equal(test, test_2)


@pytest.mark.parametrize(
    "cv_class",
    [
        LeaveOneGroupOut,
        TimeSeriesSplit,
        # GroupKFold, changed behavior within scikit-learn 1.6
        LeaveOneOut,
        LeavePGroupsOut,
        LeavePOut,
    ],
)
def test_raise_error_on_invalid_cv_class(cv_class):
    with pytest.raises(ValueError):
        CrossSessionSplitter(shuffle=True, cv_class=cv_class)


@pytest.mark.parametrize(
    "cv_class",
    [
        GroupShuffleSplit,
        StratifiedKFold,
        KFold,
        RepeatedKFold,
        RepeatedStratifiedKFold,
        ShuffleSplit,
        StratifiedGroupKFold,
        StratifiedShuffleSplit,
    ],
)
def test_cross_session_splitter_without_error(cv_class):
    splitter = CrossSessionSplitter(shuffle=True, cv_class=cv_class)
    assert splitter is not None
    assert isinstance(splitter, CrossSessionSplitter)


def test_learning_curve_splitter_metadata():
    y = np.array([0, 1] * 10)
    data_size = {"policy": "ratio", "value": np.array([0.5, 1.0])}
    n_perms = np.array([2, 1])
    splitter = LearningCurveSplitter(
        data_size=data_size, n_perms=n_perms, test_size=0.2, random_state=0
    )

    splits = list(splitter.split(np.arange(len(y)), y))
    assert len(splits) == int(np.sum(n_perms))

    for _train, _test in splits:
        meta = splitter.get_metadata()
        assert meta["data_size"] is not None
        assert meta["permutation"] is not None


def test_learning_curve_subsample_is_random_with_groups():
    """Each permutation must draw its own training set when groups are passed.

    ``== n_perms``, not ``> 1``: several training pools are possible, so the
    ascending prefix already differs between some permutations and ``> 1`` passes
    with or without the fix.
    """
    n_perms = 5
    y = np.array([0, 1] * 100)
    groups = np.repeat(np.arange(5), 40)
    splitter = LearningCurveSplitter(
        data_size={"policy": "per_class", "value": [2, 5, 10]},
        n_perms=n_perms,
        test_size=0.2,
        random_state=42,
    )

    trains = [
        tuple(sorted(train))
        for train, _ in splitter.split(np.arange(len(y)), y, groups=groups)
    ]

    by_size = {}
    for train in trains:
        by_size.setdefault(len(train), set()).add(train)
    for size, distinct in by_size.items():
        assert len(distinct) == n_perms, (
            f"{n_perms} permutations produced {len(distinct)} distinct "
            f"{size}-sample training sets"
        )


def test_learning_curve_subsample_keeps_test_folds_and_ungrouped_splits():
    """The shuffle must move the training subsample and nothing else."""
    y = np.array([0, 1] * 100)
    groups = np.repeat(np.arange(5), 40)
    kwargs = {
        "data_size": {"policy": "per_class", "value": [2, 5, 10]},
        "n_perms": 5,
        "test_size": 0.2,
        "random_state": 42,
    }

    def run(g, random_state=42):
        splitter = LearningCurveSplitter(**(kwargs | {"random_state": random_state}))
        return [
            (tuple(sorted(train)), tuple(sorted(test)))
            for train, test in splitter.split(np.arange(len(y)), y, groups=g)
        ]

    grouped, ungrouped = run(groups), run(None)

    # Draining the grouped base splitter before subsampling must preserve its
    # test folds, including when its random state is a shared mutable object.
    for random_state_factory in (lambda: 42, lambda: np.random.RandomState(42)):
        expected = [
            tuple(sorted(test))
            for _, test in GroupShuffleSplit(
                n_splits=kwargs["n_perms"],
                test_size=kwargs["test_size"],
                random_state=random_state_factory(),
            ).split(np.arange(len(y)), y, groups)
        ]
        actual = [
            test
            for _, test in run(groups, random_state_factory())[
                :: len(kwargs["data_size"]["value"])
            ]
        ]
        assert actual == expected

    # Every group appears in exactly one side of each split.
    for train, test in grouped:
        assert not set(groups[list(train)]) & set(groups[list(test)])

    # The ungrouped branch still draws distinct subsamples per permutation.
    assert len({train for train, _ in ungrouped}) == len(ungrouped)


def _assert_random_states_equal(left, right):
    """Assert equality of the full NumPy RandomState state tuple."""
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def test_learning_curve_grouped_base_splitter_stays_lazy(monkeypatch):
    """One requested split must consume only one grouped base permutation."""
    base_yields = []

    class CountingGroupShuffleSplit:
        def __init__(self, n_splits, test_size, random_state):
            self.n_splits = n_splits

        def split(self, X, y, groups):
            for perm_i in range(self.n_splits):
                base_yields.append(perm_i)
                test_mask = groups == perm_i % len(np.unique(groups))
                yield np.flatnonzero(~test_mask), np.flatnonzero(test_mask)

    monkeypatch.setattr(splitters_module, "GroupShuffleSplit", CountingGroupShuffleSplit)
    y = np.array([0, 1] * 50)
    groups = np.repeat(np.arange(5), 20)
    splitter = LearningCurveSplitter(
        data_size={"policy": "ratio", "value": [1.0]},
        n_perms=50,
        test_size=0.2,
        random_state=42,
    )

    iterator = splitter.split(np.arange(len(y)), y, groups=groups)
    next(iterator)

    assert base_yields == [0]


@pytest.mark.parametrize("random_state_kind", ["shared", "global"])
def test_learning_curve_partial_iteration_advances_only_one_base_fold(random_state_kind):
    """Private subsampling must not pre-consume the caller-owned RNG stream."""
    y = np.array([0, 1] * 100)
    groups = np.repeat(np.arange(5), 40)
    kwargs = {
        "data_size": {"policy": "per_class", "value": [2, 5, 10]},
        "n_perms": 5,
        "test_size": 0.2,
    }

    if random_state_kind == "shared":
        direct_rng = np.random.RandomState(42)
        next(
            GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=direct_rng).split(
                np.arange(len(y)), y, groups
            )
        )
        expected_state = direct_rng.get_state()

        actual_rng = np.random.RandomState(42)
        next(
            LearningCurveSplitter(**kwargs, random_state=actual_rng).split(
                np.arange(len(y)), y, groups=groups
            )
        )
        actual_state = actual_rng.get_state()
    else:
        np.random.seed(42)
        next(
            GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=None).split(
                np.arange(len(y)), y, groups
            )
        )
        expected_state = np.random.get_state()

        np.random.seed(42)
        next(
            LearningCurveSplitter(**kwargs, random_state=None).split(
                np.arange(len(y)), y, groups=groups
            )
        )
        actual_state = np.random.get_state()

    _assert_random_states_equal(actual_state, expected_state)


@pytest.mark.parametrize(
    ("policy", "values", "expected_counts"),
    [
        ("ratio", [0.1, 0.25, 0.5], {16: 5, 40: 3, 80: 1}),
        ("per_class", [2, 5, 10], {4: 5, 10: 3, 20: 1}),
    ],
)
def test_learning_curve_grouped_vector_permutations_are_nested_and_reproducible(
    policy, values, expected_counts
):
    """Both policies honor vector counts, nesting, grouping, and RNG modes."""
    y = np.array([0, 1] * 100)
    groups = np.repeat(np.arange(5), 40)
    kwargs = {
        "data_size": {"policy": policy, "value": values},
        "n_perms": [5, 3, 1],
        "test_size": 0.2,
    }

    def run(random_state):
        splitter = LearningCurveSplitter(**kwargs, random_state=random_state)
        result = []
        for train, test in splitter.split(np.arange(len(y)), y, groups=groups):
            metadata = splitter.get_metadata()
            result.append((metadata["permutation"], tuple(train), tuple(test)))
        return result

    first = run(42)
    assert first == run(42)
    assert len(first) == sum(expected_counts.values())
    assert {
        size: sum(len(train) == size for _, train, _ in first) for size in expected_counts
    } == expected_counts

    by_permutation = {}
    for permutation, train, test in first:
        assert not set(groups[list(train)]) & set(groups[list(test)])
        by_permutation.setdefault(permutation, []).append(set(train))
    for subsets in by_permutation.values():
        for smaller, larger in zip(subsets, subsets[1:]):
            assert smaller < larger

    shared = np.random.RandomState(42)
    assert run(shared) != run(shared)
    np.random.seed(42)
    global_first = run(None)
    np.random.seed(42)
    assert global_first == run(None)


@pytest.mark.parametrize(
    "splitter",
    [
        WithinSessionSplitter,
        WithinSubjectSplitter,
        CrossSessionSplitter,
        CrossSubjectSplitter,
        CrossDatasetSplitter,
    ],
)
def test_learning_curve_as_cv_class(splitter, data):
    """Test that LearningCurveSplitter can be used as cv_class for all splitters."""
    _, y, metadata = data
    if splitter == CrossDatasetSplitter:
        metadata = _metadata_with_dataset_column(metadata)

    data_size = {"policy": "ratio", "value": np.array([0.5, 1.0])}
    n_perms = np.array([2, 1])

    # CrossSessionSplitter requires shuffle=True when using random_state
    extra_kwargs = {}
    if splitter == CrossSessionSplitter:
        extra_kwargs["shuffle"] = True

    split = splitter(
        cv_class=LearningCurveSplitter,
        data_size=data_size,
        n_perms=n_perms,
        test_size=0.2,
        random_state=42,
        **extra_kwargs,
    )

    splits = list(split.split(y, metadata))
    assert len(splits) > 0

    for train, test in splits:
        # Check that we get valid train/test indices
        assert len(train) > 0
        assert len(test) > 0
        # Check no overlap between train and test
        assert len(set(train) & set(test)) == 0
        if splitter == CrossSessionSplitter:
            train_meta = metadata.loc[train]
            test_meta = metadata.loc[test]
            train_subjects = set(train_meta["subject"])
            test_subjects = set(test_meta["subject"])
            assert len(train_subjects) == 1
            assert train_subjects == test_subjects
            train_sessions = set(train_meta["session"])
            test_sessions = set(test_meta["session"])
            assert train_sessions.isdisjoint(test_sessions)
        elif splitter == CrossSubjectSplitter:
            train_subjects = set(metadata.loc[train]["subject"])
            test_subjects = set(metadata.loc[test]["subject"])
            assert train_subjects.isdisjoint(test_subjects)
        elif splitter == CrossDatasetSplitter:
            train_datasets = set(metadata.loc[train]["dataset"])
            test_datasets = set(metadata.loc[test]["dataset"])
            assert train_datasets.isdisjoint(test_datasets)


@pytest.mark.parametrize(
    "splitter_cls",
    [
        WithinSessionSplitter,
        WithinSubjectSplitter,
        CrossSessionSplitter,
        CrossSubjectSplitter,
        CrossDatasetSplitter,
    ],
)
def test_splitter_metadata_interface(splitter_cls, data):
    """Test get_metadata() access for splitters using metadata-aware inner CV."""
    _, y, metadata = data
    if splitter_cls == CrossDatasetSplitter:
        metadata = _metadata_with_dataset_column(metadata)

    data_size = {"policy": "ratio", "value": np.array([0.5, 1.0])}
    n_perms = np.array([2, 1])

    extra_kwargs = {}
    if splitter_cls == CrossSessionSplitter:
        extra_kwargs["shuffle"] = True

    split = splitter_cls(
        cv_class=LearningCurveSplitter,
        data_size=data_size,
        n_perms=n_perms,
        test_size=0.2,
        random_state=42,
        **extra_kwargs,
    )

    has_split = False
    for _train, _test in split.split(y, metadata):
        has_split = True
        meta = split.get_metadata()
        assert meta is not None
        assert meta["data_size"] is not None
        assert meta["permutation"] is not None
    assert has_split


def test_cross_dataset_requires_group_column(data):
    _, y, metadata = data
    splitter = CrossDatasetSplitter(group_column="does_not_exist")
    with pytest.raises(ValueError):
        list(splitter.split(y, metadata))


# ---------------------------------------------------------------------------
# Metadata-driven ``groups`` and callable ``cv_kwargs`` across the splitters.
# ---------------------------------------------------------------------------


def test_cross_subject_groups_compound_key(data):
    """groups=["subject", "session"] yields one fold per (subject, session)."""
    _, y, metadata = data
    split = CrossSubjectSplitter(cv_class=LeaveOneGroupOut, groups=["subject", "session"])
    n_groups = metadata.groupby(["subject", "session"]).ngroups
    folds = list(split.split(y, metadata))
    assert len(folds) == n_groups
    assert split.get_n_splits(metadata) == n_groups
    for train, test in folds:
        test_meta = metadata.loc[test]
        assert test_meta.groupby(["subject", "session"]).ngroups == 1
        train_keys = set(map(tuple, metadata.loc[train][["subject", "session"]].values))
        test_keys = set(map(tuple, test_meta[["subject", "session"]].values))
        assert train_keys.isdisjoint(test_keys)


def test_cross_subject_predefined_split_single_fold(data):
    """cv_class=PredefinedSplit with a callable test_fold targets one fold."""
    _, y, metadata = data
    split = CrossSubjectSplitter(
        cv_class=PredefinedSplit,
        test_fold=lambda md: np.where(
            (md["subject"] == 1) & (md["session"] == "0"), 0, -1
        ),
    )
    folds = list(split.split(y, metadata))
    assert len(folds) == 1
    assert split.get_n_splits(metadata) == 1
    train, test = folds[0]
    test_meta = metadata.loc[test]
    assert set(test_meta["subject"]) == {1}
    assert set(test_meta["session"]) == {"0"}
    assert len(train) + len(test) == len(metadata)


def test_cross_dataset_groups_callable_and_list(data):
    """CrossDatasetSplitter accepts groups as a list of columns or a callable."""
    _, y, metadata = data
    metadata = _metadata_with_dataset_column(metadata)
    n_datasets = metadata["dataset"].nunique()

    split_list = CrossDatasetSplitter(groups=["dataset"])
    folds_list = list(split_list.split(y, metadata))
    assert len(folds_list) == n_datasets

    split_call = CrossDatasetSplitter(groups=lambda md: md["dataset"].to_numpy())
    folds_call = list(split_call.split(y, metadata))
    assert len(folds_call) == n_datasets
    for train, test in folds_call:
        train_ds = set(metadata.loc[train, "dataset"])
        test_ds = set(metadata.loc[test, "dataset"])
        assert train_ds.isdisjoint(test_ds)


def test_cross_dataset_group_column_backcompat(data):
    """The deprecated group_column= keyword still drives the folds."""
    _, y, metadata = data
    metadata = _metadata_with_dataset_column(metadata)
    with pytest.warns(DeprecationWarning):
        split = CrossDatasetSplitter(group_column="dataset")
    folds = list(split.split(y, metadata))
    assert len(folds) == metadata["dataset"].nunique()
    # Per-fold metadata is preserved.
    meta = split.get_metadata()
    assert "test_dataset" in meta
    assert "train_datasets" in meta


@pytest.mark.parametrize(
    "splitter_class,groups,expected_folds",
    [
        (WithinSessionSplitter, _group_run, 8),
        (WithinSubjectSplitter, _group_session, 4),
        (CrossSessionSplitter, _group_session, 4),
        (CrossSubjectSplitter, _group_subject, 2),
        (CrossDatasetSplitter, _group_dataset, 3),
    ],
    ids=[
        "within-session",
        "within-subject",
        "cross-session",
        "cross-subject",
        "cross-dataset",
    ],
)
def test_known_false_random_state_remains_direct_splitter_only(
    data, small_data, splitter_class, groups, expected_folds
):
    """A LOGO-incompatible public seed remains deterministic splitter state."""
    if splitter_class is CrossDatasetSplitter:
        _, y, metadata = data
        metadata = _metadata_with_dataset_column(metadata)
    else:
        _, y, metadata = small_data

    def create():
        return splitter_class(cv_class=LeaveOneGroupOut, groups=groups, random_state=17)

    splitter = create()
    repeated_splitter = create()
    folds = list(splitter.split(y, metadata))
    repeated = list(repeated_splitter.split(y, metadata))
    assert splitter.random_state == 17
    assert len(folds) == expected_folds
    _assert_disjoint_split_groups(folds, metadata, groups)
    _assert_same_split_arrays(folds, repeated)


def test_cross_dataset_opaque_cv_class_splits(data):
    """Opaque LOGO executes real dataset-group splits without guessed defaults."""
    _, y, metadata = data
    metadata = _metadata_with_dataset_column(metadata)
    splitter = CrossDatasetSplitter(
        cv_class=OpaqueLeaveOneGroupOut, groups=_group_dataset
    )
    folds = list(splitter.split(y, metadata))
    assert len(folds) == 3
    _assert_disjoint_split_groups(folds, metadata, _group_dataset)


def test_cross_dataset_group_kfold_seed_pair_changes_canonical_assignments(data):
    """The fixed seed pair changes the direct four-dataset oracle."""
    _, metadata = _four_equal_dataset_groups(data)
    expected_17 = _direct_dataset_group_kfold_assignments(metadata, seed=17)
    expected_18 = _direct_dataset_group_kfold_assignments(metadata, seed=18)
    assert expected_17 != expected_18


def test_cross_dataset_opaque_group_kfold_preserves_explicit_seed(data):
    """Opaque direct dataset CV preserves a caller-selected non-None seed."""
    y, metadata = _four_equal_dataset_groups(data)

    def run(seed):
        splitter = CrossDatasetSplitter(
            cv_class=OpaqueGroupKFold,
            groups=_group_dataset,
            n_splits=3,
            shuffle=True,
            random_state=seed,
        )
        return list(splitter.split(y, metadata))

    folds_17 = run(17)
    folds_17_repeat = run(17)
    folds_18 = run(18)
    observed_17 = _canonical_dataset_assignments(folds_17, metadata)
    observed_18 = _canonical_dataset_assignments(folds_18, metadata)
    expected_17 = _direct_dataset_group_kfold_assignments(metadata, seed=17)
    expected_18 = _direct_dataset_group_kfold_assignments(metadata, seed=18)
    assert len(folds_17) == 3
    assert len(folds_18) == 3
    _assert_same_split_arrays(folds_17, folds_17_repeat)
    assert observed_17 == expected_17
    assert observed_18 == expected_18
    assert observed_18 != observed_17


def test_cross_session_groups_default_and_callable(data):
    """Default groups='session' is unchanged; a callable reproduces it."""
    _, y, metadata = data
    default_folds = list(CrossSessionSplitter().split(y, metadata))
    call_split = CrossSessionSplitter(groups=lambda md: md["session"].to_numpy())
    call_folds = list(call_split.split(y, metadata))
    assert len(call_folds) == len(default_folds)
    for (tr1, te1), (tr2, te2) in zip(default_folds, call_folds):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


@pytest.mark.parametrize("splitter_cls", [WithinSessionSplitter, WithinSubjectSplitter])
def test_within_groups_none_is_unchanged(splitter_cls, data):
    """groups=None reproduces the splits produced without the argument."""
    _, y, metadata = data
    base = list(splitter_cls(n_folds=5, shuffle=False).split(y, metadata))
    with_none = list(
        splitter_cls(n_folds=5, shuffle=False, groups=None).split(y, metadata)
    )
    assert len(base) == len(with_none)
    for (tr1, te1), (tr2, te2) in zip(base, with_none):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)


def test_within_session_groups_routes_through(data):
    """A group-aware cv_class receives groups resolved per (subject, session)."""
    _, y, metadata = data
    split = WithinSessionSplitter(shuffle=False, cv_class=LeaveOneGroupOut, groups="run")
    folds = list(split.split(y, metadata))
    n_subjects = metadata["subject"].nunique()
    n_sessions = metadata["session"].nunique()
    n_runs = metadata["run"].nunique()
    assert len(folds) == n_subjects * n_sessions * n_runs
    for train, test in folds:
        test_meta = metadata.loc[test]
        assert test_meta.groupby(["subject", "session"]).ngroups == 1
        assert test_meta["run"].nunique() == 1
        subj = test_meta["subject"].iloc[0]
        sess = test_meta["session"].iloc[0]
        held_out_run = test_meta["run"].iloc[0]
        same_partition = metadata.loc[train]
        same_partition = same_partition[
            (same_partition["subject"] == subj) & (same_partition["session"] == sess)
        ]
        assert held_out_run not in set(same_partition["run"])


def test_within_subject_groups_routes_through(data):
    """A group-aware cv_class receives groups resolved per subject."""
    _, y, metadata = data
    split = WithinSubjectSplitter(
        shuffle=False, cv_class=LeaveOneGroupOut, groups="session"
    )
    folds = list(split.split(y, metadata))
    n_subjects = metadata["subject"].nunique()
    n_sessions = metadata["session"].nunique()
    assert len(folds) == n_subjects * n_sessions
    for train, test in folds:
        test_meta = metadata.loc[test]
        assert test_meta["subject"].nunique() == 1
        assert test_meta["session"].nunique() == 1
        subj = test_meta["subject"].iloc[0]
        held_out_session = test_meta["session"].iloc[0]
        same_subject_train = metadata.loc[train]
        same_subject_train = same_subject_train[same_subject_train["subject"] == subj]
        assert held_out_session not in set(same_subject_train["session"])


# ---------------------------------------------------------------------------
# Cross-subject transfer learning: the target-calibration slice.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("calibration_size", [0.0, 0.3, 1.0])
def test_cross_subject_calibration(calibration_size, data):
    """Calibration folds are consumed uniformly with ``train, *cal, test``."""
    _, y, metadata = data
    base = CrossSubjectSplitter()
    cal_split = CrossSubjectSplitter(calibration_size=calibration_size)

    base_folds = list(base.split(y, metadata))
    cal_folds = list(cal_split.split(y, metadata))
    assert len(cal_folds) == len(base_folds) == cal_split.get_n_splits(metadata)

    for (b_train, b_test), fold in zip(base_folds, cal_folds):
        train, *cal, test = fold  # generic consumption: 2- or 3-tuple
        calib = cal[0] if cal else test[:0]

        assert np.array_equal(train, b_train)
        assert np.intersect1d(train, b_test).size == 0

        if calibration_size == 0.0:
            assert calib.size == 0
            assert np.array_equal(test, b_test)
        elif calibration_size == 1.0:
            assert np.array_equal(calib, b_test)
            assert np.array_equal(test, b_test)
        else:
            assert calib.size >= 1 and test.size >= 1
            assert np.array_equal(np.union1d(calib, test), b_test)


def test_cross_subject_calibration_invalid_size():
    with pytest.raises(ValueError):
        CrossSubjectSplitter(calibration_size=1.5)


@pytest.mark.parametrize("calibration_labeled", [False, True])
def test_cross_subject_calibration_leakage_boundary(calibration_labeled, data):
    """Pin the leakage contract of the transfer split.

    The held-out fold is a single target subject. The calibration slice is
    carved out of that target subject and reaches the estimator only through
    ``fit`` metadata routing -- never through ``train_idx`` -- and it is
    removed from the scored test set. Train, calibration and test are
    pairwise trial-disjoint.
    """
    _, y, metadata = data
    splitter = CrossSubjectSplitter(
        calibration_size=0.2, calibration_labeled=calibration_labeled
    )
    folds = list(splitter.split(y, metadata))
    assert len(folds) == metadata["subject"].nunique()

    for train_idx, calib_idx, test_idx in folds:
        target = set(metadata.loc[test_idx, "subject"])
        # Exactly one held-out target subject per fold.
        assert len(target) == 1
        # The calibration slice belongs to that same target subject...
        assert set(metadata.loc[calib_idx, "subject"]) == target
        assert calib_idx.size > 0
        # ...and no trial of the target subject is in the training fold.
        assert target.isdisjoint(set(metadata.loc[train_idx, "subject"]))
        # Pairwise trial-disjoint: nothing is scored that was also fitted.
        assert np.intersect1d(train_idx, test_idx).size == 0
        assert np.intersect1d(train_idx, calib_idx).size == 0
        assert np.intersect1d(calib_idx, test_idx).size == 0
        # Calibration + test partition the target subject exactly.
        target_idx = metadata.index[metadata["subject"].isin(target)].to_numpy()
        assert np.array_equal(np.union1d(calib_idx, test_idx), target_idx)


def test_cross_subject_calibration_keeps_every_target_session(data):
    """Calibration must not consume an entire target session."""
    _, y, metadata = data
    cv_kwargs = {
        "cv_class": GroupShuffleSplit,
        "n_splits": 2,
        "test_size": 2,
        "random_state": 0,
    }
    baseline = list(CrossSubjectSplitter(**cv_kwargs).split(y, metadata))
    calibrated = list(
        CrossSubjectSplitter(calibration_size=0.5, **cv_kwargs).split(y, metadata)
    )

    for (_, base_test), (_, _calib, test) in zip(baseline, calibrated):
        assert set(
            metadata.loc[test, ["subject", "session"]].itertuples(index=False, name=None)
        ) == set(
            metadata.loc[base_test, ["subject", "session"]].itertuples(
                index=False, name=None
            )
        )
