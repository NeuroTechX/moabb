from moabb.datasets import BNCI2014_001, BNCI2014_008


def test_bnci2014_001_metadata():
    """Test metadata for BNCI2014-001."""
    dataset = BNCI2014_001()
    subject = 1
    session_name = list(dataset.get_data(subjects=[subject])[subject].keys())[0]
    raw = dataset.get_data(subjects=[subject])[subject][session_name]["0"]

    assert "birthday" in raw.info["subject_info"]
    assert "sex" in raw.info["subject_info"]
    assert "hand" in raw.info["subject_info"]
    assert raw.info["subject_info"]["sex"] in [1, 2]
    assert raw.info["meas_date"].year == 2008
    assert raw.get_montage() is not None
    assert raw.info["line_freq"] == 50.0


def test_bnci2014_008_metadata():
    """Test metadata for BNCI2014-008."""
    dataset = BNCI2014_008()
    subject = 1
    session_name = list(dataset.get_data(subjects=[subject])[subject].keys())[0]
    raw = dataset.get_data(subjects=[subject])[subject][session_name]["0"]

    assert "birthday" in raw.info["subject_info"]
    assert "sex" in raw.info["subject_info"]
    assert "hand" in raw.info["subject_info"]
    assert raw.info["subject_info"]["sex"] in [1, 2]
    assert raw.info["meas_date"].year == 2012
    assert raw.get_montage() is not None
    assert raw.info["line_freq"] == 50.0
    assert "ALSfrs" in raw.info["description"]
