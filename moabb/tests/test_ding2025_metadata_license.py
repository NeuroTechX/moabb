from moabb.datasets.ding2025 import Ding2025


def test_ding2025_kilthub_license_is_cc_by_nc_40():
    assert Ding2025.METADATA.documentation.license == "CC-BY-NC-4.0"
