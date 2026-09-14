import pytest

from moabb.run import parser_init


@pytest.mark.parametrize(
    "argv,expected",
    [
        (["-e", "WithinSession"], ["WithinSession"]),
        (["-e", "WithinSession", "CrossSession"], ["WithinSession", "CrossSession"]),
        ([], None),
    ],
)
def test_parser_evaluations(argv, expected):
    assert parser_init().parse_args(argv).evaluations == expected
