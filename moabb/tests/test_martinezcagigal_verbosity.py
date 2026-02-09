import inspect
import logging

from moabb.datasets.martinezcagigal2023_pary_cvep import MartinezCagigal2023Pary


def test_verbosity():
    # Setup logger to capture output
    logger = logging.getLogger("moabb.datasets.martinezcagigal2023_pary_cvep")
    logger.setLevel(logging.INFO)

    MartinezCagigal2023Pary()

    # Verify the source file uses logging instead of print statements
    source_file = inspect.getfile(MartinezCagigal2023Pary)
    with open(source_file, "r") as f:
        content = f.read()
        assert "print(" not in content or 'if __name__ == "__main__":' in content
        assert "log.info(" in content
        assert "log.error(" in content
