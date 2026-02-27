import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dl-data",
        action="store_true",
        help="Run the download tests. This requires an internet connection and can take a long time.",
    )
    parser.addoption(
        "--update-doi-cache",
        action="store_true",
        default=False,
        help="Re-resolve all DOIs from network and update the persistent cache file.",
    )


@pytest.fixture
def dl_data(request):
    return request.config.getoption("--dl-data")


@pytest.fixture(autouse=True, scope="session")
def _handle_update_doi_cache(request):
    if request.config.getoption("--update-doi-cache"):
        import moabb.tests.test_doi_validation as mod

        mod._UPDATE_DOI_CACHE = True
        mod._DOI_CACHE.clear()


def pytest_sessionfinish(session, exitstatus):
    """Auto-save DOI cache to disk and print resolution summary."""
    try:
        import moabb.tests.test_doi_validation as mod

        if mod._DOI_CACHE_DIRTY:
            mod._save_doi_cache()

        newly = mod._NEWLY_RESOLVED_DOIS
        if newly:
            lines = [
                "",
                "=" * 55,
                "DOI cache updated",
                "=" * 55,
                f"Resolved and cached {len(newly)} new DOI(s):",
            ]
            for doi in newly:
                entry = mod._DOI_CACHE.get(doi, {}) or {}
                title = entry.get("title", "Unknown title")
                lines.append(f"  + {doi}  ({title})")
            lines.append(f"Cache file: {mod._DOI_CACHE_PATH}")
            lines.append("=" * 55)
            print("\n".join(lines))
    except (ImportError, AttributeError):
        pass


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--dl-data"):
        skip_download = pytest.mark.skip(reason="need --dl-data option to run")
        for item in items:
            if "download" in item.keywords:
                item.add_marker(skip_download)
