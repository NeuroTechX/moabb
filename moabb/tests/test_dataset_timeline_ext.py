import re

from docs.source.sphinxext import dataset_timeline_ext as ext


def test_extract_description_skips_directive_blocks():
    lines = [
        "Intro paragraph.",
        "",
        ".. note::",
        "   Keep this out of the teaser.",
        "",
        ".. code-block:: text",
        "",
        "   hidden code",
        "",
        "Closing sentence.",
        "",
        ".. rubric:: References",
        "",
        "[1] ref",
    ]

    assert ext._extract_description_text(lines) == [
        "Intro paragraph.",
        "",
        "Closing sentence.",
    ]


def test_overview_teaser_uses_stable_overview_anchor():
    html = ext._make_overview_teaser_html(["Intro paragraph."], "SampleDataset")

    assert ".. note::" not in html
    assert 'href="#ds-overview-sampledataset"' in html
    assert "document.querySelector('.ds-doc-tabs .sd-tab-label')" in html


def test_restructure_docstring_adds_tabset_class_and_anchor():
    lines = ["Intro paragraph.", "", ".. admonition:: Dataset summary", "", "   body"]

    result = ext._restructure_docstring_lines(lines, "SampleDataset")

    assert result is not None
    assert "   :class: ds-doc-tabs" in result
    assert "      .. _ds-overview-sampledataset:" in result


def test_citation_impact_hides_empty_pageviews():
    html = ext._make_citation_impact_html(
        {"paper_doi": "10.1000/test"},
        {},
        live_citations=False,
        pageview_counts={},
        pageview_rank={},
        pageview_meta={},
    )

    assert "Paper DOI" in html
    assert "Page Views" not in html


def test_citation_impact_shows_zero_pageviews_when_dataset_exists():
    html = ext._make_citation_impact_html(
        {"paper_doi": "10.1000/test"},
        {},
        live_citations=False,
        pageview_counts={"last30": 0, "all_time": 0, "weekly_12": [0, 0]},
        pageview_rank={"rank": 3, "total": 10, "top_percent": 30},
        pageview_meta={"generated_at_utc": "2026-03-05T12:00:00Z"},
    )

    assert "Page Views" in html
    assert "30d: <strong>0</strong>" in html
    assert "all-time: <strong>0</strong>" in html


def test_header_html_quickstart_button_reveals_code_panel():
    html = ext._make_header_html(
        "SampleDataset",
        {"paradigm": "cvep", "default_subject": 7},
        live_citations=False,
        pageview_counts={},
        pageview_rank={},
        pageview_meta={},
    )

    assert 'id="ds-quickstart-btn-sampledataset"' in html
    assert 'aria-controls="ds-quickstart-sampledataset"' in html
    assert 'aria-labelledby="ds-quickstart-btn-sampledataset"' in html
    assert 'aria-hidden="true" hidden' in html
    assert "Toggle quickstart code" not in html
    text = re.sub(r"<[^>]+>", "", html)
    assert "from moabb.datasets import SampleDataset" in text
    assert "dataset = SampleDataset()" in text
    assert "data = dataset.get_data(subjects=[7])" in text
