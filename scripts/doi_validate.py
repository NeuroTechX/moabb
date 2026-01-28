#!/usr/bin/env python
"""Standalone DOI metadata validator for MOABB datasets.

This script validates DOI metadata in the MOABB catalog against authoritative
sources (Crossref, DataCite).

Requirements:
    pip install habanero requests

Usage:
    python scripts/doi_validate.py --output report.md
    python scripts/doi_validate.py --format json --output report.json
    python scripts/doi_validate.py --limit 10 --verbose
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional


# DOI format regex - must start with 10. followed by registrant code
DOI_PATTERN = re.compile(r"^10\.\d{4,9}/[^\s]+$")

# DOI prefixes that use DataCite instead of Crossref
DATACITE_PREFIXES = frozenset(
    {
        "10.5281",  # Zenodo
        "10.6084",  # Figshare
        "10.7910",  # Harvard Dataverse
        "10.13026",  # PhysioNet
        "10.48550",  # arXiv
        "10.34973",  # Donders Repository
        "10.6094",  # FreiDok (University of Freiburg)
        "10.71569",  # GESIS (Leibniz Institute)
    }
)


@dataclass
class DOIResolutionResult:
    """Result from resolving a single DOI."""

    doi: str
    source: str
    success: bool
    error: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[List[str]] = None
    publisher: Optional[str] = None
    raw_data: Optional[dict] = field(default=None, repr=False)


@dataclass
class ValidationResult:
    """Result from validating a dataset's metadata against DOI data."""

    dataset_name: str
    doi: Optional[str] = None
    issues: List[str] = field(default_factory=list)
    moabb_year: Optional[int] = None
    doi_year: Optional[int] = None
    year_match: bool = False
    author_match: str = "unknown"
    doi_resolved: bool = False
    doi_source: Optional[str] = None
    resolution_result: Optional[DOIResolutionResult] = field(default=None, repr=False)


def get_doi_source(doi: str) -> str:
    """Determine which API to use for a DOI based on its prefix."""
    parts = doi.split("/")
    if len(parts) >= 1:
        prefix = parts[0]
        if prefix in DATACITE_PREFIXES:
            return "datacite"
    return "crossref"


def validate_doi_format(doi: str) -> tuple:
    """Validate DOI format."""
    if not doi:
        return False, "DOI is empty or None"

    cleaned = doi
    for prefix in [
        "https://doi.org/",
        "http://doi.org/",
        "doi.org/",
        "doi:",
        "DOI:",
    ]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :]

    if not DOI_PATTERN.match(cleaned):
        return False, f"Invalid DOI format: '{doi}' (should match 10.XXXX/...)"

    return True, None


def extract_last_names(authors: Optional[List[str]]) -> set:
    """Extract last names from a list of author strings."""
    if not authors:
        return set()

    last_names = set()
    for author in authors:
        if not author:
            continue

        if "," in author:
            parts = author.split(",")
            last_names.add(parts[0].strip().lower())
        else:
            parts = author.strip().split()
            if parts:
                last_names.add(parts[-1].strip().lower())

    return last_names


def compare_authors(
    moabb_authors: Optional[List[str]], doi_authors: Optional[List[str]]
) -> str:
    """Compare author lists between MOABB and DOI metadata."""
    moabb_names = extract_last_names(moabb_authors)
    doi_names = extract_last_names(doi_authors)

    if not moabb_names or not doi_names:
        return "unknown"

    intersection = moabb_names & doi_names

    if not intersection:
        return "none"
    elif intersection == moabb_names or intersection == doi_names:
        return "full"
    else:
        return "partial"


class DOIResolver:
    """Resolves DOIs using Crossref (habanero) and DataCite APIs."""

    def __init__(
        self,
        email: Optional[str] = None,
        timeout: int = 30,
        rate_limit_delay: float = 0.1,
    ):
        self.email = email
        self.timeout = timeout
        self.rate_limit_delay = rate_limit_delay
        self._habanero = None
        self._last_request_time = 0.0

    def _get_habanero(self):
        """Lazily initialize habanero client."""
        if self._habanero is None:
            from habanero import Crossref

            self._habanero = Crossref(mailto=self.email, timeout=self.timeout)
        return self._habanero

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _clean_doi(self, doi: str) -> str:
        """Clean DOI by removing URL prefixes."""
        cleaned = doi
        for prefix in [
            "https://doi.org/",
            "http://doi.org/",
            "doi.org/",
            "doi:",
            "DOI:",
        ]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix) :]
        return cleaned

    def _resolve_crossref(self, doi: str) -> DOIResolutionResult:
        """Resolve DOI using Crossref API via habanero."""
        try:
            cr = self._get_habanero()
            self._rate_limit()
            result = cr.works(ids=doi)

            if not result or "message" not in result:
                return DOIResolutionResult(
                    doi=doi,
                    source="crossref",
                    success=False,
                    error="No message in Crossref response",
                )

            message = result["message"]

            title = None
            if "title" in message and message["title"]:
                title = (
                    message["title"][0]
                    if isinstance(message["title"], list)
                    else message["title"]
                )

            year = None
            for date_field in [
                "published-print",
                "published-online",
                "issued",
                "created",
            ]:
                if date_field in message and "date-parts" in message[date_field]:
                    date_parts = message[date_field]["date-parts"]
                    if date_parts and date_parts[0]:
                        year = date_parts[0][0]
                        break

            authors = []
            if "author" in message:
                for author in message["author"]:
                    if "family" in author:
                        if "given" in author:
                            authors.append(f"{author['family']}, {author['given']}")
                        else:
                            authors.append(author["family"])

            publisher = message.get("publisher")

            return DOIResolutionResult(
                doi=doi,
                source="crossref",
                success=True,
                title=title,
                year=year,
                authors=authors if authors else None,
                publisher=publisher,
                raw_data=message,
            )

        except Exception as e:
            return DOIResolutionResult(
                doi=doi,
                source="crossref",
                success=False,
                error=str(e),
            )

    def _resolve_datacite(self, doi: str) -> DOIResolutionResult:
        """Resolve DOI using DataCite API."""
        import requests

        try:
            self._rate_limit()
            url = f"https://api.datacite.org/dois/{doi}"
            response = requests.get(url, timeout=self.timeout)

            if response.status_code == 404:
                return DOIResolutionResult(
                    doi=doi,
                    source="datacite",
                    success=False,
                    error="DOI not found in DataCite",
                )

            response.raise_for_status()
            data = response.json()

            if "data" not in data or "attributes" not in data["data"]:
                return DOIResolutionResult(
                    doi=doi,
                    source="datacite",
                    success=False,
                    error="Invalid DataCite response structure",
                )

            attrs = data["data"]["attributes"]

            title = None
            if "titles" in attrs and attrs["titles"]:
                title = attrs["titles"][0].get("title")

            year = attrs.get("publicationYear")

            authors = []
            if "creators" in attrs:
                for creator in attrs["creators"]:
                    name = creator.get("name")
                    if name:
                        authors.append(name)
                    elif "familyName" in creator:
                        given = creator.get("givenName", "")
                        family = creator["familyName"]
                        if given:
                            authors.append(f"{family}, {given}")
                        else:
                            authors.append(family)

            publisher = attrs.get("publisher")

            return DOIResolutionResult(
                doi=doi,
                source="datacite",
                success=True,
                title=title,
                year=year,
                authors=authors if authors else None,
                publisher=publisher,
                raw_data=attrs,
            )

        except Exception as e:
            return DOIResolutionResult(
                doi=doi,
                source="datacite",
                success=False,
                error=str(e),
            )

    def resolve(self, doi: str) -> DOIResolutionResult:
        """Resolve a DOI using the appropriate API."""
        cleaned_doi = self._clean_doi(doi)
        is_valid, error = validate_doi_format(cleaned_doi)
        if not is_valid:
            return DOIResolutionResult(
                doi=doi,
                source="none",
                success=False,
                error=error,
            )

        source = get_doi_source(cleaned_doi)
        if source == "datacite":
            return self._resolve_datacite(cleaned_doi)
        else:
            return self._resolve_crossref(cleaned_doi)


class MetadataValidator:
    """Validates MOABB metadata against DOI-resolved sources."""

    def __init__(
        self,
        resolver: Optional[DOIResolver] = None,
        year_tolerance: int = 1,
    ):
        self.resolver = resolver or DOIResolver()
        self.year_tolerance = year_tolerance

    def validate_dataset(self, name: str, metadata) -> ValidationResult:
        """Validate a single dataset's metadata against DOI data."""
        issues = []

        if metadata.documentation is None:
            return ValidationResult(
                dataset_name=name,
                issues=["No documentation metadata available"],
            )

        doc = metadata.documentation
        doi = doc.doi

        if not doi:
            return ValidationResult(
                dataset_name=name,
                issues=["No DOI in metadata"],
                moabb_year=doc.publication_year,
            )

        is_valid, error = validate_doi_format(doi)
        if not is_valid:
            issues.append(error)

        resolution = self.resolver.resolve(doi)

        if not resolution.success:
            issues.append(f"DOI resolution failed: {resolution.error}")
            return ValidationResult(
                dataset_name=name,
                doi=doi,
                issues=issues,
                moabb_year=doc.publication_year,
                doi_resolved=False,
            )

        moabb_year = doc.publication_year
        doi_year = resolution.year
        year_match = False

        if moabb_year is not None and doi_year is not None:
            if abs(moabb_year - doi_year) <= self.year_tolerance:
                year_match = True
            else:
                issues.append(f"Year mismatch: MOABB={moabb_year}, DOI={doi_year}")
        elif moabb_year is None:
            issues.append("No publication_year in MOABB metadata")
        elif doi_year is None:
            issues.append("No year in DOI metadata")

        moabb_authors = doc.investigators
        doi_authors = resolution.authors
        author_match = compare_authors(moabb_authors, doi_authors)

        if author_match == "none":
            issues.append(f"No author overlap: MOABB={moabb_authors}, DOI={doi_authors}")

        return ValidationResult(
            dataset_name=name,
            doi=doi,
            issues=issues,
            moabb_year=moabb_year,
            doi_year=doi_year,
            year_match=year_match,
            author_match=author_match,
            doi_resolved=True,
            doi_source=resolution.source,
            resolution_result=resolution,
        )

    def validate_catalog(
        self, catalog: dict, progress_callback=None
    ) -> List[ValidationResult]:
        """Validate all datasets in the catalog."""
        results = []
        items = list(catalog.items())
        total = len(items)

        for i, (name, metadata) in enumerate(items):
            result = self.validate_dataset(name, metadata)
            results.append(result)
            if progress_callback:
                progress_callback(i + 1, total)

        return results


def generate_markdown_report(results: List[ValidationResult]) -> str:
    """Generate a Markdown report."""
    lines = []

    lines.append("# DOI Metadata Validation Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    total = len(results)
    with_doi = sum(1 for r in results if r.doi)
    resolved = sum(1 for r in results if r.doi_resolved)
    year_matches = sum(1 for r in results if r.year_match)
    no_issues = sum(1 for r in results if not r.issues)

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Datasets | {total} |")
    lines.append(f"| Datasets with DOI | {with_doi} |")
    lines.append(f"| DOIs Resolved | {resolved} |")
    lines.append(f"| Year Matches | {year_matches} |")
    lines.append(f"| No Issues | {no_issues} |")
    lines.append("")

    sources = {}
    for r in results:
        if r.doi_source:
            sources[r.doi_source] = sources.get(r.doi_source, 0) + 1

    if sources:
        lines.append("### DOI Sources")
        lines.append("")
        lines.append("| Source | Count |")
        lines.append("|--------|-------|")
        for source, count in sorted(sources.items()):
            lines.append(f"| {source} | {count} |")
        lines.append("")

    issues_results = [r for r in results if r.issues]
    if issues_results:
        lines.append("## Issues Found")
        lines.append("")

        year_issues = [
            r for r in results if r.moabb_year and r.doi_year and not r.year_match
        ]
        if year_issues:
            lines.append("### Year Mismatches")
            lines.append("")
            lines.append("| Dataset | MOABB Year | DOI Year |")
            lines.append("|---------|------------|----------|")
            for r in year_issues:
                lines.append(f"| {r.dataset_name} | {r.moabb_year} | {r.doi_year} |")
            lines.append("")

        resolution_failures = [r for r in results if r.doi and not r.doi_resolved]
        if resolution_failures:
            lines.append("### DOI Resolution Failures")
            lines.append("")
            lines.append("| Dataset | DOI | Error |")
            lines.append("|---------|-----|-------|")
            for r in resolution_failures:
                error = next(
                    (i for i in r.issues if "resolution failed" in i.lower()),
                    "Unknown",
                )
                lines.append(f"| {r.dataset_name} | {r.doi} | {error} |")
            lines.append("")

        missing_dois = [r for r in results if not r.doi]
        if missing_dois:
            lines.append("### Missing DOIs")
            lines.append("")
            for r in missing_dois:
                lines.append(f"- {r.dataset_name}")
            lines.append("")

        author_issues = [
            r
            for r in results
            if r.author_match == "none" and r not in resolution_failures
        ]
        if author_issues:
            lines.append("### Author Mismatches")
            lines.append("")
            for r in author_issues:
                lines.append(f"#### {r.dataset_name}")
                for issue in r.issues:
                    if "author" in issue.lower():
                        lines.append(f"- {issue}")
                lines.append("")

    successful = [r for r in results if not r.issues]
    if successful:
        lines.append("## Successfully Validated")
        lines.append("")
        lines.append("| Dataset | DOI | Year | Authors |")
        lines.append("|---------|-----|------|---------|")
        for r in successful:
            lines.append(
                f"| {r.dataset_name} | {r.doi or 'N/A'} | "
                f"{r.moabb_year or 'N/A'} | {r.author_match} |"
            )
        lines.append("")

    return "\n".join(lines)


def generate_json_report(results: List[ValidationResult]) -> str:
    """Generate a JSON report."""
    total = len(results)
    with_doi = sum(1 for r in results if r.doi)
    resolved = sum(1 for r in results if r.doi_resolved)
    year_matches = sum(1 for r in results if r.year_match)
    no_issues = sum(1 for r in results if not r.issues)

    sources = {}
    for r in results:
        if r.doi_source:
            sources[r.doi_source] = sources.get(r.doi_source, 0) + 1

    report = {
        "metadata": {
            "generated": datetime.now().isoformat(),
            "format_version": "1.0.0",
        },
        "summary": {
            "total_datasets": total,
            "datasets_with_doi": with_doi,
            "dois_resolved": resolved,
            "year_matches": year_matches,
            "no_issues": no_issues,
            "doi_sources": sources,
        },
        "validation_results": [
            {
                "dataset_name": r.dataset_name,
                "doi": r.doi,
                "issues": r.issues,
                "moabb_year": r.moabb_year,
                "doi_year": r.doi_year,
                "year_match": r.year_match,
                "author_match": r.author_match,
                "doi_resolved": r.doi_resolved,
                "doi_source": r.doi_source,
            }
            for r in results
        ],
    }

    return json.dumps(report, indent=2)


def main():
    """Run DOI metadata validation."""
    parser = argparse.ArgumentParser(
        description="Validate MOABB dataset DOI metadata against Crossref/DataCite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output report.md
  %(prog)s --format json --output report.json
  %(prog)s --email user@example.com --verbose
  %(prog)s --limit 10 --verbose
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    parser.add_argument(
        "--email",
        "-e",
        type=str,
        help="Email for Crossref polite pool (higher rate limits)",
    )

    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        help="Limit number of datasets to validate (for testing)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show progress during validation",
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)",
    )

    parser.add_argument(
        "--rate-limit",
        type=float,
        default=0.1,
        help="Delay between requests in seconds (default: 0.1)",
    )

    parser.add_argument(
        "--year-tolerance",
        type=int,
        default=1,
        help="Tolerance for year matching, e.g., 1 means +/-1 year (default: 1)",
    )

    args = parser.parse_args()

    # Check for habanero
    try:
        import habanero  # noqa: F401
    except ImportError:
        print(
            "Error: habanero is required for DOI validation.",
            file=sys.stderr,
        )
        print(
            "Install it with: pip install habanero",
            file=sys.stderr,
        )
        sys.exit(1)

    # Import MOABB catalog
    try:
        from moabb.datasets.metadata import DATASET_METADATA_CATALOG
    except ImportError as e:
        print(f"Error importing MOABB: {e}", file=sys.stderr)
        print(
            "Make sure you are in the moabb directory and have installed dependencies.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply limit if specified
    catalog = dict(DATASET_METADATA_CATALOG)
    if args.limit:
        catalog = dict(list(catalog.items())[: args.limit])

    # Create resolver and validator
    resolver = DOIResolver(
        email=args.email,
        timeout=args.timeout,
        rate_limit_delay=args.rate_limit,
    )

    validator = MetadataValidator(
        resolver=resolver,
        year_tolerance=args.year_tolerance,
    )

    # Progress callback
    def progress_callback(current, total):
        if args.verbose:
            print(f"\rValidating: {current}/{total}", end="", file=sys.stderr)

    # Run validation
    if args.verbose:
        print("Starting DOI metadata validation...", file=sys.stderr)

    results = validator.validate_catalog(
        catalog,
        progress_callback=progress_callback if args.verbose else None,
    )

    if args.verbose:
        print(file=sys.stderr)
        print(f"Validated {len(results)} datasets", file=sys.stderr)

    # Generate report
    if args.format == "json":
        report = generate_json_report(results)
    else:
        report = generate_markdown_report(results)

    # Output
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(report, encoding="utf-8")
        if args.verbose:
            print(f"Report written to: {output_path}", file=sys.stderr)
    else:
        print(report)

    # Exit with non-zero status if there are issues
    issues_count = sum(1 for r in results if r.issues)
    if issues_count > 0:
        if args.verbose:
            print(f"Found {issues_count} datasets with issues", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
