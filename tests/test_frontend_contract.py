"""Static contract checks: local assets resolve and only documented APIs are called."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FRONTEND = Path(__file__).resolve().parents[1] / "frontend"
HTML_FILES = sorted(FRONTEND.glob("*.html"))

REFERENCE = re.compile(r'(?:src|href)="([^"]+)"')
ES_IMPORT = re.compile(r'from\s+"(\./[^"]+)"')
API_CALL = re.compile(r'request\(\s*[`"](/[\w\-/?=&{}$.]*)')
SUPPORTED_PATHS = {
    "/health",
    "/model-info",
    "/statistics",
    "/history",
    "/predict",
    "/feedback",
}


def iter_local_references(html: str):
    for match in REFERENCE.finditer(html):
        value = match.group(1)
        if value.startswith(("http://", "https://", "#", "mailto:", "/", "data:")):
            continue
        yield value.split("#", 1)[0].split("?", 1)[0]


@pytest.mark.parametrize("html_path", HTML_FILES, ids=lambda path: path.name)
def test_local_assets_exist(html_path):
    html = html_path.read_text(encoding="utf-8")
    missing = [
        reference
        for reference in iter_local_references(html)
        if reference and not (FRONTEND / reference).exists()
    ]
    assert not missing, f"{html_path.name} references missing files: {missing}"


def test_es_module_imports_resolve():
    missing: list[str] = []
    for script in sorted((FRONTEND / "js").glob("*.js")):
        for relative in ES_IMPORT.findall(script.read_text(encoding="utf-8")):
            target = (script.parent / relative).resolve()
            if not target.exists():
                missing.append(f"{script.name} -> {relative}")
    assert not missing, missing


def test_only_supported_api_paths_are_called():
    source = (FRONTEND / "js" / "api.js").read_text(encoding="utf-8")
    called = {path.split("?", 1)[0] for path in API_CALL.findall(source)}
    assert called, "No API calls detected in api.js"
    assert called <= SUPPORTED_PATHS, f"Undocumented API paths: {sorted(called - SUPPORTED_PATHS)}"


def test_no_cdn_script_dependencies():
    for html_path in HTML_FILES:
        html = html_path.read_text(encoding="utf-8")
        for match in re.finditer(r'<script[^>]+src="([^"]+)"', html):
            assert not match.group(1).startswith(("http://", "https://")), (
                f"{html_path.name} loads a remote script: {match.group(1)}"
            )


def test_expected_pages_exist():
    expected = {"index.html", "analyzer.html", "dashboard.html", "about.html"}
    assert {path.name for path in HTML_FILES} == expected


def test_headline_present():
    landing = (FRONTEND / "index.html").read_text(encoding="utf-8")
    assert "Turn Waste Into a" in landing
    assert "Sustainable Future" in landing


@pytest.mark.parametrize("html_path", HTML_FILES, ids=lambda path: path.name)
def test_theme_bootstrap_and_toggle(html_path):
    html = html_path.read_text(encoding="utf-8")
    script = re.search(r'<script[^>]+src="js/theme\.js"', html)
    assert script, f"{html_path.name} does not load js/theme.js"
    first_stylesheet = html.find('rel="stylesheet"')
    assert first_stylesheet == -1 or script.start() < first_stylesheet, (
        f"{html_path.name} loads theme.js after the stylesheet, which causes a flash of the wrong theme"
    )
    assert html.count("data-theme-toggle") == 1, (
        f"{html_path.name} should expose exactly one theme toggle button"
    )
    assert 'aria-label="Switch to' in html


def test_theme_icons_defined():
    icons = (FRONTEND / "assets" / "icons.svg").read_text(encoding="utf-8")
    for symbol in ("icon-sun", "icon-moon"):
        assert f'id="{symbol}"' in icons
