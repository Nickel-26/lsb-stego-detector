from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_streamlit_app_executes_without_exception():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    titles = [t.value for t in at.title]
    assert any("LSB" in t for t in titles)
