import subprocess
import sys
import urllib.error
from pathlib import Path

from scripts import build_wafct_extract as wafct

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "knowledge_base" / "data" / "wafct2019_selected.csv"
MD = ROOT / "knowledge_base" / "docs" / "wafct2019_ghana_foods.md"


def test_committed_extract_skips_download_and_rebuild(capsys):
    csv_before, md_before = CSV.read_bytes(), MD.read_bytes()
    csv_mtime, md_mtime = CSV.stat().st_mtime_ns, MD.stat().st_mtime_ns
    assert wafct.main(force=False) == 0
    out = capsys.readouterr().out
    assert "skipping download and rebuild" in out
    assert CSV.read_bytes() == csv_before and MD.read_bytes() == md_before
    assert CSV.stat().st_mtime_ns == csv_mtime and MD.stat().st_mtime_ns == md_mtime


def test_cli_skips_when_extract_exists():
    proc = subprocess.run(
        [sys.executable, "scripts/build_wafct_extract.py"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "skipping download and rebuild" in proc.stdout
    assert proc.stderr == ""


def _forbid(req, timeout=120):
    assert "Mozilla" in (req.get_header("User-agent") or "")
    raise urllib.error.HTTPError(wafct.URL, 403, "Forbidden", hdrs=None, fp=None)


def test_download_sends_browser_user_agent(monkeypatch, tmp_path):
    seen = {}

    class Resp:
        def read(self, n=-1):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(req, timeout=120):
        seen["ua"] = req.get_header("User-agent")
        return Resp()

    monkeypatch.setattr(wafct.urllib.request, "urlopen", urlopen)
    dest = tmp_path / "WAFCT_2019.xlsx"
    wafct.download_xlsx(dest)
    assert "Mozilla/5.0" in seen["ua"]
    assert dest.exists() and not dest.with_suffix(".xlsx.part").exists()


def test_failed_download_exits_0_when_csv_exists(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wafct, "ROOT", tmp_path)
    csv = tmp_path / "knowledge_base" / "data" / "wafct2019_selected.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("kept\n", encoding="utf-8")
    monkeypatch.setattr(wafct.urllib.request, "urlopen", _forbid)
    assert wafct.main(force=True) == 0
    err = capsys.readouterr()
    assert "Warning:" in err.out and "403" in err.out
    assert csv.read_text(encoding="utf-8") == "kept\n"
    assert not (tmp_path / "data" / "raw" / "WAFCT_2019.xlsx").exists()
    assert not (tmp_path / "data" / "raw" / "WAFCT_2019.xlsx.part").exists()


def test_failed_download_exits_nonzero_when_csv_missing(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(wafct, "ROOT", tmp_path)
    monkeypatch.setattr(wafct.urllib.request, "urlopen", _forbid)
    assert wafct.main(force=False) == 1
    err = capsys.readouterr()
    assert "Error:" in err.err and "missing" in err.err
    assert not (tmp_path / "knowledge_base" / "data" / "wafct2019_selected.csv").exists()
