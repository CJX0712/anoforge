import subprocess

PY = "C:/Users/Administrator/.workbuddy/binaries/python/envs/anoforge/Scripts/python.exe"
ROOT = "."


def test_cli_list():
    r = subprocess.run(
        [PY, "cli.py", "list"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8"
    )
    assert r.returncode == 0, r.stderr
    assert "可用检测器" in r.stdout


def test_cli_demo_smoke():
    r = subprocess.run(
        [PY, "cli.py", "demo", "--nseeds", "2", "--out", "_cli_bench.json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert r.returncode == 0, r.stderr
    assert "AnoFuse" in r.stdout
