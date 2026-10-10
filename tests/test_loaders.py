from data.loaders import load_csv


def test_load_csv_last_col_label(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("1,2,0\n3,4,0\n10,10,1\n", encoding="utf-8")
    ds = load_csv(str(p))
    assert ds.X.shape == (3, 2)
    assert ds.y.tolist() == [0, 0, 1]
    assert 0 < ds.contamination < 1


def test_load_csv_explicit_label_col(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("0,1,2\n1,3,4\n1,10,10\n", encoding="utf-8")
    ds = load_csv(str(p), label_col=0)
    assert ds.X.shape == (3, 2)
    assert ds.y.tolist() == [0, 1, 1]


def test_load_csv_skips_comments(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("# header\n1,2,0\n3,4,0\n", encoding="utf-8")
    ds = load_csv(str(p))
    assert ds.X.shape == (2, 2)
