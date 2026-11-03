import io

import pandas as pd
import pytest

from analyst.data.ingest import infer_spec, load_dataset

FRAME = pd.DataFrame({"Customer": ["Müller", "Zoë"], "Sales": [1.5, 2.0]})


def test_csv_falls_back_to_cp1252(tmp_path):
    path = tmp_path / "legacy.csv"
    FRAME.to_csv(path, index=False, encoding="cp1252")
    assert load_dataset(path)["Customer"].tolist() == ["Müller", "Zoë"]


def test_csv_with_utf8_bom(tmp_path):
    path = tmp_path / "bom.csv"
    FRAME.to_csv(path, index=False, encoding="utf-8-sig")
    assert list(load_dataset(path).columns) == ["Customer", "Sales"]


def test_file_object_uses_filename_and_retries_encoding():
    buf = io.BytesIO(FRAME.to_csv(index=False).encode("cp1252"))
    assert load_dataset(buf, filename="upload.csv")["Customer"].iloc[0] == "Müller"


def test_excel_round_trip(tmp_path):
    path = tmp_path / "data.xlsx"
    with pd.ExcelWriter(path) as writer:
        FRAME.to_excel(writer, sheet_name="first", index=False)
        FRAME.assign(Sales=0.0).to_excel(writer, sheet_name="second", index=False)
    assert load_dataset(path)["Sales"].tolist() == [1.5, 2.0]
    assert load_dataset(path, sheet_name="second")["Sales"].tolist() == [0.0, 0.0]


def test_infer_spec_excludes_identifiers():
    raw = pd.DataFrame({
        "Row ID": [1, 2, 3],
        "Order Date": ["11/8/2016", "11/9/2016", "11/10/2016"],
        "Postal Code": [42420, 1040, 90036],
        "Sales": [1.0, 2.5, 3.0],
    })
    spec = infer_spec(raw)
    assert spec.timestamp_col == "Order Date"
    assert spec.target_cols == ["Sales"]


def test_unsupported_suffix(tmp_path):
    with pytest.raises(ValueError, match="unsupported"):
        load_dataset(tmp_path / "data.json")
