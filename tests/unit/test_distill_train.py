from __future__ import annotations

import json
from typing import TYPE_CHECKING

from firewall.judge.distill.train import prepare_mlx_data, to_mlx_records

if TYPE_CHECKING:
    from pathlib import Path


def _record(text: str, decision: str) -> dict[str, object]:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps({"decision": decision})},
        ],
        "meta": {"text": text, "decision": decision, "provenance": "gen"},
    }


def test_to_mlx_records_keeps_only_messages() -> None:
    out = to_mlx_records([_record("a", "BLOCK")])
    assert out == [
        {
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": json.dumps({"decision": "BLOCK"})},
            ]
        }
    ]
    assert "meta" not in out[0]  # meta is training-irrelevant and must not leak in


def test_prepare_mlx_data_writes_train_and_valid(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "train.jsonl").write_text(
        "\n".join(json.dumps(_record(f"t{i}", "BLOCK")) for i in range(3)) + "\n"
    )
    (src / "val.jsonl").write_text(json.dumps(_record("v0", "PASS")) + "\n")

    out = tmp_path / "mlx"
    counts = prepare_mlx_data(src, out)

    assert counts == {"train": 3, "valid": 1}
    # mlx-lm requires the file named `valid.jsonl`, not `val.jsonl`.
    assert (out / "train.jsonl").exists()
    assert (out / "valid.jsonl").exists()
    assert not (out / "val.jsonl").exists()
    first = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert set(first) == {"messages"}
    assert [m["role"] for m in first["messages"]] == ["system", "user", "assistant"]
