"""Soft guard on the test split (spec §9).

`data.schema.load_test` requires a `TestAccess` token; only `authorize(run)` issues one, and only for a
finished run tagged `final=true`. Training code holds no token. Each test evaluation is counted and logged.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_ISSUER = object()
TEST_LOG = Path("eval") / "test" / "log.jsonl"


class TestAccess:
    """Capability to read test.parquet for one run."""

    __slots__ = ("run_id", "_issuer")

    def __init__(self, run_id: str, _issuer: object = None) -> None:
        if _issuer is not _ISSUER:
            raise PermissionError("TestAccess tokens are issued only by cp4285_proj.evaluation.guard.authorize(run)")
        self.run_id = run_id
        self._issuer = _issuer


def check_token(token: Any) -> None:
    if not isinstance(token, TestAccess) or token._issuer is not _ISSUER:
        raise PermissionError("reading test.parquet requires a TestAccess token from evaluation.guard.authorize(run)")


def authorize(run: Any) -> TestAccess:
    """`run` is an MLflow Run. Requires tag final=true and status FINISHED (so no run can authorise itself mid-training)."""
    run_id = run.info.run_id
    if run.data.tags.get("final") != "true":
        raise PermissionError(f"run {run_id} is not a final run (tag final=true required) and may not be evaluated on test")
    if run.info.status != "FINISHED":
        raise PermissionError(f"run {run_id} has status {run.info.status}; only finished runs may be evaluated on test")
    return TestAccess(run_id, _ISSUER)


def previous_test_evals(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / TEST_LOG
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def warn_if_evaluated(run: Any, run_dir: Path) -> None:
    count = int(run.data.tags.get("test_eval_count", "0"))
    if count > 0:
        lines = [f"  - {e['timestamp']}  panel {e['panel_hash'][:12]}  reason: {e.get('reason') or '-'}" for e in previous_test_evals(run_dir)]
        print(f"WARNING: run {run.info.run_id} has already been evaluated on test {count} time(s):\n" + "\n".join(lines))


def record_test_eval(client: Any, run: Any, run_dir: Path, panel_hash: str, reason: str | None) -> int:
    """Increment the MLflow tag `test_eval_count` and append to eval/test/log.jsonl. Returns the new count."""
    count = int(client.get_run(run.info.run_id).data.tags.get("test_eval_count", "0")) + 1
    client.set_tag(run.info.run_id, "test_eval_count", str(count))
    path = run_dir / TEST_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "panel_hash": panel_hash, "reason": reason, "test_eval_count": count}
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return count
