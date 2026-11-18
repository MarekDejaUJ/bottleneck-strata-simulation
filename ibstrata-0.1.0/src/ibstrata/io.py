"""Reading the data export of a Card Surveys survey (format ``cardsurveys-export-1``)."""

from __future__ import annotations

import io as _io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TABLES = ("responses", "answers", "blocks", "pairs", "screens", "allocations")


@dataclass
class SurveyExport:
    """The tables and the description of one survey export."""

    description: dict
    survey: dict
    responses: pd.DataFrame
    answers: pd.DataFrame
    blocks: pd.DataFrame
    pairs: pd.DataFrame
    screens: pd.DataFrame
    allocations: pd.DataFrame
    extra: dict = field(default_factory=dict)

    def submitted(self) -> SurveyExport:
        """The export restricted to submitted responses."""
        ids = set(self.responses.loc[self.responses["status"] == "submitted", "response_uuid"])
        keep = {}
        for name in TABLES:
            df = getattr(self, name)
            keep[name] = df[df["response_uuid"].isin(ids)].copy() if "response_uuid" in df else df
        return SurveyExport(self.description, self.survey, **keep, extra=self.extra)

    def module(self, key: str) -> dict:
        """The bundle definition of a module."""
        for m in self.survey.get("modules", []):
            if m.get("key") == key:
                return m
        raise KeyError(f"no module {key!r} in the survey")


def read_cardsurveys(path: str | Path) -> SurveyExport:
    """Read an export folder or ZIP file."""
    path = Path(path)
    files: dict[str, bytes] = {}
    if path.is_dir():
        for p in path.rglob("*"):
            if p.is_file():
                files[p.relative_to(path).as_posix()] = p.read_bytes()
    else:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if not name.endswith("/"):
                    files[name.split("/", 1)[1] if _single_root(z) else name] = z.read(name)
    desc = json.loads(files["description.json"].decode("utf-8"))
    if desc.get("format") != "cardsurveys-export-1":
        raise ValueError(f"unknown export format {desc.get('format')!r}")
    survey = json.loads(files["survey.json"].decode("utf-8")) if "survey.json" in files else {}
    tables = {}
    for name in TABLES:
        raw = files.get(f"{name}.csv")
        tables[name] = (pd.read_csv(_io.BytesIO(raw), dtype=str, keep_default_na=False)
                        if raw is not None else pd.DataFrame())
    extra = {k: v for k, v in files.items()
             if k not in {"description.json", "survey.json"} | {f"{t}.csv" for t in TABLES}}
    return SurveyExport(desc, survey, **tables, extra=extra)


def _single_root(z: zipfile.ZipFile) -> bool:
    roots = {n.split("/", 1)[0] for n in z.namelist() if "/" in n}
    plain = [n for n in z.namelist() if "/" not in n]
    return len(roots) == 1 and not plain
