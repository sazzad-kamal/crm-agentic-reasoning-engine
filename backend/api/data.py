"""Data explorer endpoints for CRM tables."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

CSV_DIR = Path(__file__).parent.parent / "data" / "csv"


class DataResponse(BaseModel):
    data: list[dict[str, Any]]
    total: int
    columns: list[str]


def load_csv(name: str) -> tuple[list[dict[str, Any]], list[str]]:
    path = CSV_DIR / name
    if not path.exists():
        return [], []
    df = pd.read_csv(path)
    return df.to_dict("records"), df.columns.tolist()


def load_jsonl(name: str) -> list[dict[str, Any]]:
    path = CSV_DIR / name
    if not path.exists():
        return []
    df = pd.read_json(path, lines=True)
    if "metadata" in df.columns:
        meta_df = pd.json_normalize(df["metadata"]).add_prefix("metadata_")
        df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
    return df.to_dict("records")  # type: ignore[no-any-return]


def _group_by(data: list[dict], key: str) -> dict[str, list[dict]]:
    result = defaultdict(list)
    for record in data:
        if k := record.get(key):
            result[k].append(record)
    return dict(result)


@router.get("/data/companies", response_model=DataResponse, summary="Get all companies")
async def get_companies() -> DataResponse:
    data, columns = load_csv("companies.csv")
    texts = _group_by(load_jsonl("texts.jsonl"), "metadata_company_id")
    for row in data:
        row["_private_texts"] = texts.get(row.get("company_id", ""), [])
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/contacts", response_model=DataResponse, summary="Get all contacts")
async def get_contacts() -> DataResponse:
    data, columns = load_csv("contacts.csv")
    texts = _group_by(load_jsonl("texts.jsonl"), "metadata_contact_id")
    for row in data:
        row["_private_texts"] = texts.get(row.get("contact_id", ""), [])
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/opportunities", response_model=DataResponse, summary="Get all opportunities")
async def get_opportunities() -> DataResponse:
    data, columns = load_csv("opportunities.csv")
    texts = _group_by(load_jsonl("texts.jsonl"), "metadata_opportunity_id")
    for row in data:
        row["_private_texts"] = texts.get(row.get("opportunity_id", ""), [])
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/activities", response_model=DataResponse, summary="Get all activities")
async def get_activities() -> DataResponse:
    data, columns = load_csv("activities.csv")
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/history", response_model=DataResponse, summary="Get all history")
async def get_history() -> DataResponse:
    data, columns = load_csv("history.csv")
    return DataResponse(data=data, total=len(data), columns=columns)
