"""Data explorer endpoints for CRM tables."""

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
    records: list[dict[str, Any]] = df.to_dict("records")  # type: ignore[assignment]
    return records, df.columns.tolist()


@router.get("/data/companies", response_model=DataResponse, summary="Get all companies")
async def get_companies() -> DataResponse:
    data, columns = load_csv("companies.csv")
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/contacts", response_model=DataResponse, summary="Get all contacts")
async def get_contacts() -> DataResponse:
    data, columns = load_csv("contacts.csv")
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/opportunities", response_model=DataResponse, summary="Get all opportunities")
async def get_opportunities() -> DataResponse:
    data, columns = load_csv("opportunities.csv")
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/activities", response_model=DataResponse, summary="Get all activities")
async def get_activities() -> DataResponse:
    data, columns = load_csv("activities.csv")
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/history", response_model=DataResponse, summary="Get all history")
async def get_history() -> DataResponse:
    data, columns = load_csv("history.csv")
    return DataResponse(data=data, total=len(data), columns=columns)
