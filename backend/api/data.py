"""Data explorer endpoints for CRM tables."""

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.core.config import get_settings, Settings

router = APIRouter()


class DataResponse(BaseModel):
    data: list[dict[str, Any]]
    total: int
    columns: list[str]


def load_csv_data(csv_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load data from a CSV file."""
    if not csv_path.exists():
        return [], []
    df = pd.read_csv(csv_path)
    return df.to_dict("records"), df.columns.tolist()


def load_jsonl_data(jsonl_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load data from a JSONL file, flattening metadata if present."""
    if not jsonl_path.exists():
        return [], []
    df = pd.read_json(jsonl_path, lines=True)
    if "metadata" in df.columns:
        meta_df = pd.json_normalize(df["metadata"]).add_prefix("metadata_")
        df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
    return df.to_dict("records"), df.columns.tolist()


def _group_by_key(data: list[dict], key_field: str, extract_field: str | None = None) -> dict[str, list[dict]]:
    """Group records by a key field."""
    result: dict[str, list[dict]] = {}
    for record in data:
        key = record.get(extract_field or key_field) or record.get(key_field, "")
        if key:
            result.setdefault(key, []).append(record)
    return result


def _create_simple_data_endpoint(file_name: str, is_jsonl: bool = False):
    """Factory for simple data endpoints."""
    async def endpoint(settings: Settings = Depends(get_settings)) -> DataResponse:
        path = settings.data_dir / "csv" / file_name
        loader = load_jsonl_data if is_jsonl else load_csv_data
        data, columns = loader(path)
        return DataResponse(data=data, total=len(data), columns=columns)
    return endpoint


@router.get("/data/companies", response_model=DataResponse, summary="Get all companies")
async def get_companies(settings: Settings = Depends(get_settings)) -> DataResponse:
    """Get all company data with nested private texts."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "companies.csv")
    private_texts, _ = load_jsonl_data(settings.data_dir / "csv" / "private_texts.jsonl")
    texts_by_company = _group_by_key(private_texts, "company_id", "metadata_company_id")
    for company in data:
        company["_private_texts"] = texts_by_company.get(company.get("company_id", ""), [])
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/contacts", response_model=DataResponse, summary="Get all contacts")
async def get_contacts(settings: Settings = Depends(get_settings)) -> DataResponse:
    """Get all contact data with nested private texts."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "contacts.csv")
    private_texts, _ = load_jsonl_data(settings.data_dir / "csv" / "private_texts.jsonl")
    texts_by_contact = _group_by_key(private_texts, "contact_id", "metadata_contact_id")
    for contact in data:
        contact["_private_texts"] = texts_by_contact.get(contact.get("contact_id", ""), [])
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/opportunities", response_model=DataResponse, summary="Get all opportunities")
async def get_opportunities(settings: Settings = Depends(get_settings)) -> DataResponse:
    """Get all opportunity data with descriptions and attachments."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "opportunities.csv")
    descriptions, _ = load_csv_data(settings.data_dir / "csv" / "opportunity_descriptions.csv")
    attachments, _ = load_csv_data(settings.data_dir / "csv" / "attachments.csv")
    desc_by_opp = _group_by_key(descriptions, "opportunity_id")
    attach_by_opp = _group_by_key(attachments, "opportunity_id")
    for opp in data:
        opp_id = opp.get("opportunity_id", "")
        opp["_descriptions"] = desc_by_opp.get(opp_id, [])
        opp["_attachments"] = attach_by_opp.get(opp_id, [])
    return DataResponse(data=data, total=len(data), columns=columns)


@router.get("/data/groups", response_model=DataResponse, summary="Get all groups")
async def get_groups(settings: Settings = Depends(get_settings)) -> DataResponse:
    """Get all group data with nested members."""
    data, columns = load_csv_data(settings.data_dir / "csv" / "groups.csv")
    members, _ = load_csv_data(settings.data_dir / "csv" / "group_members.csv")
    members_by_group = _group_by_key(members, "group_id")
    for group in data:
        group["_members"] = members_by_group.get(group.get("group_id", ""), [])
    return DataResponse(data=data, total=len(data), columns=columns)


# Simple data endpoints
router.get("/data/activities", response_model=DataResponse, summary="Get all activities")(_create_simple_data_endpoint("activities.csv"))
router.get("/data/private-texts", response_model=DataResponse, summary="Get all private texts")(_create_simple_data_endpoint("private_texts.jsonl", is_jsonl=True))
router.get("/data/history", response_model=DataResponse, summary="Get all history")(_create_simple_data_endpoint("history.csv"))
router.get("/data/group-members", response_model=DataResponse, summary="Get all group members")(_create_simple_data_endpoint("group_members.csv"))
router.get("/data/attachments", response_model=DataResponse, summary="Get all attachments")(_create_simple_data_endpoint("attachments.csv"))
router.get("/data/opportunity-descriptions", response_model=DataResponse, summary="Get opportunity descriptions")(_create_simple_data_endpoint("opportunity_descriptions.csv"))


class StarterQuestionsResponse(BaseModel):
    questions: list[str]


@router.get("/data/starter-questions", response_model=StarterQuestionsResponse, summary="Get starter questions")
async def get_starter_questions() -> StarterQuestionsResponse:
    """Return starter questions for the chat interface."""
    from backend.agent.question_tree import get_starters
    return StarterQuestionsResponse(questions=get_starters())
