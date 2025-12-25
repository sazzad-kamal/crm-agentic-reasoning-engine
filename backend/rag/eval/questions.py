"""
Evaluation question generation for Account-aware RAG.

Generates test questions from actual CRM CSV data for evaluation.
"""

import random

import pandas as pd

from backend.rag.pipeline.account import load_companies_df


# =============================================================================
# Configuration
# =============================================================================

NUM_QUESTIONS_PER_COMPANY = 3
NUM_COMPANIES = 4
RANDOM_SEED = 42

# Question templates for different question types
QUESTION_TEMPLATES = [
    {
        "type": "history_summary",
        "template": "Summarize the recent interactions and history with {company_name}. What calls, emails, or meetings have occurred?",
    },
    {
        "type": "opportunity_status",
        "template": "What are the current opportunities for {company_name}? What stages are they in and what are the risks or next steps?",
    },
    {
        "type": "attachments",
        "template": "What documents or attachments are associated with {company_name}'s opportunities? Summarize what they contain.",
    },
]


# =============================================================================
# Question Generation
# =============================================================================

def generate_eval_questions(
    seed: int = RANDOM_SEED,
    num_companies: int = NUM_COMPANIES,
    num_questions_per_company: int = NUM_QUESTIONS_PER_COMPANY,
) -> list[dict]:
    """
    Generate evaluation questions from actual CSV data.
    
    Args:
        seed: Random seed for reproducibility
        num_companies: Number of companies to generate questions for
        num_questions_per_company: Questions per company
    
    Returns:
        List of dicts with:
            - id: question ID
            - company_id: target company
            - company_name: company name
            - question: the question text
            - question_type: category of question
    """
    random.seed(seed)
    
    # Load companies
    df = load_companies_df()
    
    # Filter to active companies with data
    active = df[df["status"].isin(["Active", "Trial"])]
    
    # Select companies (deterministic)
    selected = active.head(num_companies)
    
    questions = []
    q_id = 1
    
    for _, company in selected.iterrows():
        company_id = company["company_id"]
        company_name = company["name"]
        
        for tmpl in QUESTION_TEMPLATES[:num_questions_per_company]:
            question = tmpl["template"].format(
                company_name=company_name,
                company_id=company_id,
            )
            
            questions.append({
                "id": f"acct_q{q_id}",
                "company_id": company_id,
                "company_name": company_name,
                "question": question,
                "question_type": tmpl["type"],
            })
            q_id += 1
    
    return questions
