"""Quick test for SQL Sorcerer approach."""

from dotenv import load_dotenv

load_dotenv()

from backend.agent.route.sql_planner import get_sql_plan  # noqa: E402

TEST_QUESTIONS = [
    "What deals are in the pipeline?",
    "How are deals distributed by stage?",
    "What are the highest value deals?",
    "Which deals involve Acme Corp?",
    "What deals are in Proposal stage?",
    "Which accounts are at risk?",
    "Who are the decision makers?",
    "What's the history with Acme?",
]


def main():
    print("Testing SQL Sorcerer approach\n")
    print("=" * 60)

    for q in TEST_QUESTIONS:
        print(f"\nQ: {q}")
        try:
            result = get_sql_plan(q)
            print(f"SQL: {result.sql}")
            print(f"RAG: {result.needs_rag}")
        except Exception as e:
            print(f"ERROR: {e}")
        print("-" * 60)


if __name__ == "__main__":
    main()
