"""
CLI entrypoint for the Acme CRM RAG system.

Usage:
    # Ask a single question
    python -m backend.rag.cli "What is an Opportunity?"
    
    # Interactive mode
    python -m backend.rag.cli
"""

import sys
import argparse

from backend.rag.retrieval import create_backend, RetrievalBackend
from backend.rag.pipeline import answer_question


def format_answer(result: dict) -> str:
    """Format the RAG result for display."""
    output = []
    
    output.append("\n" + "=" * 60)
    output.append("ANSWER")
    output.append("=" * 60)
    output.append(result["answer"])
    
    output.append("\n" + "-" * 60)
    output.append("Sources:")
    for doc_id in result["doc_ids_used"]:
        output.append(f"  • {doc_id}")
    
    if result.get("cited_docs"):
        output.append(f"\nCited in answer: {', '.join(result['cited_docs'])}")
    
    output.append(f"\nChunks used: {result['num_chunks_used']}")
    output.append(f"Context tokens: ~{result['context_tokens']}")
    
    metrics = result.get("metrics", {})
    if metrics:
        output.append(f"Answer latency: {metrics.get('answer_latency_ms', 0):.0f}ms")
        output.append(f"Total tokens: {metrics.get('total_tokens', 0)}")
    
    return "\n".join(output)


def run_interactive(backend: RetrievalBackend) -> None:
    """Run interactive REPL mode."""
    print("\n" + "=" * 60)
    print("Acme CRM AI Companion - Interactive Mode")
    print("=" * 60)
    print("Ask questions about Acme CRM Suite.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not question:
            continue
        
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        try:
            result = answer_question(question, backend, verbose=False)
            print(format_answer(result))
        except Exception as e:
            print(f"\nError: {e}")


def run_single_question(question: str, backend: RetrievalBackend, verbose: bool = False) -> None:
    """Run a single question and print the result."""
    print(f"\nQuestion: {question}")
    
    try:
        result = answer_question(question, backend, verbose=verbose)
        print(format_answer(result))
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Acme CRM AI Companion - Ask questions about CRM documentation"
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask (if not provided, enters interactive mode)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show verbose output (retrieval details)"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of indexes"
    )
    
    args = parser.parse_args()
    
    # Initialize backend
    print("Initializing RAG backend...")
    try:
        backend = create_backend(rebuild=args.rebuild)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Run 'python -m backend.rag.ingest.docs' first to ingest the documents.")
        sys.exit(1)
    
    print("Backend ready.\n")
    
    # Run appropriate mode
    if args.question:
        run_single_question(args.question, backend, verbose=args.verbose)
    else:
        run_interactive(backend)


if __name__ == "__main__":
    main()
