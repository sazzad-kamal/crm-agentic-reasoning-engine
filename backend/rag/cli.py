"""
CLI entrypoint for the Acme CRM RAG system.

Usage:
    # Ask a single question
    python -m backend.rag.cli ask "What is an Opportunity?"
    
    # Interactive mode
    python -m backend.rag.cli chat
"""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt

from backend.rag.retrieval import create_backend, RetrievalBackend
from backend.rag.pipeline import answer_question


# Initialize Typer app and Rich console
app = typer.Typer(
    name="rag",
    help="Acme CRM AI Companion - Ask questions about CRM documentation",
    add_completion=False,
)
console = Console()


def format_answer(result: dict) -> None:
    """Format and display the RAG result using Rich."""
    # Answer panel
    console.print(Panel(result["answer"], title="Answer", border_style="green"))
    
    # Sources table
    table = Table(title="Sources", show_header=True, header_style="bold cyan")
    table.add_column("Document", style="dim")
    table.add_column("Cited", justify="center")
    
    cited_docs = set(result.get("cited_docs", []))
    for doc_id in result["doc_ids_used"]:
        cited = "✓" if doc_id in cited_docs else ""
        table.add_row(doc_id, cited)
    
    console.print(table)
    
    # Metrics
    metrics = result.get("metrics", {})
    console.print(
        f"\n[dim]Chunks: {result['num_chunks_used']} | "
        f"Tokens: ~{result['context_tokens']} | "
        f"Latency: {metrics.get('answer_latency_ms', 0):.0f}ms[/dim]"
    )


def get_backend() -> RetrievalBackend:
    """Initialize and return the RAG backend."""
    with console.status("[bold green]Initializing RAG backend..."):
        try:
            backend = create_backend()
        except (FileNotFoundError, ValueError) as e:
            console.print(f"[red]Error:[/red] {e}")
            console.print("[yellow]Run 'python -m backend.rag.ingest.docs' first.[/yellow]")
            raise typer.Exit(1)
    
    console.print("[green]✓[/green] Backend ready\n")
    return backend


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask about CRM"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show retrieval details"),
):
    """Ask a single question and get an answer."""
    backend = get_backend()
    
    console.print(f"[bold]Question:[/bold] {question}\n")
    
    with console.status("[bold green]Searching..."):
        result = answer_question(question, backend, verbose=verbose)
    
    format_answer(result)


@app.command()
def chat():
    """Start interactive chat mode."""
    backend = get_backend()
    
    console.print(Panel(
        "Ask questions about Acme CRM Suite.\n"
        "Type [bold]quit[/bold] or [bold]exit[/bold] to stop.",
        title="Acme CRM AI Companion",
        border_style="blue",
    ))
    
    while True:
        try:
            question = Prompt.ask("\n[bold cyan]You[/bold cyan]")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        if not question.strip():
            continue
        
        if question.strip().lower() in ("quit", "exit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        
        try:
            with console.status("[bold green]Thinking..."):
                result = answer_question(question, backend, verbose=False)
            format_answer(result)
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


# Keep backward compatibility with old CLI interface
def main():
    """Main CLI entrypoint (backward compatible)."""
    app()


if __name__ == "__main__":
    main()
