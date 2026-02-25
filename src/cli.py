"""AI Assistant CLI - Your personal AI helper"""
import click
import typer
import typer.core
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

# Get version from package metadata
try:
    from importlib.metadata import version
    __version__ = version("ai-assistant")
except Exception:
    __version__ = "0.1.0"


class _DefaultAskGroup(typer.core.TyperGroup):
    """Typer group that treats unrecognized commands as 'ask' queries.

    Enables ``ai "set timer for 5 min"`` in addition to ``ai ask "..."``.
    """

    def resolve_command(self, ctx: click.Context, args: list[str]):  # type: ignore[override]
        cmd_name = args[0] if args else None
        if cmd_name and cmd_name not in self.commands:
            # Not a known subcommand → treat ALL positional args as one query
            query = " ".join(args)
            return "ask", self.commands["ask"], [query]  # type: ignore[index]
        return super().resolve_command(ctx, args)


app_cli = typer.Typer(
    name="ai",
    help="🤖 Your personal AI assistant — just type: ai \"your question\"",
    add_completion=False,
    no_args_is_help=True,
    cls=_DefaultAskGroup,
)
console = Console()


def version_callback(value: bool):
    """Show version and exit"""
    if value:
        console.print(f"[bold cyan]🤖 AI Assistant[/] version [green]{__version__}[/]")
        raise typer.Exit()


@app_cli.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-V",
        help="Show version and exit",
        callback=version_callback,
        is_eager=True,
    ),
):
    """AI Assistant — just type: ai \"your question or task\""""
    pass


# ================================================================
# RAG subcommand group  (original advanced-rag, untouched logic)
# ================================================================
rag_cli = typer.Typer(help="Ask questions about your books (RAG)")
app_cli.add_typer(rag_cli, name="rag")
@rag_cli.command("ask")
def rag_ask(
    question: str = typer.Argument(..., help="Your question about books"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream intermediate steps"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
):
    """Ask a question about your books using RAG pipeline"""
    from src.core import setup_environment
    from src.graph import create_graph

    setup_environment()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Loading documents and creating vectorstore...", total=None)
        rag_app = create_graph()

    console.print(Panel(f"[bold cyan] Question:[/] {question}", border_style="cyan"))

    inputs = {"question": question}

    if stream:
        last_state = None
        for step in rag_app.stream(inputs): # type: ignore
            for node_name, state in step.items():
                console.print(f"  [dim]→ {node_name}[/]")
                if verbose:
                    console.print(f"    [dim]keys: {list(state.keys())}[/]")
                last_state = state

        if last_state and "generation" in last_state:
            console.print(Panel(
                Markdown(last_state["generation"]),
                title="[bold green]✅ Answer[/]",
                border_style="green",
                padding=(1, 2),
            ))
            if verbose and "documents" in last_state:
                console.print("\n[bold]Documents used:[/]")
                for i, doc in enumerate(last_state["documents"][:5], 1):
                    source = doc.metadata.get("source", "Unknown")
                    console.print(f"  {i}. {source}")

            # If some (but not the majority) docs were irrelevant, offer a web search follow-up
            irrelevant = last_state.get("irrelevant_count", 0)
            total = last_state.get("total_retrieved", 0)
            web_search_done = last_state.get("web_search", "No") == "Yes"
            if irrelevant > 0 and not web_search_done:
                console.print(
                    f"\n[yellow]⚠️  {irrelevant}/{total} retrieved document(s) were not relevant "
                    f"to your question. The answer above is based on the {total - irrelevant} "
                    f"relevant document(s) found.[/]"
                )
                if typer.confirm("🌐 Would you like to supplement with a web search?", default=False):
                    _run_web_supplemented_rag(question, last_state.get("generation", ""), rag_app)
        else:
            console.print("[red]❌ No answer generated.[/]")
    else:
        state = rag_app.invoke(inputs) # type: ignore
        console.print(Panel(
            Markdown(state.get("generation", "No answer generated")),
            title="[bold green]✅ Answer[/]",
            border_style="green",
            padding=(1, 2),
        ))

        irrelevant = state.get("irrelevant_count", 0)
        total = state.get("total_retrieved", 0)
        web_search_done = state.get("web_search", "No") == "Yes"
        if irrelevant > 0 and not web_search_done:
            console.print(
                f"\n[yellow]⚠️  {irrelevant}/{total} retrieved document(s) were not relevant "
                f"to your question. The answer above is based on the {total - irrelevant} "
                f"relevant document(s) found.[/]"
            )
            if typer.confirm("🌐 Would you like to supplement with a web search?", default=False):
                _run_web_supplemented_rag(question, state.get("generation", ""), rag_app)


def _run_web_supplemented_rag(question: str, local_answer: str, rag_app):
    """Run a web-supplemented RAG pass after the user opts in."""
    from src.core import (
        create_question_rewriter,
        get_web_search_tool,
        create_rag_chain,
    )
    from langchain_core.documents import Document

    console.print("\n[bold cyan]🌐 Searching the web...[/]")

    try:
        rewriter = create_question_rewriter()
        web_tool = get_web_search_tool()
        rag_chain = create_rag_chain()

        better_question = rewriter.invoke({"question": question})
        docs = web_tool.invoke({"query": better_question})
        web_content = "\n".join(d["content"] for d in docs)
        web_doc = Document(page_content=web_content)

        generation = rag_chain.invoke({
            "context": [web_doc],
            "question": question,
        })

        console.print(Panel(
            Markdown(generation),
            title="[bold green]✅ Web-Supplemented Answer[/]",
            border_style="green",
            padding=(1, 2),
        ))
    except Exception as e:
        console.print(f"[red]❌ Web search failed: {e}[/]")


@rag_cli.command("rebuild")
def rebuild_index():
    """🔄 Rebuild the vector index from scratch"""
    from src.core import setup_environment, create_vectorstore
    
    setup_environment()
    
    if typer.confirm("⚠️  This will delete and rebuild the entire index. Continue?"):
        with console.status("[bold yellow]Rebuilding index..."):
            create_vectorstore(force_rebuild=True)
        console.print("[green]✅ Index rebuilt successfully![/]")
    else:
        console.print("[yellow]Cancelled[/]")


@rag_cli.command("status")
def index_status():
    """📊 Show vector index statistics"""
    from pathlib import Path
    from src.core import setup_environment
    
    setup_environment()
    
    project_root = Path(__file__).parent.parent
    persist_dir = project_root / "chroma_db"
    
    if not persist_dir.exists():
        console.print("[yellow]⚠️  No index found. Run 'ai rag ask <question>' to create one.[/]")
        return
    
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings
    
    with console.status("[bold cyan]Loading index..."):
        vectorstore = Chroma(
            collection_name="rag-chroma",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(persist_dir)
        )
        
        count = vectorstore._collection.count()
        size_mb = sum(f.stat().st_size for f in persist_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    
    console.print(Panel(
        f"[bold]Index Location:[/] {persist_dir}\n"
        f"[bold]Total Chunks:[/] {count:,}\n"
        f"[bold]Disk Size:[/] {size_mb:.2f} MB\n"
        f"[bold]Status:[/] [green]Ready ✅[/]",
        title="[bold cyan]📊 Vector Index Status[/]",
        border_style="cyan"
    ))


# ================================================================
# ask  –  THE main command. Coordinator routes to the right agents.
# ================================================================
@app_cli.command("ask")
def ask(
    query: str = typer.Argument(..., help="Ask anything — the AI figures out the rest"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed agent steps"),
):
    """
    🤖 Ask anything. The coordinator automatically picks the right agents.

    Examples:\n
      ai ask "What does my book say about stoicism?"\n
      ai ask "Summarize my calendar for today"\n
      ai ask "Set a reminder in 30 minutes to stretch"\n
      ai "What time is it in Tokyo?"\n
    """
    _run_ask(query, verbose)


def _run_ask(query: str, verbose: bool = False) -> None:
    """Core ask logic — shared by the ``ask`` subcommand and the bare ``ai "query"`` callback."""
    from src.core import setup_environment
    from src.agents.graph import create_multi_agent_graph

    setup_environment()

    console.print(Panel(f"[bold cyan]🤖 Query:[/] {query}", border_style="cyan"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Coordinator is planning...", total=None)
        graph = create_multi_agent_graph()
        progress.remove_task(task)

    # Each run needs a unique thread_id so the MemorySaver checkpointer
    # can store and resume state if a human_check interrupt fires.
    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    # Initial state
    initial_state = {
        "query": query,
        "language": "",
        "translated_query": "",
        "plan": [],
        "plan_reasoning": "",
        "current_step": 0,
        "agent_results": [],
        "response": "",
        "agents_used": [],
        "needs_human_confirm": False,
        "human_confirm_message": "",
        "should_stop": False,
    }

    def _stream_until_interrupt(input_or_command):
        """Stream steps and return (last_state, interrupt_value) — interrupt_value
        is non-None when the graph has paused for human confirmation."""
        from langgraph.types import Command

        accumulated = {}
        for step in graph.stream(input_or_command, config=config):  # type: ignore[arg-type]
            # LangGraph surfaces interrupts as a special __interrupt__ key
            if "__interrupt__" in step:
                interrupt_data = step["__interrupt__"][0]
                return accumulated, interrupt_data.value
            for node_name, state in step.items():
                # Show coordinator plan
                if node_name == "coordinator" and state.get("plan"):
                    plan_str = " → ".join(state["plan"])
                    console.print(f"\n  [bold]🎯 Plan:[/] {plan_str}")
                    if verbose and state.get("plan_reasoning"):
                        console.print(f"  [dim]   Reasoning: {state['plan_reasoning']}[/]")
                    console.print()

                # Show which agent is working
                elif node_name == "dispatcher":
                    used = state.get("agents_used", [])
                    if used:
                        latest = used[-1]
                        console.print(f"  {latest} working...")
                        if verbose:
                            results = state.get("agent_results", [])
                            if results:
                                last_result = results[-1]
                                conf = last_result.get("confidence", "")
                                if conf:
                                    console.print(f"    [dim]confidence: {conf}[/]")
                                sources = last_result.get("sources", [])
                                for s in sources[:3]:
                                    console.print(f"    [dim]source: {s}[/]")

                accumulated = {**accumulated, **state}
        return accumulated, None

    # ── Run, handling any human-confirmation interrupts ─────────
    from langgraph.types import Command

    last_state, interrupt_val = _stream_until_interrupt(initial_state)

    while interrupt_val is not None:
        msg = interrupt_val.get("message", "Continue?") if isinstance(interrupt_val, dict) else str(interrupt_val)
        console.print(f"\n[bold yellow]❓ {msg}[/]")
        confirmed = typer.confirm("  Proceed?", default=True)
        answer = "yes" if confirmed else "no"
        last_state, interrupt_val = _stream_until_interrupt(Command(resume=answer))

    # ── Display final answer ────────────────────────────────────
    if last_state and last_state.get("response"):
        console.print(Panel(
            Markdown(last_state["response"]),
            title="[bold green]✅ Answer[/]",
            border_style="green",
            padding=(1, 2),
        ))

        if verbose:
            used = last_state.get("agents_used", [])
            if used:
                console.print(f"\n  [dim]Agents used: {' → '.join(used)}[/]")
    else:
        console.print("[red]❌ No answer generated.[/]")


# ================================================================
# quick  –  Old "ask" — Simple direct LLM question (no agents)
# ================================================================
@app_cli.command("quick")
def quick(
    question: str = typer.Argument(..., help="Ask a quick question"),
):
    """⚡ Quick LLM answer (no agents, no documents, just fast)"""
    from src.core import setup_environment
    from openai import OpenAI

    setup_environment()

    console.print(Panel(f"[bold blue]⚡ Quick:[/] {question}", border_style="blue"))

    with console.status("[bold green]Thinking..."):
        client = OpenAI()
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are a helpful, concise AI assistant. Answer clearly and briefly."},
                {"role": "user", "content": question},
            ],
        )
        answer = resp.choices[0].message.content or ""

    console.print(Panel(
        Markdown(answer),
        title="[bold green]✅ Answer[/]",
        border_style="green",
        padding=(1, 2),
    ))


# ================================================================
# search  –  Direct single-agent: Researcher
# ================================================================
@app_cli.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    num: int = typer.Option(3, "--num", "-n", help="Number of results"),
):
    """🔍 Search the web (Researcher agent — single task)"""
    from src.core import setup_environment
    from src.agents.researcher import search_web

    setup_environment()

    console.print(Panel(f"[bold blue]🔍 Searching:[/] {query}", border_style="blue"))

    with console.status("[bold green]Researcher searching the web..."):
        results = search_web(query, num_results=num)

    if not results:
        console.print("[yellow]No results found.[/]")
        return

    for i, doc in enumerate(results, 1):
        url = doc.get("url", "")
        content = doc.get("content", "No content")
        console.print(Panel(
            f"[link={url}]{url}[/link]\n\n{content}",
            title=f"[cyan]Result {i}[/]",
            border_style="cyan",
        ))


# ================================================================
# joke  –  Random joke (no API key needed)
# ================================================================
@app_cli.command("joke")
def joke():
    """Get a random joke"""
    import httpx

    with console.status("[bold yellow]Finding something funny..."):
        try:
            resp = httpx.get(
                "https://official-joke-api.appspot.com/random_joke",
                timeout=5.0,
            )
            data = resp.json()
            console.print(Panel(
                f"[bold yellow]{data['setup']}[/]\n\n[green]{data['punchline']}[/]",
                title="[bold] Joke[/]",
                border_style="yellow",
            ))
        except Exception:
            console.print("[red]Couldn't fetch a joke right now. Try again![/]")


# ================================================================
# summarize  –  Direct single-agent: Summarizer
# ================================================================
@app_cli.command("summarize")
def summarize(
    text: str = typer.Argument(..., help="Text or URL to summarize"),
):
    """📝 Summarize text or a webpage (Summarizer agent — single task)"""
    from src.core import setup_environment
    from src.agents.summarizer import summarize_text

    setup_environment()

    # If it looks like a URL, fetch the content first
    content = text
    if text.startswith("http://") or text.startswith("https://"):
        import httpx
        console.print(f"[dim]Fetching content from {text}...[/]")
        try:
            resp = httpx.get(text, timeout=10.0, follow_redirects=True)
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = soup.get_text(separator="\n", strip=True)[:8000]
        except Exception as e:
            console.print(f"[red]Failed to fetch URL: {e}[/]")
            raise typer.Exit(1)

    console.print(Panel("[bold blue]📝 Summarizing...[/]", border_style="blue"))

    with console.status("[bold green]Summarizer working..."):
        result = summarize_text(content)

    console.print(Panel(
        Markdown(result),
        title="[bold green]📝 Summary[/]",
        border_style="green",
        padding=(1, 2),
    ))


# ================================================================
# translate  –  Direct single-agent: Translator
# ================================================================
@app_cli.command("translate")
def translate(
    text: str = typer.Argument(..., help="Text to translate"),
    to: str = typer.Option("English", "--to", "-t", help="Target language"),
):
    """🌍 Translate text (Translator agent — single task)"""
    from src.core import setup_environment
    from src.agents.translator import translate_text

    setup_environment()

    console.print(Panel(
        f"[bold blue]🌍 Translating to {to}:[/]\n{text}",
        border_style="blue",
    ))

    with console.status("[bold green]Translator working..."):
        result = translate_text(text, target_language=to)

    console.print(Panel(
        result,
        title=f"[bold green]🌍 {to}[/]",
        border_style="green",
        padding=(1, 2),
    ))


# ================================================================
# info  –  Show available commands
# ================================================================
@app_cli.command("info")
def info():
    """ℹ️  Show system information and available commands"""
    import sys

    console.print(Panel(
        f"[bold cyan]🤖 AI Assistant[/] v{__version__}\n\n"
        f"  Python:   {sys.version.split()[0]}\n"
        f"  Platform: {sys.platform}\n\n"
        f"[bold]🤖 Unified Command (just ask anything):[/]\n\n"
        f"  [cyan]ai[/]  [dim]\"anything\"[/]            The coordinator routes automatically\n"
        f"  [cyan]ai ask[/]  [dim]\"anything\"[/]        Same thing, explicit subcommand\n\n"
        f"[bold]📚 Info Agents:[/]\n\n"
        f"  [dim]\"search for latest AI news\"[/]          🔍 Researcher\n"
        f"  [dim]\"what does my book say about X\"[/]      📚 Librarian (RAG)\n"
        f"  [dim]\"summarize https://example.com\"[/]      📝 Summarizer\n"
        f"  [dim]\"translate 侘寂\"[/]                     🌍 Translator\n\n"
        f"[bold]🖥️  Actions / Tasks:[/]\n\n"
        f"  [dim]\"set reminder in 30 min to stretch\"[/]  ⏰ Reminder\n"
        f"  [dim]\"show my calendar tomorrow\"[/]          📅 Calendar (get)\n"
        f"  [dim]\"create event Meeting at 3 PM\"[/]       📅 Calendar (create)\n"
        f"  [dim]\"write a note: meeting notes...\"[/]     📝 Notes\n"
        f"  [dim]\"set timer for 5 minutes\"[/]            ⏱️  Timer\n"
        f"  [dim]\"start stopwatch\"[/]                    ⏱️  Stopwatch\n"
        f"  [dim]\"what time is it in Tokyo?\"[/]          🌍 World Clock\n\n"
        f"[bold]🔧 Direct Commands:[/]\n\n"
        f"  [cyan]ai search[/]  [dim]\"..\"[/]            🔍 Researcher only\n"
        f"  [cyan]ai summarize[/]  [dim]\"..\"[/]         📝 Summarizer only\n"
        f"  [cyan]ai translate[/]  [dim]\"..\"[/]         🌍 Translator only\n"
        f"  [cyan]ai rag ask[/]  [dim]\"..\"[/]           📚 Librarian (RAG) only\n"
        f"  [cyan]ai quick[/]  [dim]\"..\"[/]             ⚡ Fast LLM (no agents)\n\n"
        f"[bold]⚡ Utilities:[/]\n\n"
        f"  [cyan]ai joke[/]                    😂 Random joke\n"
        f"  [cyan]ai rag status[/]              📊 Index statistics\n"
        f"  [cyan]ai rag rebuild[/]             🔄 Rebuild index\n"
        f"  [cyan]ai info[/]                    ℹ️  This screen\n"
        f"  [cyan]ai --version[/]               📦 Version\n",
        title="[bold blue]ℹ️  System Info[/]",
        border_style="blue",
    ))


if __name__ == "__main__":
    app_cli()

