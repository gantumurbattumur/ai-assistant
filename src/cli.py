"""AI Assistant CLI - Your personal AI helper"""
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

app_cli = typer.Typer(
    name="ai",
    help="Your personal AI assistant",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


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


# ================================================================
# ask  –  Quick LLM question (no RAG, no documents)
# ================================================================
@app_cli.command("ask")
def ask(
    question: str = typer.Argument(..., help="Ask anything"),
):
    """Quick ask using LLM (no documents needed)"""
    from src.core import setup_environment
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    setup_environment()

    console.print(Panel(f"[bold blue] Question:[/] {question}", border_style="blue"))

    with console.status("[bold green]Thinking..."):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful, concise AI assistant. Answer clearly and briefly."),
            ("human", "{question}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"question": question})

    console.print(Panel(
        Markdown(response.content),
        title="[bold green]✅ Answer[/]",
        border_style="green",
        padding=(1, 2),
    ))


# ================================================================
# search  –  Web search via Tavily
# ================================================================
@app_cli.command("search")
def search(
    query: str = typer.Argument(..., help="Search query"),
    num: int = typer.Option(3, "--num", "-n", help="Number of results"),
):
    """🔍 Search the web"""
    from src.core import setup_environment, get_web_search_tool

    setup_environment()

    console.print(Panel(f"[bold blue] Searching:[/] {query}", border_style="blue"))

    with console.status("[bold green]Searching the web..."):
        tool = get_web_search_tool()
        results = tool.invoke({"query": query})

    for i, doc in enumerate(results[:num], 1):
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
                title="[bold]😂 Joke[/]",
                border_style="yellow",
            ))
        except Exception:
            console.print("[red]Couldn't fetch a joke right now. Try again![/]")


# ================================================================
# summarize  –  Summarize text or a URL
# ================================================================
@app_cli.command("summarize")
def summarize(
    text: str = typer.Argument(..., help="Text or URL to summarize"),
):
    """Summarize text or a webpage"""
    from src.core import setup_environment
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

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
            # Remove script and style elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            content = soup.get_text(separator="\n", strip=True)[:8000]
        except Exception as e:
            console.print(f"[red]Failed to fetch URL: {e}[/]")
            raise typer.Exit(1)

    console.print(Panel("[bold blue]📝 Summarizing...[/]", border_style="blue"))

    with console.status("[bold green]Generating summary..."):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a summarization expert. Provide a clear, concise summary "
             "with key points as bullet points."),
            ("human", "Summarize the following:\n\n{content}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"content": content})

    console.print(Panel(
        Markdown(response.content),
        title="[bold green]📝 Summary[/]",
        border_style="green",
        padding=(1, 2),
    ))


# ================================================================
# translate  –  Translate text
# ================================================================
@app_cli.command("translate")
def translate(
    text: str = typer.Argument(..., help="Text to translate"),
    to: str = typer.Option("English", "--to", "-t", help="Target language"),
):
    """🌍 Translate text to another language"""
    from src.core import setup_environment
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate

    setup_environment()

    console.print(Panel(
        f"[bold blue]🌍 Translating to {to}:[/]\n{text}",
        border_style="blue",
    ))

    with console.status("[bold green]Translating..."):
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a professional translator. Translate the given text to {language}. "
             "Only output the translation, nothing else."),
            ("human", "{text}"),
        ])
        chain = prompt | llm
        response = chain.invoke({"text": text, "language": to})

    console.print(Panel(
        response.content,
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
        f"[bold cyan]🤖 AI Assistant[/] v0.1.0\n\n"
        f"  Python:   {sys.version.split()[0]}\n"
        f"  Platform: {sys.platform}\n\n"
        f"[bold]Available Commands:[/]\n\n"
        f"  [cyan]ai ask[/]  [dim]\"..\"[/]           💬 Quick LLM question\n"
        f"  [cyan]ai rag ask[/]  [dim]\"..\"[/]       📚 Ask about your books (RAG)\n"
        f"  [cyan]ai search[/]  [dim]\"..\"[/]        🔍 Search the web\n"
        f"  [cyan]ai joke[/]                😂 Get a random joke\n"
        f"  [cyan]ai summarize[/]  [dim]\"..\"[/]     📝 Summarize text or URL\n"
        f"  [cyan]ai translate[/]  [dim]\"..\"[/]     🌍 Translate text\n"
        f"  [cyan]ai info[/]                ℹ️  This screen\n",
        title="[bold blue]ℹ️  System Info[/]",
        border_style="blue",
    ))


if __name__ == "__main__":
    app_cli()

