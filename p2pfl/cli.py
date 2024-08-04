#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/federated_learning_p2p).
# Copyright (c) 2024 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""CLI for the p2pfl platform."""

import os
import subprocess
import sys
from typing import Annotated, Dict

import pkg_resources
import typer
from rich.box import HEAVY_HEAD
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

####
# ASCII Art
####

logo = r"""[italic]

[blue]                66666
[blue]          6666666666666666
[blue]       6666666666666666666666
[blue]      6666666666666666666666666
[blue]    6666  6666666666666666666666       [white]                                       666666
[blue]   6666       6666  6666666666666      [white]            666666666                666666666
[blue]   666            666666666666666      [white]666666666   6666666666  666666666   66666666666
[blue]   66           666666666666666666     [white]6666666666      66666   6666666666  66666666666
[blue]   66           666666666666666666     [white]6666   6666    66666    6666   6666  6666  6666
[blue]   66  66666    666666666666666666     [white]66666666666  666666666  6666666666   6666  6666
[blue]   6666666666    6666666666666666      [white]6666666666  6666666666  666666666    6666  66666
[blue]   6666666666      66666666666666      [white]66666                   6666                66666
[blue]    6666666666      666666666666       [white]6666                    6666
[blue]      66666666      66666  6666
[blue]       6666666          66666
[blue]          66666666666666666
[blue]              66666666

"""


####
# CLI Commands
####

console = Console()
app = typer.Typer(
    rich_markup_mode="rich",
    help="[bold blue]:earth_americas: P2PFL-CLI[/bold blue] | Peer-to-Peer Federated Learning Command Line Tool",
)


@app.command()
def login(
    token: Annotated[str, typer.Option(help="🔑 Your API token")] = "",
) -> None:  # prompt="🔑 Enter your API token"
    """
    Authenticate with the p2pfl platform using your API token.

    Args:
        token: Your API token.

    """
    console.print(":sweat_smile: [bold yellow]Not implemented yet![/bold yellow] \n:rocket: Comming soon!")
    # console.print(f"Authenticating with token: {token}...")


@app.command()
def remote() -> None:
    """Interact with a remote node in the p2pfl network."""
    console.print(":sweat_smile: [bold yellow]Not implemented yet![/bold yellow] \n:rocket: Comming soon!")


@app.command()
def launch() -> None:
    """Launch a new node in the p2pfl network."""
    console.print(":sweat_smile: [bold yellow]Not implemented yet![/bold yellow] \n:rocket: Comming soon!")


####
# Experiment Commands
####

exp_app = typer.Typer(help="Run experiments on the p2pfl platform.")
app.add_typer(exp_app, name="experiment")

EXAMPLES_DIR = pkg_resources.resource_filename('p2pfl', 'examples')

def __get_available_examples() -> Dict[str, str]:
    # Load the available examples
    files = [filename[:-3] for filename in os.listdir(EXAMPLES_DIR) if filename.endswith(".py")]

    # Read the docstrings of the examples
    return {file: __read_docstring(os.path.join(EXAMPLES_DIR, file + ".py")) for file in files}


def __read_docstring(file) -> str:
    with open(file) as f:
        content = f.read().split('"""')
        docstring = content[1] if len(content) > 1 else ""
        return docstring.strip()


@exp_app.command()
def list() -> None:
    """List available examples."""
    # Get the available examples
    examples = __get_available_examples()

    # Table enhancements
    table = Table(
        title=":page_with_curl: Available Examples",
        show_header=True,
        box=HEAVY_HEAD,
        show_lines=True,
        expand=True,  # Allow table to expand to full width
        header_style="bold magenta",
    )
    table.add_column("Name", style="green", width=12)
    table.add_column("Description")

    # Generate the table
    for name, description in examples.items():
        table.add_row(name, description)

    console.print(table)


@exp_app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(
    ctx: typer.Context,
    example: str,
) -> None:
    """
    Run an example.

    Args:
        example: The name of the example
        ctx: The Typer context

    """
    # Check if the example exists
    examples = __get_available_examples()
    if example not in examples:
        console.print(
            Panel(
                f":x: [bold red]Error:[/bold red] Example [bold yellow]{example}[/bold yellow] not found. \n\n"
                f"Use [bold green]experiment list[/bold green] to see available examples.",
                title="[bold red]Example Not Found[/bold red]",
            )
        )
        raise typer.Exit(code=1)

    # Run the example
    console.print(logo)
    console.print(f":tada: [bold yellow]Running example {example}...[/bold yellow]\n\n")

    try:
        process = subprocess.Popen(
            [sys.executable, f"{EXAMPLES_DIR}/{example}.py"] + ctx.args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr for easier streaming
            bufsize=1,  # Line buffering for real-time output
            text=True,  # Decode output as text
        )

        if process.stdout:
            # Stream output line by line
            for line in process.stdout:
                print(line, end="")  # Print the line without a newline

            # Close the stdout pipe
            process.stdout.close()

        # Wait for the process to finish
        returncode = process.wait()
        if returncode != 0:
            raise

        # Print completion message
        console.print(f"\n\n:sparkles: [bold green]Example {example} completed![/bold green]")

    except Exception:
        console.print(f"\n\n:x: [bold red]Error running {example}[/bold red]")


if __name__ == "__main__":
    app()
