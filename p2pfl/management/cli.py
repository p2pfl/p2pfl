#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
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
from glob import glob
from typing import Annotated, Dict, TypedDict

import typer
import yaml
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


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


class ExampleInfo(TypedDict):
    """Information about an example."""

    description: str
    path: str


def __get_available_examples() -> Dict[str, ExampleInfo]:
    """Get all the available yaml examples and their descriptions."""
    examples = {}
    # Find all yaml files in examples subdirectories
    for yaml_path in glob(os.path.join(EXAMPLES_DIR, "*/*.yaml")):
        dirname = os.path.basename(os.path.dirname(yaml_path))
        with open(yaml_path) as f:
            content = yaml.safe_load(f)
            if isinstance(content, dict) and "description" in content:
                examples[dirname] = ExampleInfo(description=content["description"], path=yaml_path)
    return examples


@app.command()
def run(file_or_example: str) -> None:
    """Run an experiment from a YAML configuration file or a predefined example name."""
    yaml_file = file_or_example

    # If the file doesn't end with .yaml/.yml, try to find it as an example
    if not file_or_example.lower().endswith((".yaml", ".yml")):
        examples = __get_available_examples()
        if file_or_example in examples:
            yaml_file = examples[file_or_example]["path"]
        else:
            available = ", ".join(examples.keys())
            console.print(
                Panel(
                    f":x: [bold red]Error:[/bold red] Example [bold yellow]{file_or_example}[/bold yellow] not found.\n"
                    f"Available examples: {available}",
                    title="[bold red]Example Not Found[/bold red]",
                )
            )
            raise typer.Exit(code=1)
    elif not os.path.exists(yaml_file) or os.path.isdir(yaml_file):
        console.print(
            Panel(
                f":x: [bold red]Error:[/bold red] Experiment [bold yellow]{yaml_file}[/bold yellow] not found.",
                title="[bold red]Experiment Not Found[/bold red]",
            )
        )
        raise typer.Exit(code=1)

    console.print(logo)
    console.print(f":tada: [bold yellow]Running experiment from {yaml_file}...[/bold yellow]\n\n")
    from p2pfl.management.launch_from_yaml import run_from_yaml

    run_from_yaml(yaml_file)


@app.command()
def list_examples() -> None:
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
    for name, info in examples.items():
        table.add_row(name, info["description"])

    console.print(table)


if __name__ == "__main__":
    app()
