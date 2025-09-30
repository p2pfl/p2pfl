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
import sys
from glob import glob
from typing import Annotated, TypedDict

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

if len(sys.argv) > 1 and sys.argv[1] == "help":
    sys.argv[1] = "--help"

console = Console()
app = typer.Typer(
    rich_markup_mode="rich",
    help="[bold blue]:earth_americas: P2PFL-CLI[/bold blue] | Peer-to-Peer Federated Learning Command Line Tool",
)


@app.command()
def login(
    url: Annotated[str | None, typer.Option(help="The P2PFL Web Services URL")] = None,
    token: Annotated[str | None, typer.Option(help="Your API token")] = None,
) -> None:
    """
    Authenticate with the P2PFL Web Services platform.

    This command will set the necessary environment variables for web logging.

    Args:
        url: The P2PFL Web Services URL.
        token: Your API token.

    """
    # Show the logo first
    console.print(logo)

    # Interactive prompts if not provided via command line
    if url is None:
        url = typer.prompt("🌐 Enter P2PFL Web Services URL")

    if token is None:
        token = typer.prompt("🔑 Enter your API token", hide_input=True)

    # Validate inputs
    if not url:
        console.print(
            Panel(
                ":x: [bold red]Error:[/bold red] URL cannot be empty.",
                title="[bold red]Invalid Input[/bold red]",
            )
        )
        raise typer.Exit(code=1)

    if not token:
        console.print(
            Panel(
                ":x: [bold red]Error:[/bold red] Token cannot be empty.",
                title="[bold red]Invalid Input[/bold red]",
            )
        )
        raise typer.Exit(code=1)

    # Set environment variables
    os.environ["P2PFL_WEB_LOGGER_URL"] = url
    os.environ["P2PFL_WEB_LOGGER_KEY"] = token

    # Save to a .env file for persistence (optional)
    env_file = os.path.join(os.path.expanduser("~"), ".p2pfl_env")
    try:
        with open(env_file, "w") as f:
            f.write(f"P2PFL_WEB_LOGGER_URL={url}\n")
            f.write(f"P2PFL_WEB_LOGGER_KEY={token}\n")

        console.print(
            Panel(
                f":white_check_mark: [bold green]Successfully authenticated![/bold green]\n\n"
                f"[dim]Environment variables set:[/dim]\n"
                f"  • P2PFL_WEB_LOGGER_URL={url}\n"
                f"  • P2PFL_WEB_LOGGER_KEY=[hidden]\n\n"
                f"[dim]Configuration saved to: {env_file}[/dim]",
                title="[bold green]Authentication Successful[/bold green]",
            )
        )

        console.print("\n:information_source: [dim]To use these credentials in future sessions, run:[/dim]")
        console.print(f"[bold cyan]source {env_file}[/bold cyan]")

    except Exception as e:
        console.print(
            Panel(
                f":warning: [bold yellow]Warning:[/bold yellow] Could not save configuration file: {e}\n"
                f"Environment variables have been set for this session only.",
                title="[bold yellow]Partial Success[/bold yellow]",
            )
        )


@app.command()
def remote() -> None:
    """Interact with a remote node in the p2pfl network."""
    console.print(":sweat_smile: [bold yellow]Not implemented yet![/bold yellow] \n:rocket: Comming soon!")


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


class ExampleInfo(TypedDict):
    """Information about an example."""

    description: str
    path: str


def __get_available_examples() -> dict[str, ExampleInfo]:
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


@app.command(name="run-variations")
def run_variations(
    yaml_path: str,
    aggregators: Annotated[list[str] | None, typer.Option(help="List of aggregator classes")] = None,
    seeds: Annotated[list[int] | None, typer.Option(help="List of random seeds")] = None,
    nodes: Annotated[list[int] | None, typer.Option(help="Number of nodes")] = None,
    rounds: Annotated[list[int] | None, typer.Option(help="Number of rounds")] = None,
    epochs: Annotated[list[int] | None, typer.Option(help="Number of epochs per round")] = None,
    topologies: Annotated[list[str] | None, typer.Option(help="Network topologies")] = None,
    partitioning: Annotated[list[str] | None, typer.Option(help="Dataset partitioning strategies")] = None,
    models: Annotated[list[str] | None, typer.Option(help="Model packages/architectures")] = None,
    batch_sizes: Annotated[list[int] | None, typer.Option(help="Batch sizes")] = None,
    param: Annotated[list[str] | None, typer.Option(help="Custom parameter in format 'path.to.param=value1,value2'")] = None,
    output_dir: Annotated[str | None, typer.Option(help="Base directory for results (default: results/variations)")] = None,
    skip_existing: Annotated[bool, typer.Option(help="Skip experiments with existing results")] = True,
    force: Annotated[bool, typer.Option(help="Force re-run all experiments, ignoring existing results")] = False,
    full_param_names: Annotated[bool, typer.Option(help="Use full parameter paths in folder names")] = False,
) -> None:
    """
    Run experiments with parameter variations for grid search and hyperparameter optimization.

    Examples:
        # Run with different aggregators and seeds
        p2pfl run-variations config.yaml --aggregators FedAvg FedMedian --seeds 42 123

        # Run with different network configurations
        p2pfl run-variations config.yaml --nodes 5 10 20 --topologies star ring full

        # Run with custom parameters using dot notation
        p2pfl run-variations config.yaml --param experiment.dataset.batch_size=32,64,128

    """
    from p2pfl.utils.run_variations import run_variations_experiment

    # Prepare arguments for the run_variations function
    result = run_variations_experiment(
        yaml_path=yaml_path,
        aggregators=aggregators,
        seeds=seeds,
        nodes=nodes,
        rounds=rounds,
        epochs=epochs,
        topologies=topologies,
        partitioning=partitioning,
        models=models,
        batch_sizes=batch_sizes,
        custom_params=param,
        output_dir=output_dir,
        skip_existing=skip_existing,
        force=force,
        full_param_names=full_param_names,
        console=console,
    )

    if result != 0:
        raise typer.Exit(code=result)


if __name__ == "__main__":
    app()
