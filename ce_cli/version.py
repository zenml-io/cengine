import click

from ce_cli.cli import cli
from ce_standards.version import __release__


@cli.command()
def version():
    """Version of the Core Engine"""
    click.echo(click.style(r"""      
                     _       _     _____               ______             _
                    (_)     | |   / ____|             |  ____|           (_)
     _ __ ___   __ _ _  ___ | |_ | |     ___  _ __ ___| |__   _ __   __ _ _ _ __   ___
    | '_ ` _ \ / _` | |/ _ \| __|| |    / _ \| '__/ _ \  __| | '_ \ / _` | | '_ \ / _ \
    | | | | | | (_| | | (_) | |_ | |___| (_) | | |  __/ |____| | | | (_| | | | | |  __/
    |_| |_| |_|\__,_|_|\___/ \__| \_____\___/|_|  \___|______|_| |_|\__, |_|_| |_|\___|
                                                                    __/ |
                                                                   |___/
         """, fg='green'))
    click.echo(click.style(f"version: {__release__}", bold=True))
