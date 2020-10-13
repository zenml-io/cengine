#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
.. currentmodule:: ce_cli.cli
.. moduleauthor:: maiot GmbH <support@maiot.io>
"""
import logging
import os
import sys

import click

from ce_cli.utils import pass_info

# set tensorflow logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

LOGGING_LEVELS = {
    0: logging.NOTSET,
    1: logging.ERROR,
    2: logging.WARN,
    3: logging.INFO,
    4: logging.DEBUG,
}


@click.group()
@click.option("--verbose", "-v", default=0, count=True,
              help="Enable verbose output.")
@pass_info
def cli(info, verbose: int):
    """maiot Core Engine"""
    info.load()
    if verbose > 0:
        logging.basicConfig(
            level=LOGGING_LEVELS[verbose]
            if verbose in LOGGING_LEVELS
            else logging.DEBUG
        )
        click.echo(
            click.style(
                f"Verbose logging is enabled. "
                f"(LEVEL={logging.getLogger().getEffectiveLevel()})",
                fg="yellow",
            )
        )
    else:
        logging.disable(sys.maxsize)
        logging.getLogger().disabled = True


if __name__ == '__main__':
    cli()
