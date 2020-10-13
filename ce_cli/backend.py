import json
import os

import click
import yaml
from tabulate import tabulate

import ce_api
from ce_api.models import BackendCreate
from ce_cli.cli import cli, pass_info
from ce_cli.utils import api_client, api_call
from ce_cli.utils import check_login_status
from ce_cli.utils import declare
from ce_cli.utils import format_uuid, parse_unknown_options


@cli.group()
@pass_info
def backend(info):
    """Set up your backend configurations with the Core Engine"""
    check_login_status(info)


@backend.command('create', context_settings=dict(ignore_unknown_options=True))
@click.argument('name', type=str)
@click.argument('backend_class', type=str)
@click.argument('backend_type', type=str)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@pass_info
def create_backend(info, name, backend_class, backend_type, args):
    """Create backend for orchestration, processing, training, serving"""
    parsed_args = parse_unknown_options(args)

    for k in parsed_args:
        v = parsed_args[k]
        if v.endswith('.json') and os.path.isfile(v):
            parsed_args[k] = json.load(open(v))
        if v.endswith('.yaml') and os.path.isfile(v):
            parsed_args[k] = yaml.load(open(v))

    click.echo('Registering the backend.')

    api = ce_api.BackendsApi(api_client(info))
    api_call(api.create_backend_api_v1_backends_post,
             BackendCreate(backend_class=backend_class,
                           name=name,
                           type=backend_type,
                           args=parsed_args))


@backend.command('list')
@click.argument('backend_class', required=False, default=None, type=str)
@pass_info
def list_backends(info, backend_class):
    """Lists all created backends"""
    b_api = ce_api.BackendsApi(api_client(info))
    b_list = api_call(b_api.get_loggedin_backend_api_v1_backends_get)

    if backend_class:
        b_list = [b for b in b_list if b.backend_class == backend_class]

    declare('You have {count} different {class_}backend(s) so '
            'far. \n'.format(count=len(b_list),
                             class_=backend_class + ' '
                             if backend_class else ''))

    if b_list:
        b_list = sorted(b_list, key=lambda b: b.backend_class)
        table = []
        for b in b_list:
            table.append({'ID': format_uuid(b.id),
                          'Name': b.name,
                          'Backend Class': b.backend_class,
                          'Backend Type': b.type,
                          'Created At': b.created_at})
        click.echo(tabulate(table, headers='keys', tablefmt='presto'))
        click.echo()
