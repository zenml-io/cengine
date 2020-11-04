import json
import os

import click
import yaml
from tabulate import tabulate

import ce_api
from ce_api.models import ProviderCreate
from ce_cli.cli import cli, pass_info
from ce_cli.utils import api_client, api_call
from ce_cli.utils import check_login_status
from ce_cli.utils import declare
from ce_cli.utils import format_uuid, find_closest_uuid, parse_unknown_options
from ce_standards import constants


@cli.group()
@pass_info
def provider(info):
    """Set up your cloud provider with the Core Engine"""
    check_login_status(info)


@provider.command('create', context_settings=dict(ignore_unknown_options=True))
@click.argument('name', type=str)
@click.argument('type', type=str)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@pass_info
def create_provider(info, name, type, args):
    """Create a provider with a unique name"""
    parsed_args = parse_unknown_options(args)

    for k in parsed_args:
        v = parsed_args[k]
        if v.endswith('.json') and os.path.isfile(v):
            parsed_args[k] = json.load(open(v))
        if v.endswith('.yaml') and os.path.isfile(v):
            parsed_args[k] = yaml.load(open(v))

    click.echo('Registering the provider.')

    api = ce_api.ProvidersApi(api_client(info))
    api_call(api.create_provider_api_v1_providers_post,
             ProviderCreate(name=name,
                            type=type,
                            args=parsed_args))


@provider.command('list')
@pass_info
def list_providers(info):
    p_api = ce_api.ProvidersApi(api_client(info))
    p_list = api_call(p_api.get_loggedin_provider_api_v1_providers_get)

    declare('You have {count} different providers(s) so '
            'far. \n'.format(count=len(p_list)))

    user = info[constants.ACTIVE_USER]
    if constants.ACTIVE_PROVIDER in info[user]:
        active_p = info[user][constants.ACTIVE_PROVIDER]
    else:
        active_p = None

    if p_list:
        table = []
        for p in p_list:
            table.append({'Selection': '*' if p.id == active_p else '',
                          'ID': format_uuid(p.id),
                          'Name': p.name,
                          'Type': p.type,
                          'Created At': p.created_at})
        click.echo(tabulate(table, headers='keys', tablefmt='presto'))
        click.echo()


@provider.command('set')
@click.argument('provider_id', type=str)
@pass_info
def set_provider(info, provider_id):
    user = info[constants.ACTIVE_USER]

    api = ce_api.ProvidersApi(api_client(info))
    p_list = api_call(api.get_loggedin_provider_api_v1_providers_get)
    p_id = find_closest_uuid(provider_id, p_list)

    info[user][constants.ACTIVE_PROVIDER] = p_id
    info.save()
    declare('Active provider set to id: {id}'.format(id=format_uuid(p_id)))
