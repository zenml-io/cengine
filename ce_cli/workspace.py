import click
from tabulate import tabulate

import ce_api
from ce_standards import constants
from ce_api.models import WorkspaceCreate
from ce_cli.cli import cli, pass_info
from ce_cli.utils import check_login_status, api_client, api_call, declare, \
    format_uuid, find_closest_uuid


@cli.group()
@pass_info
def workspace(info):
    """Interaction with workspaces"""
    check_login_status(info)


@workspace.command('list')
@pass_info
def list_workspaces(info):
    """List of all workspaces available to the user"""
    user = info[constants.ACTIVE_USER]

    api = ce_api.WorkspacesApi(api_client(info))
    ws_list = api_call(api.get_loggedin_workspaces_api_v1_workspaces_get)

    if constants.ACTIVE_WORKSPACE in info[user]:
        active_w = info[user][constants.ACTIVE_WORKSPACE]
    else:
        active_w = None

    declare('You have created {count} different '
            'workspace(s). \n'.format(count=len(ws_list)))
    if ws_list:
        table = []
        for w in ws_list:
            table.append({'Selection': '*' if w.id == active_w else '',
                          'ID': format_uuid(w.id),
                          'Name': w.name,
                          'Provider': format_uuid(w.provider_id)})
        click.echo(tabulate(table, headers='keys', tablefmt='presto'))
        click.echo()


@workspace.command('set')
@click.argument("workspace_id", default=None, type=str)
@pass_info
def set_workspace(info, workspace_id):
    """Set workspace to be active"""
    user = info[constants.ACTIVE_USER]

    api = ce_api.WorkspacesApi(api_client(info))
    all_ws = api_call(api.get_loggedin_workspaces_api_v1_workspaces_get)
    ws_uuid = find_closest_uuid(workspace_id, all_ws)

    api_call(api.get_workspace_api_v1_workspaces_workspace_id_get,
             ws_uuid)

    info[user][constants.ACTIVE_WORKSPACE] = ws_uuid
    info.save()
    declare('Active workspace set to id: {id}'.format(id=format_uuid(
        ws_uuid)))


@workspace.command('create')
@click.argument("name", type=str)
@click.argument("provider_id", type=str)
@pass_info
@click.pass_context
def create_workspace(ctx, info, provider_id, name):
    """Create a workspace and set it to be active."""
    click.echo('Registering the workspace "{}"...'.format(name))

    w_api = ce_api.WorkspacesApi(api_client(info))
    p_api = ce_api.ProvidersApi(api_client(info))

    p_list = api_call(p_api.get_loggedin_provider_api_v1_providers_get)
    p_uuid = find_closest_uuid(provider_id, p_list)

    ws = api_call(w_api.create_workspace_api_v1_workspaces_post,
                  WorkspaceCreate(name=name,
                                  provider_id=p_uuid))

    declare('Workspace registered.')
    ctx.invoke(set_workspace, workspace_id=ws.id)
