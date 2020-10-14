from operator import attrgetter

import click
import yaml
from tabulate import tabulate

import ce_api
from ce_api.models import DatasourceCreate, DatasourceCommitCreate
from ce_cli.cli import cli, pass_info
from ce_cli.utils import check_login_status, api_client, api_call, declare, \
    format_uuid, find_closest_uuid, format_date, \
    parse_unknown_options, resolve_datasource_commits, confirmation, error
from ce_standards import constants


@cli.group()
@pass_info
def datasource(info):
    """Interaction with datasources"""
    check_login_status(info)


@datasource.command('create',
                    context_settings=dict(ignore_unknown_options=True))
@click.argument("name", type=str)
@click.argument("ds_type", type=str)
@click.argument("source", type=str)
@click.argument("provider_id", type=str)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@pass_info
def create_datasource(info,
                      name,
                      ds_type,
                      source,
                      provider_id,
                      args):
    """Create a datasource"""
    click.echo('Registering datasource {}...'.format(name))

    parsed_args = parse_unknown_options(args)

    api = ce_api.DatasourcesApi(api_client(info))
    p_api = ce_api.ProvidersApi(api_client(info))

    p_list = api_call(p_api.get_loggedin_provider_api_v1_providers_get)
    p_uuid = find_closest_uuid(provider_id, p_list)

    ds = api_call(
        api.create_datasource_api_v1_datasources_post,
        DatasourceCreate(
            name=name,
            type=ds_type,
            source=source,
            provider_id=p_uuid,
            args=parsed_args,
        ))

    declare('Datasource registered with ID: {}'.format(
        format_uuid(ds.id)))


@datasource.command('commit')
@click.argument("datasource_id", default=None, type=str)
@click.option("-m", "--message", help='Message of the commit')
@click.option("-s", "--schema", type=click.Path(exists=True),
              help='Schema of the datasource')
@click.option('--orchestration_backend', required=False, default=None, type=str)
@click.option('--orchestration_args', required=False, default={}, type=dict)
@click.option('--processing_backend', required=False, default=None, type=str)
@click.option('--processing_args', required=False, default={}, type=dict)
@click.option('-f', '--force', is_flag=True, default=False,
              help='Force commit with no prompts')
@pass_info
@click.pass_context
def commit_datasource(ctx,
                      info,
                      datasource_id,
                      message,
                      schema,
                      orchestration_backend,
                      orchestration_args,
                      processing_backend,
                      processing_args,
                      force):
    """Creates a commit for a datasource"""
    api = ce_api.DatasourcesApi(api_client(info))

    if not force:
        confirmation('Committing will trigger a pipeline that will create a '
                     'snapshot of your datasources current state. '
                     'This might take a while. '
                     'Are you sure you wish to continue?', abort=True)

    # find closest, this a heavy call for now
    all_ds = api_call(api.get_datasources_api_v1_datasources_get)
    ds_uuid = find_closest_uuid(datasource_id, all_ds)

    if schema:
        try:
            with open(schema, 'rt', encoding='utf8') as f:
                schema_dict = yaml.load(f)
        except:
            error('Badly formatted YAML!')
            schema_dict = dict()
    else:
        schema_dict = dict()

    commit = api_call(
        api.create_datasource_commit_api_v1_datasources_ds_id_commits_post,
        DatasourceCommitCreate(
            message=message,
            used_schema=schema_dict,
            orchestration_backend=orchestration_backend,
            orchestration_args=orchestration_args,
            processing_backend=processing_backend,
            processing_args=processing_args,
        ),
        ds_id=ds_uuid,
    )
    declare('Commit successful: {}'.format(format_uuid(commit.id)))

    active_commit = '{datasource_id}:{commit_id}'.format(datasource_id=ds_uuid,
                                                         commit_id=commit.id)

    user = info[constants.ACTIVE_USER]
    info[user][constants.ACTIVE_DATASOURCE_COMMIT] = active_commit
    info.save()
    declare('Active datasource commit set to: {}'.format(
        format_uuid(active_commit)))


@datasource.command("list")
@pass_info
def list_datasources(info):
    """List of all the available datasources"""
    user = info[constants.ACTIVE_USER]
    if constants.ACTIVE_DATASOURCE_COMMIT in info[user]:
        active_dc = info[user][constants.ACTIVE_DATASOURCE_COMMIT]
        active_dc = active_dc.split(':')[1]
    else:
        active_dc = None
    api = ce_api.DatasourcesApi(api_client(info))
    ds_list = api_call(api.get_datasources_api_v1_datasources_get)

    declare('You have created {count} different '
            'datasource(s).\n'.format(count=len(ds_list)))
    declare("Use 'cengine datasource commits DATASOURCE_ID' see commits of  "
            "any datasource.\n")

    if ds_list:
        table = []
        for ds in ds_list:
            dcs = [x.id for x in ds.datasource_commits]
            status = 'No Commit'
            latest_created_at = 'No Commit'
            if len(dcs) != 0:
                latest = min(ds.datasource_commits,
                             key=attrgetter('created_at'))
                latest_created_at = format_date(latest.created_at)

            latest_n_bytes = latest.n_bytes if latest else ''
            latest_n_datapoints = latest.n_datapoints if latest else ''
            latest_n_features = latest.n_features if latest else ''

            table.append({'Selection': '*' if active_dc in dcs else '',
                          'ID': format_uuid(ds.id),
                          'Name': ds.name,
                          'Type': ds.type,
                          '# Commits': len(ds.datasource_commits),
                          'Latest Commit Status': status,
                          'Latest Commit Date': latest_created_at,
                          'Latest Commit Bytes': latest_n_bytes,
                          'Latest Commit # Datapoints': latest_n_datapoints,
                          'Latest Commit # Features': latest_n_features
                          })
        click.echo(tabulate(table, headers='keys', tablefmt='presto'))
        click.echo()


@datasource.command("commits")
@click.argument("datasource_id", type=str)
@pass_info
def list_datasource_commits(info, datasource_id):
    """List of all the available datasources"""
    api = ce_api.DatasourcesApi(api_client(info))

    # find closest, this a heavy call for now
    all_ds = api_call(api.get_datasources_api_v1_datasources_get)
    ds_uuid = find_closest_uuid(datasource_id, all_ds)

    ds = api_call(
        api.get_datasource_api_v1_datasources_ds_id_get,
        ds_id=ds_uuid)

    declare('There are {count} different commits for datasource {name}'
            '.\n'.format(count=len(ds.datasource_commits), name=ds.name))

    user = info[constants.ACTIVE_USER]
    if constants.ACTIVE_DATASOURCE_COMMIT in info[user]:
        _, c_id = info[user][constants.ACTIVE_DATASOURCE_COMMIT].split(':')
    else:
        c_id = None

    if ds.datasource_commits:
        table = []
        for commit in ds.datasource_commits:
            status = api_call(
                api.get_datasource_commit_status_api_v1_datasources_ds_id_commits_commit_id_status_get,
                ds.id,
                commit.id,
            )
            table.append({
                'Selection': '*' if commit.id == c_id else '',
                'ID': format_uuid(commit.id),
                'Created At': format_date(commit.created_at),
                'Status': status,
                'Message': commit.message,
                'Bytes': commit.n_bytes,
                '# Datapoints': commit.n_datapoints,
                '# Features': commit.n_features
            })
        click.echo(tabulate(table, headers='keys', tablefmt='presto'))
        click.echo()


@datasource.command('set')
@click.argument("source_id", type=str)
@pass_info
def set_datasource(info, source_id):
    ds_id, c_id = resolve_datasource_commits(info, source_id)

    """Set datasource to be active"""
    active_commit = '{datasource_id}:{commit_id}'.format(
        datasource_id=ds_id,
        commit_id=c_id)

    user = info[constants.ACTIVE_USER]
    info[user][constants.ACTIVE_DATASOURCE_COMMIT] = active_commit
    info.save()

    declare('Active datasource commit set to: {}'.format(
        ':'.join([format_uuid(x) for x in active_commit.split(':')])
    ))


@datasource.command('peek')
@click.argument("source_id", default=None, type=str)
@click.option("-s", "--sample_size", default=10,
              help='Number of samples to peek at')
@pass_info
def peek_datasource(info, source_id, sample_size):
    """Randomly sample datasource and print to console."""
    api = ce_api.DatasourcesApi(api_client(info))

    ds_id, c_id = resolve_datasource_commits(info, source_id)

    declare('Randomly generating {} samples from datasource {}:{}'.format(
        sample_size,
        format_uuid(ds_id),
        format_uuid(c_id)
    ))

    data = api_call(
        api.get_datasource_commit_data_sample_api_v1_datasources_ds_id_commits_commit_id_data_get,
        ds_id=ds_id,
        commit_id=c_id,
        sample_size=sample_size)

    click.echo(tabulate(data, headers='keys', tablefmt='plain'))
