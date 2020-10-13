import click
import ce_api
import base64
import os
from ce_cli.cli import cli, pass_info
from ce_cli.utils import check_login_status
from ce_cli.utils import api_client, api_call
from ce_api.models import FunctionCreate, FunctionVersionCreate
from ce_cli.utils import declare, notice
from tabulate import tabulate
from ce_cli.utils import format_uuid, find_closest_uuid


@cli.group()
@pass_info
def function(info):
    """Integrate your own custom logic to the Core Engine"""
    check_login_status(info)


@function.command('create')
@click.argument('name', type=str)
@click.argument('local_path', type=click.Path(exists=True))
@click.argument('func_type', type=str)
@click.argument('udf_name', type=str)
@click.option('--message', type=str, help='Description of the function',
              default='')
@pass_info
def create_function(info, local_path, name, func_type, udf_name, message):
    """Register a custom function to use with the Core Engine"""
    click.echo('Registering the function {}.'.format(udf_name))

    with open(local_path, 'rb') as file:
        data = file.read()
    encoded_file = base64.b64encode(data).decode()

    api = ce_api.FunctionsApi(api_client(info))
    api_call(api.create_function_api_v1_functions_post,
             FunctionCreate(name=name,
                            function_type=func_type,
                            udf_path=udf_name,
                            message=message,
                            file_contents=encoded_file))

    declare('Function registered.')


@function.command('update')
@click.argument('function_id', type=str)
@click.argument('local_path', type=click.Path(exists=True))
@click.argument('udf_name', type=str)
@click.option('--message', type=str, help='Description of the function',
              default='')
@pass_info
def update_function(info, function_id, local_path, udf_name, message):
    """Add a new version to a function and update it"""
    click.echo('Updating the function {}.'.format(
        format_uuid(function_id)))

    api = ce_api.FunctionsApi(api_client(info))

    f_list = api_call(api.get_functions_api_v1_functions_get)
    f_uuid = find_closest_uuid(function_id, f_list)

    with open(local_path, 'rb') as file:
        data = file.read()
    encoded_file = base64.b64encode(data).decode()

    api_call(
        api.create_function_version_api_v1_functions_function_id_versions_post,
        FunctionVersionCreate(udf_path=udf_name,
                              message=message,
                              file_contents=encoded_file),
        f_uuid)

    declare('Function updated!')


@function.command('list')
@pass_info
def list_functions(info):
    """List the given custom functions"""
    api = ce_api.FunctionsApi(api_client(info))
    f_list = api_call(api.get_functions_api_v1_functions_get)
    declare('You have declared {count} different '
            'function(s) so far. \n'.format(count=len(f_list)))

    if f_list:
        table = []
        for f in f_list:
            table.append({'ID': format_uuid(f.id),
                          'Name': f.name,
                          'Type': f.function_type,
                          'Created At': f.created_at})
        click.echo(tabulate(table, headers='keys', tablefmt='presto'))
        click.echo()


@function.command('versions')
@click.argument('function_id', type=str)
@pass_info
def list_versions(info, function_id):
    """List of versions for a selected custom function"""
    api = ce_api.FunctionsApi(api_client(info))
    f_list = api_call(api.get_functions_api_v1_functions_get)
    f_uuid = find_closest_uuid(function_id, f_list)

    v_list = api_call(
        api.get_function_versions_api_v1_functions_function_id_versions_get,
        f_uuid)

    declare('Function with {id} has {count} '
            'versions.\n'.format(id=format_uuid(function_id),
                                 count=len(v_list)))

    if v_list:
        table = []
        for v in v_list:
            table.append({'ID': format_uuid(v.id),
                          'Created At': v.created_at,
                          'Description': v.message})
        click.echo(tabulate(table, headers='keys', tablefmt='presto'))
        click.echo()


@function.command('pull')
@click.argument('function_id', type=str)
@click.argument('version_id', type=str)
@click.option('--output_path', default=None, type=click.Path(),
              help='Path to save the custom function')
@pass_info
def pull_function_version(info, function_id, version_id, output_path):
    """Download a version of a given custom function"""
    api = ce_api.FunctionsApi(api_client(info))

    # Infer the function uuid and name
    f_list = api_call(api.get_functions_api_v1_functions_get)
    f_uuid = find_closest_uuid(function_id, f_list)
    f_name = [f.name for f in f_list if f.id == f_uuid][0]

    # Infer the version uuid
    v_list = api_call(
        api.get_function_versions_api_v1_functions_function_id_versions_get,
        f_uuid)
    v_uuid = find_closest_uuid(version_id, v_list)

    notice('Downloading the function with the following parameters: \n'
           'Name: {f_name}\n'
           'function_id: {f_id}\n'
           'version_id: {v_id}\n'.format(f_name=f_name,
                                         f_id=format_uuid(f_uuid),
                                         v_id=format_uuid(v_uuid)))

    # Get the file and write it to the output path
    encoded_file = api_call(
        api.get_function_version_api_v1_functions_function_id_versions_version_id_get,
        f_uuid,
        v_uuid)

    # Derive the output path and download
    if output_path is None:
        output_path = os.path.join(os.getcwd(), '{}@{}.py'.format(f_name,
                                                                  v_uuid))

    with open(output_path, 'wb') as f:
        f.write(base64.b64decode(encoded_file.file_contents))

    declare('File downloaded to {}'.format(output_path))
