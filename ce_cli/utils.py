import base64
import itertools
import json
import logging
import os
import shutil
import threading
import time
import urllib.request
from pathlib import Path

import click
import py
from dateutil import tz

import ce_api
from ce_api.models.pipeline_run_create import PipelineRunCreate
from ce_api.rest import ApiException
from ce_cli.pretty_yaml import save_pretty_yaml
from ce_standards import constants
from ce_standards.constants import API_HOST
from ce_standards.enums import GDPComponent, PipelineStatusTypes


class Spinner(object):
    """
    Shamlessly copied from: https://github.com/click-contrib/click-spinner
    """
    spinner_cycle = itertools.cycle(['-', '/', '|', '\\'])

    def __init__(self, beep=False, disable=False, force=False,
                 stream=click.get_text_stream('stdout')):
        self.disable = disable
        self.beep = beep
        self.force = force
        self.stream = stream
        self.stop_running = None
        self.spin_thread = None

    def start(self):
        if self.disable:
            return
        if self.stream.isatty() or self.force:
            self.stop_running = threading.Event()
            self.spin_thread = threading.Thread(target=self.init_spin)
            self.spin_thread.start()

    def stop(self):
        if self.spin_thread:
            self.stop_running.set()
            self.spin_thread.join()

    def init_spin(self):
        while not self.stop_running.is_set():
            self.stream.write(next(self.spinner_cycle))
            self.stream.flush()
            time.sleep(0.25)
            self.stream.write('\b')
            self.stream.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return False
        self.stop()
        if self.beep:
            self.stream.write('\7')
            self.stream.flush()
        return False


class Info(dict):
    def __init__(self, *args, **kwargs):
        self.path = py.path.local(
            click.get_app_dir(constants.APP_NAME)).join('info.json')
        super(Info, self).__init__(*args, **kwargs)

    def load(self):
        try:
            self.update(json.loads(self.path.read()))
        except py.error.ENOENT:
            pass

    def save(self):
        self.path.ensure()
        with self.path.open('w') as f:
            f.write(json.dumps(self))


pass_info = click.make_pass_decorator(Info, ensure=True)


def title(text):
    click.echo(click.style(text.upper(), fg='cyan', bold=True, underline=True))


def confirmation(text, *args, **kwargs):
    return click.confirm(click.style(text, fg='yellow'), *args,
                         **kwargs)


def question(text, *args, **kwargs):
    return click.prompt(text=text, *args, **kwargs)


def declare(text):
    click.echo(click.style(text, fg='green'))


def notice(text):
    click.echo(click.style(text, fg='cyan'))


def error(text):
    raise click.ClickException(message=click.style(text, fg='red', bold=True))


def warning(text):
    click.echo(click.style(text, fg='yellow', bold=True))


def check_login_status(info):
    # TODO: APi call to check the login status?
    if constants.ACTIVE_USER not in info or \
            info[constants.ACTIVE_USER] is None:
        raise click.ClickException(
            "You need to login first.\n"
            "In order to login please use: 'cengine auth login'")


def save_config(config, path, no_docs):
    # Save the pulled pipeline locally
    if os.path.isdir(path):
        raise Exception('{} is a directory. Please provide a path to a '
                        'file.'.format(path))
    elif os.path.isfile(path):
        if not path.endswith('.yaml'):
            path += '.yaml'
        if confirmation('A file with the same name exists already.'
                        ' Do you want to overwrite?'):
            save_pretty_yaml(config, path, no_docs)
            declare('Config file saved to: {}'.format(path))
    else:
        if not path.endswith('.yaml'):
            path += '.yaml'
        save_pretty_yaml(config, path, no_docs)
        declare('Config file saved to: {}'.format(path))


def api_client(info):
    active_user = info[constants.ACTIVE_USER]
    config = ce_api.Configuration()
    config.host = API_HOST
    config.access_token = info[active_user][constants.TOKEN]

    return ce_api.ApiClient(config)


def api_call(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except ApiException as e:
        error('{}: {}'.format(e.reason, e.body))
    except Exception as e:
        # raise (e)
        logging.error(str(e))
        click.echo(e)
        error('There is something wrong going on. Please contact '
              'core@maiot.io to get further information.')


def download_artifact(artifact_json, path='/'):
    """
    This will replace the folder with the files if they already exist
    """
    path = Path(path)
    if artifact_json['name'] == '/':
        full_path = path
    else:
        full_path = path / artifact_json['name']

    if artifact_json['is_dir']:
        # TODO: [LOW] Short term fix for empty files being labelled as dirs
        if len(artifact_json['children']) == 0:
            # turn it into a file
            artifact_json['is_dir'] = False
            with open(full_path, 'wb') as f:
                f.write(b'')
        else:
            os.makedirs(full_path, exist_ok=True)
            for child in artifact_json['children']:
                download_artifact(child, path=full_path)
    else:
        # Download the file from `url` and save it locally under `file_name`:
        url = artifact_json['signed_url']
        with urllib.request.urlopen(url) as response, open(full_path,
                                                           'wb') as out_file:

            shutil.copyfileobj(response, out_file)


def format_date(dt, format='%Y-%m-%d %H:%M:%S'):
    if dt is None:
        return ''
    local_zone = tz.tzlocal()
    # make sure this is UTC
    dt = dt.replace(tzinfo=tz.tzutc())
    local_time = dt.astimezone(local_zone)
    return local_time.strftime(format)


def format_timedelta(td):
    if td is None:
        return ''
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    return '{:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds))


def format_uuid(uuid: str, limit: int = 8):
    return uuid[0:limit]


def find_closest_uuid(substr: str, options):
    candidates = [x.id for x in options if x.id.startswith(substr)]
    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) == 0:
        error('No matching IDs found!')
    error('Too many matching IDs.')


def parse_unknown_options(args):
    warning_message = 'Please provide the additional optional with a proper ' \
                      'identifier as the key and the following structure: ' \
                      '--custom_argument="value"'

    assert all(a.startswith('--') for a in args), warning_message
    assert all(len(a.split('=')) == 2 for a in args), warning_message

    p_args = [a.lstrip('--').split('=') for a in args]

    assert all(k.isidentifier() for k, _ in p_args), warning_message

    r_args = {k: v for k, v in p_args}
    assert len(p_args) == len(r_args), 'Replicated arguments!'

    return r_args


def resolve_pipeline_runs(info, source_id, run_type=None):
    ws_id = info[info[constants.ACTIVE_USER]][constants.ACTIVE_WORKSPACE]
    ws_api = ce_api.WorkspacesApi(api_client(info))
    p_api = ce_api.PipelinesApi(api_client(info))

    if len(source_id.split(':')) == 2:
        pipeline_id, run_id = source_id.split(':')
        pipelines = api_call(
            ws_api.get_workspaces_pipelines_api_v1_workspaces_workspace_id_pipelines_get,
            ws_id)
        p_id = find_closest_uuid(pipeline_id, pipelines)

        runs = api_call(
            p_api.get_pipeline_runs_api_v1_pipelines_pipeline_id_runs_get,
            p_id)

        if run_type:
            runs = [r for r in runs if r.pipeline_run_type == run_type]

        r_id = find_closest_uuid(run_id, runs)
    elif len(source_id.split(':')) == 1:
        pipeline_id = source_id
        pipelines = api_call(
            ws_api.get_workspaces_pipelines_api_v1_workspaces_workspace_id_pipelines_get,
            ws_id)
        p_id = find_closest_uuid(pipeline_id, pipelines)

        runs = api_call(
            p_api.get_pipeline_runs_api_v1_pipelines_pipeline_id_runs_get,
            p_id)

        if run_type:
            runs = [r for r in runs if r.pipeline_run_type == run_type]

        runs.sort(key=lambda x: x.start_time)
        if runs:
            r_id = runs[-1].id
        else:
            r_id = None
    else:
        raise ValueError('Unresolvable pipeline ID')

    return p_id, r_id


def resolve_datasource_commits(info, source_id):
    ds_api = ce_api.DatasourcesApi(api_client(info))

    if len(source_id.split(':')) == 2:
        datasource_id, commit_id = source_id.split(':')

        datasources = api_call(
            ds_api.get_datasources_api_v1_datasources_get)
        ds_id = find_closest_uuid(datasource_id, datasources)

        commits = api_call(
            ds_api.get_commits_api_v1_datasources_ds_id_commits_get,
            ds_id)
        c_id = find_closest_uuid(commit_id, commits)
    elif len(source_id.split(':')) == 1:
        datasource_id = source_id

        datasources = api_call(
            ds_api.get_datasources_api_v1_datasources_get)
        ds_id = find_closest_uuid(datasource_id, datasources)

        commits = api_call(
            ds_api.get_commits_api_v1_datasources_ds_id_commits_get,
            ds_id)
        commits.sort(key=lambda x: x.created_at)
        c_id = commits[-1].id
    else:
        raise ValueError('Unresolvable datasource')

    return ds_id, c_id


def get_statistics_html(stats_dict):
    from tensorflow_metadata.proto.v0 import statistics_pb2

    combined_statistics = statistics_pb2.DatasetFeatureStatisticsList()

    for split, stats in stats_dict.items():
        stats_copy = combined_statistics.datasets.add()
        stats_copy.MergeFrom(stats.datasets[0])
        stats_copy.name = split

    protostr = base64.b64encode(
        combined_statistics.SerializeToString()).decode('utf-8')

    html_template = """<iframe id='facets-iframe' width="100%" height="500px"></iframe>
        <script>
        facets_iframe = document.getElementById('facets-iframe');
        facets_html = '<script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"><\/script><link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/master/facets-dist/facets-jupyter.html"><facets-overview proto-input="protostr"></facets-overview>';
        facets_iframe.srcdoc = facets_html;
         facets_iframe.id = "";
         setTimeout(() => {
           facets_iframe.setAttribute('height', facets_iframe.contentWindow.document.body.offsetHeight + 'px')
         }, 1500)
         </script>"""

    # pylint: enable=line-too-long
    html = html_template.replace('protostr', protostr)

    return html


def get_run_stage(components_status):
    """
    Gets run components and maps to a stage
    """
    stage = ' - '
    running_components = [c for c in components_status if
                          c.status == PipelineStatusTypes.Running.name]
    if len(running_components) == 1:
        # only makes sense when there is 1 component running
        running_component = running_components[0].component_type
        if running_component == GDPComponent.SplitGen.name:
            stage = 'Splitting'
        elif running_component == GDPComponent.Trainer.name:
            stage = 'Training'
        elif running_component in [
            GDPComponent.SplitStatistics.name,
            GDPComponent.PreTransformStatistics.name,
            GDPComponent.SequenceStatistics.name
        ]:
            stage = 'Statistics'
        elif running_component in [
            GDPComponent.Transform.name,
            GDPComponent.PreTransform.name,
        ]:
            stage = 'Preprocessing'
        elif running_component == GDPComponent.Trainer.name:
            stage = 'Evaluating'
        elif running_component == GDPComponent.Deployer.name:
            stage = 'Deploying'

    elif len(running_components) == 2:
        # only in one known case
        for r in running_components:
            running_component = r.component_type
            if running_component == GDPComponent.ModelValidator.name:
                stage = 'Validating results'
            elif running_component == GDPComponent.Evaluator.name:
                stage = 'Evaluating'
    return stage


def resolve_workers(distributed, workers, cpus_per_worker):
    if distributed:
        if workers is None or cpus_per_worker is None:
            workers = 5
            cpus_per_worker = 4
    else:
        assert workers is None and cpus_per_worker is None, \
            'If you want to run your pipeline in a distributed setting,' \
            'please use "--distributed" or "-d"'
        workers = 1
        cpus_per_worker = 1

    return workers, cpus_per_worker


def resolve_pipeline_creation(info,
                              pipeline_type,
                              pipeline_,
                              datasource,
                              orchestration_backend,
                              orchestration_args,
                              processing_backend,
                              processing_args,
                              force,
                              additional_args):
    active_user = info[constants.ACTIVE_USER]

    # Initiate all required APIs
    p_api = ce_api.PipelinesApi(api_client(info))

    # Resolving the datasource connection
    if datasource is not None:
        ds_id, c_id = resolve_datasource_commits(info, datasource)
    elif constants.ACTIVE_DATASOURCE_COMMIT in info[active_user]:
        ds_id, c_id = info[active_user][
            constants.ACTIVE_DATASOURCE_COMMIT].split(':')
    else:
        raise AssertionError('Please either select an active datasource '
                             'commit to work with or explicitly define it.')

    declare('Using Datasource Commit:{}'.format(format_uuid(c_id)))

    # Resolving the pipeline uuid
    pipeline_id, _ = resolve_pipeline_runs(info, pipeline_)

    run_create = PipelineRunCreate(
        pipeline_run_type=pipeline_type,
        datasource_commit_id=c_id,
        orchestration_backend=orchestration_backend,
        orchestration_args=orchestration_args,
        processing_backend=processing_backend,
        processing_args=processing_args,
        additional_args=additional_args)

    notice('Provisioning required resources. This might take a few minutes..')

    r = api_call(
        p_api.create_pipeline_run_api_v1_pipelines_pipeline_id_runs_post,
        run_create,
        pipeline_id)

    declare('Run created with ID: {id}!\n'.format(id=format_uuid(r.id)))

    declare("Use 'cengine pipeline status -p {}' to check on its "
            "status".format(format_uuid(pipeline_id)))


def check_datasource_commit(info):
    user = info[constants.ACTIVE_USER]
    if constants.ACTIVE_DATASOURCE_COMMIT in info[user]:
        active_dsc = info[user][
            constants.ACTIVE_DATASOURCE_COMMIT]
        ds_id, c_id = active_dsc.split(':')

        ds_api = ce_api.DatasourcesApi(api_client(info))
        ds = api_call(ds_api.get_datasource_api_v1_datasources_ds_id_get,
                      ds_id)

        click.echo('Currently, the active datasource is:')

        declare('Datasource Name: {}\n'
                'Datasource ID: {}\n'
                'Commit ID: {}\n'.format(ds.name,
                                         format_uuid(ds_id),
                                         format_uuid(c_id)))
    else:
        raise click.ClickException(message=error(
            "You have not selected a datasource to work on.\n"
            "You can either select one by using the argument called "
            "'datasource'\n "
            "Or you can use 'cengine datasource list' to see the "
            "possible options \n and 'cengine datasource set' to "
            "select one.\n"))


def check_workspace(info):
    user = info[constants.ACTIVE_USER]
    if constants.ACTIVE_WORKSPACE in info[user]:
        ws_api = ce_api.WorkspacesApi(api_client(info))
        ws = api_call(
            ws_api.get_workspace_api_v1_workspaces_workspace_id_get,
            info[user][constants.ACTIVE_WORKSPACE])

        click.echo('Currently, the active workspace is:')
        declare('Workspace Name: {}\n'
                'Workspace ID: {}\n'.format(ws.name,
                                            format_uuid(ws.id)))
    else:
        raise click.ClickException(message=error(
            "You have not set a workspace to work on.\n"
            "'cengine workspace list' to see the possible options \n"
            "'cengine workspace set' to select a workspace \n"))
