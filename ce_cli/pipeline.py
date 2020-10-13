import os
from datetime import datetime, timezone
from pathlib import Path

import click
import yaml
from tabulate import tabulate

import ce_api
from ce_api.models import PipelineCreate
from ce_cli import utils
from ce_cli.cli import cli
from ce_standards import constants
from ce_standards.enums import GDPComponent, PipelineRunTypes


@cli.group()
@utils.pass_info
def pipeline(info):
    """Create, configure and deploy pipeline runs"""
    utils.check_login_status(info)
    utils.check_workspace(info)


@pipeline.command('pull')
@click.argument('pipeline_id', type=click.STRING)
@click.option('--output_path',
              default=os.path.join(os.getcwd(), 'ce_config.yaml'),
              help='Path to save the config file, default: working directory')
@click.option('--no_docs', is_flag=True, default=False,
              help='Save file without additional documentation')
@utils.pass_info
def pull_pipeline(info, pipeline_id, output_path, no_docs):
    """Copy the configuration of a registered pipeline"""
    p_api = ce_api.PipelinesApi(utils.api_client(info))
    ws_api = ce_api.WorkspacesApi(utils.api_client(info))

    active_user = info[constants.ACTIVE_USER]
    ws_id = info[active_user][constants.ACTIVE_WORKSPACE]

    all_ps = utils.api_call(
        ws_api.get_workspaces_pipelines_api_v1_workspaces_workspace_id_pipelines_get,
        ws_id)
    p_uuid = utils.find_closest_uuid(pipeline_id, all_ps)

    utils.declare('Pulling pipeline: {}'.format(utils.format_uuid(p_uuid)))

    pp = utils.api_call(p_api.get_pipeline_api_v1_pipelines_pipeline_id_get,
                        pipeline_id=p_uuid)

    # Short term fix for these getting into the exp_config
    c = pp.pipeline_config
    if 'bq_args' in c:
        c.pop('bq_args')
    if 'ai_platform_training_args' in c:
        c.pop('ai_platform_training_args')

    utils.save_config(c, output_path, no_docs)


@pipeline.command('push')
@click.argument('config_path')
@click.argument('pipeline_name')
@utils.pass_info
def push_pipeline(info, config_path, pipeline_name):
    """Register a pipeline with the selected configuration"""
    active_user = info[constants.ACTIVE_USER]
    ws_id = info[active_user][constants.ACTIVE_WORKSPACE]

    try:
        with open(config_path, 'rt', encoding='utf8') as f:
            config = yaml.load(f)
    except:
        utils.error('Badly formatted YAML!')

    api = ce_api.PipelinesApi(utils.api_client(info))
    p = utils.api_call(api.create_pipeline_api_v1_pipelines_post,
                       PipelineCreate(name=pipeline_name,
                                      pipeline_config=config,
                                      workspace_id=ws_id))

    utils.declare('Pipeline pushed successfully!'.format(
        id=utils.format_uuid(p.id)))

    utils.declare(
        "Use `cengine pipeline train {} --datasource DS_COMMIT` "
        "to launch a training pipeline!".format(utils.format_uuid(p.id)))


@pipeline.command('test')
@click.argument('pipeline_id', type=click.STRING)
@click.option('--datasource', required=False, default=None, type=str)
@click.option('--orchestration_backend', required=False, default=None, type=str)
@click.option('--orchestration_args', required=False, default={}, type=dict)
@click.option('--processing_backend', required=False, default=None, type=str)
@click.option('--processing_args', required=False, default={}, type=dict)
@click.option('--training_backend', required=False, default=None, type=str)
@click.option('--training_args', required=False, default={}, type=dict)
@click.option('--serving_backend', required=False, default=None, type=str)
@click.option('--serving_args', required=False, default={}, type=dict)
@click.option('-f', '--force', is_flag=True, default=False)
@utils.pass_info
def test_pipeline(info,
                  pipeline_id,
                  datasource,
                  orchestration_backend,
                  orchestration_args,
                  processing_backend,
                  processing_args,
                  training_backend,
                  training_args,
                  serving_backend,
                  serving_args,
                  force):
    """Initiate a test run of a selected pipeline"""
    if datasource is None:
        utils.check_datasource_commit(info)
    utils.resolve_pipeline_creation(info=info,
                                    pipeline_type=PipelineRunTypes.test.name,
                                    pipeline_=pipeline_id,
                                    datasource=datasource,
                                    orchestration_backend=orchestration_backend,
                                    orchestration_args=orchestration_args,
                                    processing_backend=processing_backend,
                                    processing_args=processing_args,
                                    force=force,
                                    additional_args={
                                        'training_backend': training_backend,
                                        'training_args': training_args,
                                        'serving_backend': serving_backend,
                                        'serving_args': serving_args})


@pipeline.command('train')
@click.argument('pipeline_id', type=click.STRING)
@click.option('--datasource', required=False, default=None, type=str)
@click.option('--orchestration_backend', required=False, default=None, type=str)
@click.option('--orchestration_args', required=False, default={}, type=dict)
@click.option('--processing_backend', required=False, default=None, type=str)
@click.option('--processing_args', required=False, default={}, type=dict)
@click.option('--training_backend', required=False, default=None, type=str)
@click.option('--training_args', required=False, default={}, type=dict)
@click.option('--serving_backend', required=False, default=None, type=str)
@click.option('--serving_args', required=False, default={}, type=dict)
@click.option('-f', '--force', is_flag=True, default=False)
@utils.pass_info
def train_pipeline(info,
                   pipeline_id,
                   datasource,
                   orchestration_backend,
                   orchestration_args,
                   processing_backend,
                   processing_args,
                   training_backend,
                   training_args,
                   serving_backend,
                   serving_args,
                   force):
    """Initiate a training run of a selected pipeline"""
    if datasource is None:
        utils.check_datasource_commit(info)
    utils.resolve_pipeline_creation(info=info,
                                    pipeline_type=PipelineRunTypes.training.name,
                                    pipeline_=pipeline_id,
                                    datasource=datasource,
                                    orchestration_backend=orchestration_backend,
                                    orchestration_args=orchestration_args,
                                    processing_backend=processing_backend,
                                    processing_args=processing_args,
                                    force=force,
                                    additional_args={
                                        'training_backend': training_backend,
                                        'training_args': training_args,
                                        'serving_backend': serving_backend,
                                        'serving_args': serving_args})


@pipeline.command('infer')
@click.argument('pipeline_', type=click.STRING)
@click.option('--datasource', required=False, default=None, type=str)
@click.option('--orchestration_backend', required=False, default=None, type=str)
@click.option('--orchestration_args', required=False, default=None, type=dict)
@click.option('--processing_backend', required=False, default=None, type=str)
@click.option('--processing_args', required=False, default=None, type=dict)
@click.option('-f', '--force', is_flag=True, default=False)
@utils.pass_info
def infer_pipeline(info,
                   pipeline_,
                   datasource,
                   orchestration_backend,
                   orchestration_args,
                   processing_backend,
                   processing_args,
                   force):
    """Initiate a batch inference run of a selected pipeline"""
    # Resolving additional args
    _, run_id = utils.resolve_pipeline_runs(
        info,
        pipeline_,
        run_type=PipelineRunTypes.training.name)

    if datasource is None:
        utils.check_datasource_commit(info)

    """Initiate an infer run of a selected pipeline"""
    utils.resolve_pipeline_creation(info=info,
                                    pipeline_type=PipelineRunTypes.infer.name,
                                    pipeline_=pipeline_,
                                    datasource=datasource,
                                    orchestration_backend=orchestration_backend,
                                    orchestration_args=orchestration_args,
                                    processing_backend=processing_backend,
                                    processing_args=processing_args,
                                    force=force,
                                    additional_args={'run_id': run_id})


@pipeline.command('list')
@click.option('--pipeline_id', default=None, type=str)
@click.option('--ignore_empty', is_flag=True, default=False)
@utils.pass_info
def list_pipelines(info, pipeline_id, ignore_empty):
    """List of registered pipelines"""
    utils.notice('Fetching pipeline(s). This might take a few seconds... \n')
    active_user = info[constants.ACTIVE_USER]
    ws = info[active_user][constants.ACTIVE_WORKSPACE]
    ws_api = ce_api.WorkspacesApi(utils.api_client(info))
    p_api = ce_api.PipelinesApi(utils.api_client(info))
    d_api = ce_api.DatasourcesApi(utils.api_client(info))

    pipelines = utils.api_call(
        ws_api.get_workspaces_pipelines_api_v1_workspaces_workspace_id_pipelines_get,
        ws)

    if pipeline_id is not None:
        pipeline_id = utils.find_closest_uuid(pipeline_id, pipelines)

    pipelines.sort(key=lambda x: x.created_at)
    for p in pipelines:
        write_check = (len(p.pipeline_runs) > 0 or not ignore_empty) and \
                      (pipeline_id is None or pipeline_id == p.id)

        if write_check:
            # THIS WHOLE THING IS HERE FOR A REASON!!!!!!
            title = 'PIPELINE NAME: {} PIPELINE ID: {}'.format(
                p.name, utils.format_uuid(p.id))
            utils.declare(title)
            utils.declare('-' * len(title))
            if len(p.pipeline_runs) == 0:
                click.echo('No runs for this pipeline yet!')
            else:
                table = []
                for r in p.pipeline_runs:
                    author = utils.api_call(
                        p_api.get_pipeline_run_user_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_user_get,
                        p.id,
                        r.id)

                    # Resolve datasource
                    ds_commit = utils.api_call(
                        d_api.get_single_commit_api_v1_datasources_commits_commit_id_get,
                        r.datasource_commit_id)
                    ds = utils.api_call(
                        d_api.get_datasource_api_v1_datasources_ds_id_get,
                        ds_commit.datasource_id)

                    table.append({
                        'RUN ID': utils.format_uuid(r.id),
                        'TYPE': r.pipeline_run_type,
                        'CPUs PER WORKER': r.cpus_per_worker,
                        'WORKERS': r.workers,
                        'DATASOURCE': '{}_{}'.format(
                            ds.name,
                            utils.format_uuid(r.datasource_commit_id)),
                        'AUTHOR': author.email,
                        'CREATED AT': utils.format_date(r.start_time),
                    })
                click.echo(tabulate(table, headers='keys', tablefmt='plain'))
            click.echo('\n')


@pipeline.command('status')
@click.option('--pipeline_id', default=None, type=str)
@utils.pass_info
def get_pipeline_status(info, pipeline_id):
    """Get status of started pipelines"""
    utils.notice('Fetching pipeline(s). This might take a few seconds.. \n')
    active_user = info[constants.ACTIVE_USER]
    ws = info[active_user][constants.ACTIVE_WORKSPACE]

    ws_api = ce_api.WorkspacesApi(utils.api_client(info))
    p_api = ce_api.PipelinesApi(utils.api_client(info))
    d_api = ce_api.DatasourcesApi(utils.api_client(info))

    pipelines = utils.api_call(
        ws_api.get_workspaces_pipelines_api_v1_workspaces_workspace_id_pipelines_get,
        ws)

    if pipeline_id is not None:
        pipeline_id = utils.find_closest_uuid(pipeline_id, pipelines)

    pipelines.sort(key=lambda x: x.created_at)
    for p in pipelines:
        write_check = (len(p.pipeline_runs) > 0) and \
                      (pipeline_id is None or pipeline_id == p.id)

        if write_check:
            title = 'PIPELINE NAME: {} PIPELINE ID: {}'.format(
                p.name, utils.format_uuid(p.id))
            utils.declare(title)
            utils.declare('-' * len(title))

            table = []
            for r in p.pipeline_runs:
                run = utils.api_call(
                    p_api.get_pipeline_run_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_get,
                    p.id,
                    r.id)

                # Resolve datasource
                ds_commit = utils.api_call(
                    d_api.get_single_commit_api_v1_datasources_commits_commit_id_get,
                    r.datasource_commit_id)
                ds = utils.api_call(
                    d_api.get_datasource_api_v1_datasources_ds_id_get,
                    ds_commit.datasource_id)

                if run.end_time:
                    td = run.end_time - run.start_time
                else:
                    td = datetime.now(timezone.utc) - run.start_time

                # # Resolve component status
                # stage = utils.get_run_stage(run.pipeline_components)

                table.append({
                    'RUN ID': utils.format_uuid(run.id),
                    'TYPE': run.pipeline_run_type,
                    'STATUS': run.status,
                    # 'STAGE': stage,
                    'DATASOURCE': '{}_{}'.format(
                        ds.name, utils.format_uuid(run.datasource_commit_id)),
                    'DATAPOINTS': '{}'.format(ds_commit.n_datapoints),
                    # 'RUNNING STAGE': stage,
                    'START TIME': utils.format_date(run.start_time),
                    'DURATION': utils.format_timedelta(td),
                })

            click.echo(tabulate(table, headers='keys', tablefmt='plain'))
            click.echo('\n')


@pipeline.command('statistics')
@click.argument('pipeline_', type=click.STRING)
@utils.pass_info
def statistics_pipeline(info, pipeline_):
    """Serve the statistics of a pipeline run"""

    p_uuid, r_uuid = utils.resolve_pipeline_runs(info,
                                                 pipeline_,
                                                 run_type=PipelineRunTypes.training.name)

    utils.notice('Generating statistics for the pipeline run ID {}. If your '
                 'browser opens up to a blank window, please refresh '
                 'the page once.'.format(utils.format_uuid(r_uuid)))

    api = ce_api.PipelinesApi(utils.api_client(info))
    stat_artifact = utils.api_call(
        api.get_pipeline_artifacts_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_artifacts_component_type_get,
        pipeline_id=p_uuid,
        pipeline_run_id=r_uuid,
        component_type=GDPComponent.SplitStatistics.name)

    ws_id = info[info[constants.ACTIVE_USER]][constants.ACTIVE_WORKSPACE]
    path = Path(click.get_app_dir(constants.APP_NAME),
                'statistics',
                str(ws_id),
                p_uuid,
                r_uuid)
    utils.download_artifact(artifact_json=stat_artifact[0].to_dict(),
                            path=path)

    import tensorflow as tf
    from tensorflow_metadata.proto.v0 import statistics_pb2
    import panel as pn

    result = {}
    for split in os.listdir(path):
        stats_path = os.path.join(path, split, 'stats_tfrecord')
        serialized_stats = next(tf.compat.v1.io.tf_record_iterator(stats_path))
        stats = statistics_pb2.DatasetFeatureStatisticsList()
        stats.ParseFromString(serialized_stats)
        dataset_list = statistics_pb2.DatasetFeatureStatisticsList()
        for i, d in enumerate(stats.datasets):
            d.name = split
            dataset_list.datasets.append(d)
        result[split] = dataset_list
    h = utils.get_statistics_html(result)

    pn.serve(panels=pn.pane.HTML(h, width=1200), show=True)


@pipeline.command('model')
@click.argument('pipeline_', type=click.STRING)
@click.option('--output_path', required=True, help='Path to save the model')
@utils.pass_info
def model_pipeline(info, pipeline_, output_path):
    """Download the trained model to a specified location"""
    if os.path.exists(output_path) and os.path.isdir(output_path):
        if not [f for f in os.listdir(output_path) if
                not f.startswith('.')] == []:
            utils.error("Output path must be an empty directory!")
    if os.path.exists(output_path) and not os.path.isdir(output_path):
        utils.error("Output path must be an empty directory!")
    if not os.path.exists(output_path):
        "Creating directory {}..".format(output_path)

    p_uuid, r_uuid = utils.resolve_pipeline_runs(info, pipeline_)

    utils.notice('Downloading the trained model from pipeline run '
                 'ID {}. This might take some time if the model '
                 'resources are significantly large in size.\nYour patience '
                 'is much appreciated!'.format(utils.format_uuid(r_uuid)))

    api = ce_api.PipelinesApi(utils.api_client(info))
    artifact = utils.api_call(
        api.get_pipeline_artifacts_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_artifacts_component_type_get,
        pipeline_id=p_uuid,
        pipeline_run_id=r_uuid,
        component_type=GDPComponent.Deployer.name)

    spin = utils.Spinner()
    spin.start()
    if len(artifact) == 1:
        utils.download_artifact(artifact_json=artifact[0].to_dict(),
                                path=output_path)
        spin.stop()
    else:
        utils.error('Something unexpected happened! Please contact '
                    'core@maiot.io to get further information.')

    utils.declare('Model downloaded to: {}'.format(output_path))
    # TODO: [LOW] Make the Tensorflow version more dynamic
    utils.declare('Please note that the model is saved as a SavedModel '
                  'Tensorflow artifact, trained on Tensoflow 2.1.0.')


@pipeline.command('template')
@click.option('--datasource', required=False, default=None, type=str,
              help='The selected datasource')
@click.option('--output_path',
              default=os.path.join(os.getcwd(), 'template_config.yaml'),
              help='Path to save the config file, default: working directory')
@click.option('--no_docs', is_flag=True, default=False,
              help='Save file without additional documentation')
@click.option('--no_datasource', is_flag=True, default=False,
              help='Save template without connecting to the datasource')
@utils.pass_info
def template_pipeline(info, datasource, output_path, no_docs, no_datasource):
    """Copy the configuration of a registered pipeline"""
    # TODO: with the info we can do datasource specific templates later on
    from ce_cli.pretty_yaml import TEMPLATE_CONFIG
    if not no_datasource:
        active_user = info[constants.ACTIVE_USER]
        if datasource is not None:
            from ce_cli.utils import resolve_datasource_commits
            ds_id, c_id = resolve_datasource_commits(info, datasource)
        elif constants.ACTIVE_DATASOURCE_COMMIT in info[active_user]:
            ds_id, c_id = info[active_user][
                constants.ACTIVE_DATASOURCE_COMMIT].split(':')
        else:
            raise AssertionError('Please either select an active datasource '
                                 'commit to work on or explicitly define it.')

        api = ce_api.DatasourcesApi(utils.api_client(info))
        schema = utils.api_call(
            api.get_datasource_commit_schema_api_v1_datasources_ds_id_commits_commit_id_schema_get,
            ds_id=ds_id,
            commit_id=c_id)

        from ce_standards.standard_experiment import GlobalKeys
        TEMPLATE_CONFIG[GlobalKeys.FEATURES] = {f: {} for f in schema}

    utils.save_config(TEMPLATE_CONFIG, output_path, no_docs)


@pipeline.command('logs')
@click.argument('source_id', type=click.STRING)
@utils.pass_info
def logs_pipeline(info, source_id):
    """Get link to the logs of a pipeline"""

    p_uuid, r_uuid = utils.resolve_pipeline_runs(info, source_id)
    utils.notice(
        'Generating logs url for the pipeline run ID {}. Please visit the '
        'url for all your logs.'.format(utils.format_uuid(r_uuid)))

    api = ce_api.PipelinesApi(utils.api_client(info))
    logs_url = utils.api_call(
        api.get_pipeline_logs_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_logs_get,
        pipeline_id=p_uuid,
        pipeline_run_id=r_uuid
    )

    click.echo(logs_url)
