import base64
import logging
import os
import types
from datetime import datetime, timezone
from importlib import machinery
from pathlib import Path
from typing import Text, Dict, Any, Union, List
import json
import yaml
import click
import nbformat as nbf

import ce_api
from ce_cli import evaluation
from ce_cli.utils import Spinner, download_artifact, get_statistics_html
from ce_standards import constants
from ce_standards.enums import GDPComponent
from ce_standards.enums import PipelineRunTypes
from ce_standards.standard_experiment import GlobalKeys
from cengine.models import Datasource, DatasourceCommit
from cengine.models import Function, FunctionVersion
from cengine.models import Pipeline, PipelineRun
from cengine.models import User
from cengine.models import Workspace
from cengine.models import Provider
from cengine.models import Backend
from cengine.pipeline_config import PipelineConfig
from cengine.utils import api_utils, print_utils, client_utils

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class Client:

    # AUTH ####################################################################

    def __init__(self,
                 username: Text,
                 password: Text = None):
        """ Initialization of the Core Engine client
        """

        # Temp client to make the log-in call
        config = ce_api.Configuration()
        config.host = constants.API_HOST
        temp_client = ce_api.ApiClient(config)

        api_instance = ce_api.LoginApi(temp_client)
        output = api_utils.api_call(
            func=api_instance.login_access_token_api_v1_login_access_token_post,
            username=username,
            password=password)
        logging.info('Log-in successful!')

        config.access_token = output.access_token

        # Final client with the right token
        self.client = ce_api.ApiClient(config)

    def get_user(self):
        api = ce_api.UsersApi(self.client)
        user = api_utils.api_call(api.get_loggedin_user_api_v1_users_me_get)
        return User(**user.to_dict())

    # PROVIDERS ###############################################################

    def get_providers(self,
                      **kwargs) -> List[Provider]:
        """ Get a list of registered providers
        """
        api = ce_api.ProvidersApi(self.client)
        p_list = api_utils.api_call(
            func=api.get_loggedin_provider_api_v1_providers_get)
        providers = [Provider(**p.to_dict()) for p in p_list]

        if kwargs:
            providers = client_utils.filter_objects(providers, **kwargs)
        return providers

    def get_provider_by_id(self,
                           id: Text) -> Provider:
        """ Get a provider with a specific id
        """
        return self.get_providers(id=id)[0]

    def get_provider_by_name(self,
                             name: Text) -> Provider:
        """ Get a provider with a specific name
        """
        return self.get_providers(name=name)[0]

    def create_provider(self,
                        name: Text,
                        provider_type: Text,
                        args: Dict) -> Provider:
        """ Create a new provider in db
        """
        api = ce_api.ProvidersApi(self.client)

        for k in args:
            v = args[k]
            if v.endswith('.json') and os.path.isfile(v):
                args[k] = json.load(open(v))
            if v.endswith('.yaml') and os.path.isfile(v):
                args[k] = yaml.load(open(v))

        p = api_utils.api_call(
            func=api.create_provider_api_v1_providers_post,
            body=Provider.creator(name, provider_type, args=args))
        return Provider(**p.to_dict())

    # BACKENDS ################################################################

    def get_backends(self,
                     **kwargs) -> List[Backend]:
        """ Get a list of registered backends (based on a backend_class)
        """
        api = ce_api.BackendsApi(self.client)
        b_list = api_utils.api_call(
            func=api.get_loggedin_backend_api_v1_backends_get)

        backends = [Backend(**b.to_dict()) for b in b_list]

        if kwargs:
            backends = client_utils.filter_objects(backends, **kwargs)
        return backends

    def get_backend_by_name(self,
                            name: Text) -> Backend:
        """ Get a backend with a specific name
        """
        return self.get_backends(name=name)[0]

    def get_backend_by_id(self,
                          id: Text) -> Backend:
        """ Get a backend with a specific id
        """
        return self.get_backends(id=id)[0]

    def create_backend(self,
                       name: Text,
                       provider_id: Text,
                       backend_class: Text,
                       backend_type: Text,
                       args: Dict[Text, Any]) -> Backend:
        """ Create a new backend
        """
        api = ce_api.BackendsApi(self.client)
        b = api_utils.api_call(
            func=api.create_backend_api_v1_backends_post,
            body=Backend.creator(name=name,
                                 provider_id=provider_id,
                                 backend_class=backend_class,
                                 backend_type=backend_type,
                                 args=args))
        return Backend(**b.to_dict())

    # WORKSPACES ##############################################################

    def get_workspaces(self,
                       **kwargs) -> List[Workspace]:
        """ Get a list of workspaces
        """

        api = ce_api.WorkspacesApi(self.client)
        ws_list = api_utils.api_call(
            func=api.get_loggedin_workspaces_api_v1_workspaces_get)

        workspaces = [Workspace(**ws.to_dict()) for ws in ws_list]
        if kwargs:
            workspaces = client_utils.filter_objects(workspaces, **kwargs)
        return workspaces

    def get_workspace_by_name(self,
                              name: Text) -> Workspace:
        """ Get a workspace with a specific name
        """
        return self.get_workspaces(name=name)[0]

    def get_workspace_by_id(self,
                            id: Text) -> Workspace:
        """ Get a workspace with a specific id
        """
        return self.get_workspaces(id=id)[0]

    def create_workspace(self,
                         name: Text,
                         provider_id: Text) -> Workspace:
        """ Create a new workspace in DB
        """
        api = ce_api.WorkspacesApi(self.client)
        ws = api_utils.api_call(
            func=api.create_workspace_api_v1_workspaces_post,
            body=Workspace.creator(name, provider_id))
        return Workspace(**ws.to_dict())

    # DATASOURCES #############################################################

    def get_datasources(self,
                        **kwargs) -> List[Datasource]:
        """ Get a list of datasources
        """
        api = ce_api.DatasourcesApi(self.client)
        ds_list = api_utils.api_call(
            func=api.get_datasources_api_v1_datasources_get)
        datasources = [Datasource(**ds.to_dict()) for ds in ds_list]

        if kwargs:
            datasources = client_utils.filter_objects(datasources, **kwargs)
        return datasources

    def get_datasource_by_name(self,
                               name: Text) -> Datasource:
        """ Get a workspace with a specific name
        """
        return self.get_datasources(name=name)[0]

    def get_datasource_by_id(self,
                             id: Text) -> Datasource:
        """ Get a workspace with a specific id
        """
        return self.get_datasources(id=id)[0]

    def create_datasource(self,
                          name: Text,
                          type: Text,
                          source: Text,
                          provider_id: Text,
                          args: Dict[Text, Any]) -> Datasource:
        """ Create a new workspace in DB
        """
        api = ce_api.DatasourcesApi(self.client)
        ds = api_utils.api_call(
            func=api.create_datasource_api_v1_datasources_post,
            body=Datasource.creator(name=name,
                                    type_=type,
                                    source=source,
                                    provider_id=provider_id,
                                    args=args))
        return Datasource(**ds.to_dict())

    # DATASOURCE COMMITS ######################################################

    def get_datasource_commits(self,
                               datasource_id: Text,
                               **kwargs) -> List[DatasourceCommit]:

        api = ce_api.DatasourcesApi(self.client)

        dsc_list = api_utils.api_call(
            api.get_commits_api_v1_datasources_ds_id_commits_get,
            datasource_id)

        commits = [DatasourceCommit(**dsc.to_dict()) for dsc in dsc_list]

        if kwargs:
            commits = client_utils.filter_objects(commits, **kwargs)
        return commits

    def get_datasource_commit_by_id(self,
                                    datasource_id: Text,
                                    commit_id: Text) -> DatasourceCommit:
        """ Get a workspace with a specific id
        """
        return self.get_datasource_commits(datasource_id=datasource_id,
                                           id=commit_id)[0]

    def commit_datasource(self,
                          datasource_id: Text,
                          message: Text = None,
                          schema: Dict[Text, Any] = None,
                          orchestration_backend: Text = None,
                          orchestration_args: Dict = None,
                          processing_backend: Text = None,
                          processing_args: Dict = None) -> DatasourceCommit:
        """ Create a new workspace in DB
        """
        api = ce_api.DatasourcesApi(self.client)

        if schema is None:
            schema = {}
        else:
            assert isinstance(schema, dict), 'The schema should be a dict.'

        commit = api_utils.api_call(
            api.create_datasource_commit_api_v1_datasources_ds_id_commits_post,
            DatasourceCommit.creator(
                message=message,
                schema=schema,
                orchestration_backend=orchestration_backend,
                orchestration_args=orchestration_args,
                processing_backend=processing_backend,
                processing_args=processing_args),
            ds_id=datasource_id,
        )

        return DatasourceCommit(**commit.to_dict())

    def peek_datasource_commit(self,
                               datasource_id: Text,
                               datasource_commit_id: Text,
                               size: int = 10) -> List[Dict[Text, Any]]:
        ds_api = ce_api.DatasourcesApi(self.client)
        data = api_utils.api_call(
            ds_api.get_datasource_commit_data_sample_api_v1_datasources_ds_id_commits_commit_id_data_get,
            ds_id=datasource_id,
            commit_id=datasource_commit_id,
            sample_size=size)
        return data

    # PIPELINES ###############################################################

    def get_pipelines(self,
                      workspace_id: Text,
                      **kwargs) -> List[Pipeline]:

        api = ce_api.WorkspacesApi(self.client)
        all_ps = api_utils.api_call(
            api.get_workspaces_pipelines_api_v1_workspaces_workspace_id_pipelines_get,
            workspace_id)

        pipelines = [Pipeline(**p.to_dict()) for p in all_ps]

        if kwargs:
            pipelines = client_utils.filter_objects(pipelines, **kwargs)
        return pipelines

    def get_pipeline_by_name(self,
                             workspace_id: Text,
                             name: Text) -> Pipeline:
        """ Get a pipeline with a specific name
        """
        return self.get_pipelines(workspace_id=workspace_id,
                                  name=name)[0]

    def get_pipeline_by_id(self,
                           workspace_id: Text,
                           id: Text) -> Pipeline:
        """ Get a pipeline with a specific id
        """
        return self.get_pipelines(workspace_id=workspace_id,
                                  id=id)[0]

    def push_pipeline(self,
                      name: Text,
                      workspace_id: Text,
                      config: Union[Dict, PipelineConfig]) -> Pipeline:

        if isinstance(config, PipelineConfig):
            # config.check_completion()
            pass
        elif isinstance(config, dict):
            config = PipelineConfig(**config)
            # config.check_completion()
        else:
            raise ValueError('Please provide either a dict value or an '
                             'instance of cengine.PipelineConfig for '
                             'the config')

        api = ce_api.PipelinesApi(self.client)
        p = api_utils.api_call(
            func=api.create_pipeline_api_v1_pipelines_post,
            body=Pipeline.creator(name=name,
                                  pipeline_config=config.to_serial(),
                                  workspace_id=workspace_id))
        return Pipeline(**p.to_dict())

    def pull_pipeline(self,
                      pipeline_id: Text) -> PipelineConfig:

        api = ce_api.PipelinesApi(self.client)
        pp = api_utils.api_call(
            api.get_pipeline_api_v1_pipelines_pipeline_id_get,
            pipeline_id=pipeline_id)

        c = pp.pipeline_config
        if GlobalKeys.BQ_ARGS_ in c:
            c.pop(GlobalKeys.BQ_ARGS_)
        if GlobalKeys.CUSTOM_CODE_ in c:
            c.pop(GlobalKeys.CUSTOM_CODE_)
        if 'ai_platform_training_args' in c:
            c.pop('ai_platform_training_args')

        return PipelineConfig(**c)

    def train_pipeline(self,
                       pipeline_id: Text,
                       datasource_id: Text = None,
                       datasource_commit_id: Text = None,
                       orchestration_backend: Text = None,
                       orchestration_args: Dict = None,
                       processing_backend: Text = None,
                       processing_args: Dict = None,
                       training_backend: Text = None,
                       training_args: Dict = None,
                       serving_backend: Text = None,
                       serving_args: Dict = None) -> PipelineRun:

        if datasource_id is None is datasource_commit_id is None:
            assert ValueError('Please either define a datasource_id '
                              '(to pick the latest commit) or a '
                              'datasource_commit_id to define a source.')

        ds_api = ce_api.DatasourcesApi(self.client)

        if datasource_id is not None:
            commits = api_utils.api_call(
                ds_api.get_commits_api_v1_datasources_ds_id_commits_get,
                datasource_id)

            commits.sort(key=lambda x: x.created_at)
            c_id = commits[-1].id

        elif datasource_commit_id is not None:
            c_id = datasource_commit_id
        else:
            raise LookupError('Hello there!')

        run_create = PipelineRun.creator(
            pipeline_run_type=PipelineRunTypes.training.name,
            datasource_commit_id=c_id,
            orchestration_backend=orchestration_backend,
            orchestration_args=orchestration_args,
            processing_backend=processing_backend,
            processing_args=processing_args,
            additional_args={'training_backend': training_backend,
                             'training_args': training_args,
                             'serving_backend': serving_backend,
                             'serving_args': serving_args})

        p_api = ce_api.PipelinesApi(self.client)
        return api_utils.api_call(
            p_api.create_pipeline_run_api_v1_pipelines_pipeline_id_runs_post,
            run_create,
            pipeline_id)

    def test_pipeline(self,
                      pipeline_id: Text,
                      datasource_id: Text = None,
                      datasource_commit_id: Text = None,
                      orchestration_backend: Text = None,
                      orchestration_args: Dict = None,
                      processing_backend: Text = None,
                      processing_args: Dict = None,
                      training_backend: Text = None,
                      training_args: Dict = None,
                      serving_backend: Text = None,
                      serving_args: Dict = None) -> PipelineRun:

        if datasource_id is None is datasource_commit_id is None:
            assert ValueError('Please either define a datasource_id '
                              '(to pick the latest commit) or a '
                              'datasource_commit_id to define a source.')

        ds_api = ce_api.DatasourcesApi(self.client)

        if datasource_id is not None:
            commits = api_utils.api_call(
                ds_api.get_commits_api_v1_datasources_ds_id_commits_get,
                datasource_id)

            commits.sort(key=lambda x: x.created_at)
            c_id = commits[-1].id

        elif datasource_commit_id is not None:
            c_id = datasource_commit_id
        else:
            raise LookupError('Hello there!')

        run_create = PipelineRun.creator(
            pipeline_run_type=PipelineRunTypes.test.name,
            datasource_commit_id=c_id,
            orchestration_backend=orchestration_backend,
            orchestration_args=orchestration_args,
            processing_backend=processing_backend,
            processing_args=processing_args,
            additional_args={'training_backend': training_backend,
                             'training_args': training_args,
                             'serving_backend': serving_backend,
                             'serving_args': serving_args})

        p_api = ce_api.PipelinesApi(self.client)
        return api_utils.api_call(
            p_api.create_pipeline_run_api_v1_pipelines_pipeline_id_runs_post,
            run_create,
            pipeline_id)

    def infer_pipeline(self,
                       pipeline_id: Text = None,
                       pipeline_run_id: Text = None,
                       datasource_id: Text = None,
                       datasource_commit_id: Text = None,
                       orchestration_backend: Text = None,
                       orchestration_args: Dict = None,
                       processing_backend: Text = None,
                       processing_args: Dict = None) -> PipelineRun:

        # Resolve the pipeline run_id
        if pipeline_id is None is pipeline_run_id is None:
            raise ValueError('Please either define a pipeline_id '
                             '(to pick the latest training run) or a '
                             'pipeline_run_id to choose a trained model.')

        p_api = ce_api.PipelinesApi(self.client)
        if pipeline_id is not None:
            runs = api_utils.api_call(
                p_api.get_pipeline_runs_api_v1_pipelines_pipeline_id_runs_get,
                pipeline_id)

            runs.sort(key=lambda x: x.run_time)
            training_runs = [r for r in runs if r.pipeline_run_type ==
                             PipelineRunTypes.training.name]
            if len(training_runs) == 0:
                raise ValueError('You dont have any training runs with the '
                                 'pipeline {}'.format(pipeline_id))
            r_id = training_runs[-1].id
        elif pipeline_run_id is not None:
            # TODO: If you just have the pipeline_run_id, how do you get the
            #   run without the pipeline_id?
            # TODO: We need to check whether we have a training run here
            r_id = pipeline_run_id
        else:
            raise LookupError('Hello there!')

        if datasource_id is None is datasource_commit_id is None:
            raise ValueError('Please either define a datasource_id '
                             '(to pick the latest commit) or a '
                             'datasource_commit_id to define a source.')

        ds_api = ce_api.DatasourcesApi(self.client)

        if datasource_id is not None:
            commits = api_utils.api_call(
                ds_api.get_commits_api_v1_datasources_ds_id_commits_get,
                datasource_id)

            commits.sort(key=lambda x: x.created_at)
            c_id = commits[-1].id

        elif datasource_commit_id is not None:
            c_id = datasource_commit_id
        else:
            raise LookupError('General Kenobi!')

        run_create = PipelineRun.creator(
            pipeline_run_type=PipelineRunTypes.infer.name,
            datasource_commit_id=c_id,
            orchestration_backend=orchestration_backend,
            orchestration_args=orchestration_args,
            processing_backend=processing_backend,
            processing_args=processing_args,
            additional_args={'run_id': r_id})

        p_api = ce_api.PipelinesApi(self.client)
        return api_utils.api_call(
            p_api.create_pipeline_run_api_v1_pipelines_pipeline_id_runs_post,
            run_create,
            pipeline_id)

    def get_pipeline_status(self,
                            workspace_id: Text,
                            pipeline_id: Text = None) -> Dict:

        ws_api = ce_api.WorkspacesApi(self.client)
        p_api = ce_api.PipelinesApi(self.client)
        d_api = ce_api.DatasourcesApi(self.client)

        status_dict = {}

        pipelines = api_utils.api_call(
            ws_api.get_workspaces_pipelines_api_v1_workspaces_workspace_id_pipelines_get,
            workspace_id)

        pipelines.sort(key=lambda x: x.created_at)
        for p in pipelines:
            write_check = (len(p.pipeline_runs) > 0) and \
                          (pipeline_id is None or pipeline_id == p.id)

            if write_check:

                status_dict[p.id] = []
                for r in p.pipeline_runs:
                    run = api_utils.api_call(
                        p_api.get_pipeline_run_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_get,
                        p.id,
                        r.id)

                    # Resolve datasource
                    ds_commit = api_utils.api_call(
                        d_api.get_single_commit_api_v1_datasources_commits_commit_id_get,
                        r.datasource_commit_id)
                    ds = api_utils.api_call(
                        d_api.get_datasource_api_v1_datasources_ds_id_get,
                        ds_commit.datasource_id)

                    if run.end_time:
                        td = run.end_time - run.start_time
                    else:
                        td = datetime.now(timezone.utc) - run.start_time

                    status_dict[p.id].append({
                        'RUN ID': run.id,
                        'TYPE': run.pipeline_run_type,
                        'STATUS': run.status,
                        'DATASOURCE': '{}_{}'.format(ds.name,
                                                     run.datasource_commit_id),
                        'DATAPOINTS': '{}'.format(ds_commit.n_datapoints),
                        'START TIME': print_utils.format_date(run.start_time),
                        'DURATION': print_utils.format_timedelta(td),
                    })

        return status_dict

    # PIPELINE RUN ############################################################

    def get_pipeline_runs(self,
                          pipeline_id: Text,
                          **kwargs) -> List[PipelineRun]:

        api = ce_api.PipelinesApi(self.client)
        pr_list = api_utils.api_call(
            api.get_pipeline_runs_api_v1_pipelines_pipeline_id_runs_get,
            pipeline_id)

        runs = [PipelineRun(**pr.to_dict()) for pr in pr_list]

        if kwargs:
            runs = client_utils.filter_objects(runs, **kwargs)
        return runs

    def get_pipeline_run_by_id(self,
                               pipeline_id: Text,
                               id: Text) -> PipelineRun:
        """ Get a pipeline with a specific id
        """
        return self.get_pipeline_runs(pipeline_id=pipeline_id,
                                      id=id)[0]

    def get_pipeline_run(self,
                         pipeline_id,
                         pipeline_run_id) -> PipelineRun:

        api = ce_api.PipelinesApi(self.client)
        pr = api_utils.api_call(
            api.get_pipeline_run_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_get,
            pipeline_id,
            pipeline_run_id)

        return PipelineRun(**pr.to_dict())

    def get_pipeline_run_logs(self,
                              pipeline_id,
                              pipeline_run_id) -> PipelineRun:

        api = ce_api.PipelinesApi(self.client)
        logs_url = api_utils.api_call(
            api.get_pipeline_logs_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_logs_get,
            pipeline_id, pipeline_run_id)

        return logs_url

    # FUNCTIONS ###############################################################

    def get_functions(self,
                      **kwargs):
        api = ce_api.FunctionsApi(self.client)
        f_list = api_utils.api_call(api.get_functions_api_v1_functions_get)
        functions = [Function(**f.to_dict()) for f in f_list]

        if kwargs:
            functions = client_utils.filter_objects(functions, **kwargs)
        return functions

    def get_function_by_name(self,
                             name: Text) -> Function:
        """ Get a function with a specific name
        """
        return self.get_functions(name=name)[0]

    def get_function_by_id(self,
                           id: Text) -> Function:
        """ Get a function with a specific id
        """
        return self.get_functions(id=id)[0]

    # FUNCTION VERSIONS #######################################################

    def get_function_versions(self,
                              function_id: Text,
                              **kwargs) -> List[FunctionVersion]:
        api = ce_api.FunctionsApi(self.client)
        fv_list = api_utils.api_call(
            api.get_function_versions_api_v1_functions_function_id_versions_get,
            function_id)
        versions = [FunctionVersion(**fv.to_dict()) for fv in fv_list]

        if kwargs:
            versions = client_utils.filter_objects(versions, **kwargs)
        return versions

    def get_function_version_by_id(self,
                                   function_id: Text,
                                   version_id: Text) -> FunctionVersion:
        return self.get_function_versions(function_id=function_id,
                                          version_id=version_id)[0]

    def push_function(self,
                      name: Text,
                      function_type: Text,
                      local_path: Text,
                      udf_name: Text,
                      message: Text = None):

        with open(local_path, 'rb') as file:
            data = file.read()
        encoded_file = base64.b64encode(data).decode()

        api = ce_api.FunctionsApi(self.client)
        api_utils.api_call(api.create_function_api_v1_functions_post,
                           Function.creator(name=name,
                                            function_type=function_type,
                                            udf_path=udf_name,
                                            message=message,
                                            file_contents=encoded_file))

    def pull_function_version(self, function_id, version_id, output_path=None):
        function_version = self.get_function_version(function_id, version_id)
        if output_path is None:
            output_path = os.path.join(os.getcwd(),
                                       '{}@{}.py'.format(function_id,
                                                         version_id))

        with open(output_path, 'wb') as f:
            f.write(base64.b64decode(function_version.file_contents))

        loader = machinery.SourceFileLoader(fullname='user_module',
                                            path=output_path)
        user_module = types.ModuleType(loader.name)
        loader.exec_module(user_module)
        return getattr(user_module, function_version.udf_path)

    def magic_function(self, function_id, version_id):
        import sys
        from IPython import get_ipython

        if 'ipykernel' not in sys.modules:
            raise EnvironmentError('The magic function is only usable in a '
                                   'Jupyter notebook.')

        function_version = self.get_function_version(function_id, version_id)
        f = base64.b64decode(function_version.file_contents).decode('utf-8')
        get_ipython().set_next_input(f)

    # POST-TRAINING ###########################################################

    def get_statistics(self,
                       pipeline_id: Text,
                       pipeline_run_id: Text,
                       magic: bool = False):

        api = ce_api.PipelinesApi(self.client)

        pipeline = api_utils.api_call(
            api.get_pipeline_api_v1_pipelines_pipeline_id_get,
            pipeline_id=pipeline_id)

        run = api_utils.api_call(
            api.get_pipeline_run_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_get,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id)

        stat_artifact = api_utils.api_call(
            api.get_pipeline_artifacts_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_artifacts_component_type_get,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id,
            component_type=GDPComponent.SplitStatistics.name)

        if run.pipeline_run_type != PipelineRunTypes.training.name:
            raise TypeError('The selected pipeline should be a training '
                            'pipeline')

        workspace_id = pipeline.workspace_id

        path = Path(click.get_app_dir(constants.APP_NAME),
                    'statistics',
                    workspace_id,
                    pipeline_id,
                    pipeline_run_id)

        download_artifact(artifact_json=stat_artifact[0].to_dict(),
                          path=path)

        import tensorflow as tf
        from tensorflow_metadata.proto.v0 import statistics_pb2
        import panel as pn

        result = {}
        for split in os.listdir(path):
            stats_path = os.path.join(path, split, 'stats_tfrecord')
            serialized_stats = next(
                tf.compat.v1.io.tf_record_iterator(stats_path))
            stats = statistics_pb2.DatasetFeatureStatisticsList()
            stats.ParseFromString(serialized_stats)
            dataset_list = statistics_pb2.DatasetFeatureStatisticsList()
            for i, d in enumerate(stats.datasets):
                d.name = split
                dataset_list.datasets.append(d)
            result[split] = dataset_list
        h = get_statistics_html(result)

        if magic:
            import sys
            if 'ipykernel' not in sys.modules:
                raise EnvironmentError('The magic functions are only usable '
                                       'in a Jupyter notebook.')
            from IPython.core.display import display, HTML
            display(HTML(h))

        else:
            pn.serve(panels=pn.pane.HTML(h, width=1200), show=True)

    def evaluate_single_pipeline(self,
                                 pipeline_id: Text,
                                 pipeline_run_id: Text,
                                 magic: bool = False):
        # Resolve the pipeline run_id
        if pipeline_id is None or pipeline_run_id is None:
            raise ValueError('Please either a pipeline_id and a '
                             'pipeline_run_id to choose a trained model.')

        p_api = ce_api.PipelinesApi(self.client)

        pipeline = api_utils.api_call(
            p_api.get_pipeline_api_v1_pipelines_pipeline_id_get,
            pipeline_id=pipeline_id)
        workspace_id = pipeline.workspace_id

        trainer_path = os.path.join(click.get_app_dir(constants.APP_NAME),
                                    'eval_trainer',
                                    workspace_id,
                                    pipeline_id,
                                    pipeline_run_id)

        eval_path = os.path.join(click.get_app_dir(constants.APP_NAME),
                                 'eval_evaluator',
                                 workspace_id,
                                 pipeline_id,
                                 pipeline_run_id)

        artifact = api_utils.api_call(
            p_api.get_pipeline_artifacts_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_artifacts_component_type_get,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id,
            component_type=GDPComponent.Trainer.name)
        download_artifact(artifact[0].to_dict(), path=trainer_path)

        artifact = api_utils.api_call(
            p_api.get_pipeline_artifacts_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_artifacts_component_type_get,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id,
            component_type=GDPComponent.Evaluator.name)
        download_artifact(artifact[0].to_dict(), path=eval_path)

        # Patch to make it work locally
        import json
        with open(os.path.join(eval_path, 'eval_config.json'), 'r') as f:
            eval_config = json.load(f)
        eval_config['modelLocations'][''] = eval_path
        with open(os.path.join(eval_path, 'eval_config.json'), 'w') as f:
            json.dump(eval_config, f)

        if magic:
            from cengine.utils.shell_utils import create_new_cell
            model_block = evaluation.get_model_block(trainer_path)
            eval_block = evaluation.get_eval_block(eval_path)

            create_new_cell(eval_block)
            create_new_cell(model_block)

        else:
            nb = nbf.v4.new_notebook()
            nb['cells'] = [
                nbf.v4.new_code_cell(evaluation.get_model_block(trainer_path)),
                nbf.v4.new_code_cell(evaluation.get_eval_block(eval_path))]

            config_folder = click.get_app_dir(constants.APP_NAME)

            if not (os.path.exists(config_folder) and os.path.isdir(
                    config_folder)):
                os.makedirs(config_folder)

            final_out_path = os.path.join(config_folder,
                                          constants.EVALUATION_NOTEBOOK)
            s = nbf.writes(nb)
            if isinstance(s, bytes):
                s = s.decode('utf8')

            with open(final_out_path, 'w') as f:
                f.write(s)
            os.system('jupyter notebook "{}"'.format(final_out_path))

    def compare_multiple_pipelines(self,
                                   workspace_id: Text):

        u_api = ce_api.UsersApi(self.client)
        user = api_utils.api_call(u_api.get_loggedin_user_api_v1_users_me_get)

        info = {constants.ACTIVE_USER: user.email,
                user.email: {
                    constants.TOKEN: self.client.configuration.access_token,
                    constants.ACTIVE_WORKSPACE: workspace_id}}

        # generate notebook
        nb = nbf.v4.new_notebook()
        nb['cells'] = [
            nbf.v4.new_code_cell(evaluation.import_block()),
            nbf.v4.new_code_cell(evaluation.info_block(info)),
            nbf.v4.new_code_cell(evaluation.application_block()),
            nbf.v4.new_code_cell(evaluation.interface_block()),
        ]

        # write notebook
        config_folder = click.get_app_dir(constants.APP_NAME)

        if not (os.path.exists(config_folder) and os.path.isdir(
                config_folder)):
            os.makedirs(config_folder)

        final_out_path = os.path.join(config_folder,
                                      constants.COMPARISON_NOTEBOOK)
        s = nbf.writes(nb)
        if isinstance(s, bytes):
            s = s.decode('utf8')

        with open(final_out_path, 'w') as f:
            f.write(s)

        # serve notebook
        os.system('panel serve "{}" --show'.format(final_out_path))

    def download_model(self,
                       pipeline_id,
                       pipeline_run_id,
                       output_path):
        if os.path.exists(output_path) and os.path.isdir(output_path):
            if not [f for f in os.listdir(output_path) if
                    not f.startswith('.')] == []:
                raise NotADirectoryError("Output path must be an empty "
                                         "directory!")
        if os.path.exists(output_path) and not os.path.isdir(output_path):
            raise NotADirectoryError("Output path must be an empty directory!")
        if not os.path.exists(output_path):
            logging.info("Creating directory {}..".format(output_path))

        # Resolve the pipeline run_id
        if pipeline_id is None or pipeline_run_id is None:
            raise ValueError('Please either a pipeline_id and a '
                             'pipeline_run_id to choose a trained model.')

        p_api = ce_api.PipelinesApi(self.client)

        artifact = api_utils.api_call(
            p_api.get_pipeline_artifacts_api_v1_pipelines_pipeline_id_runs_pipeline_run_id_artifacts_component_type_get,
            pipeline_id=pipeline_id,
            pipeline_run_id=pipeline_run_id,
            component_type=GDPComponent.Deployer.name)

        spin = Spinner()
        spin.start()
        if len(artifact) == 1:
            download_artifact(artifact_json=artifact[0].to_dict(),
                              path=output_path)
            spin.stop()
        else:
            raise Exception('Something unexpected happened! Please contact '
                            'core@maiot.io to get further information.')

        logging.info('Model downloaded to: {}'.format(output_path))


class CLIClient(Client):

    def __init__(self, info):
        active_user = info[constants.ACTIVE_USER]
        config = ce_api.Configuration()
        config.host = constants.API_HOST
        config.access_token = info[active_user][constants.TOKEN]

        self.client = ce_api.ApiClient(config)
