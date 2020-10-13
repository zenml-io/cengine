from typing import Dict, Any, Text

from ce_api.models import backend
from ce_api.models import datasource, datasource_create
from ce_api.models import datasource_commit, datasource_commit_create
from ce_api.models import function, function_create
from ce_api.models import function_version, function_version_create
from ce_api.models import pipeline, pipeline_create
from ce_api.models import pipeline_run, pipeline_run_create
from ce_api.models import provider, provider_create
from ce_api.models import user
from ce_api.models import workspace, workspace_create
from cengine.utils.print_utils import to_pretty_string, PrintStyles


class User(user.User):
    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class Provider(provider.Provider):
    @classmethod
    def creator(cls,
                name: Text,
                type_: Text,
                args: Dict[Text, Any]):
        return provider_create.ProviderCreate(
            name=name,
            type=type_,
            args=args)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class Backend(backend.Backend):
    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class Datasource(datasource.Datasource):
    @classmethod
    def creator(cls,
                name: Text,
                source: Text,
                type_: Text,
                args: Dict[Text, Any]):
        return datasource_create.DatasourceCreate(
            name=name,
            source=source,
            type=type_,
            args=args)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class DatasourceCommit(datasource_commit.DatasourceCommit):
    @classmethod
    def creator(cls,
                message,
                schema,
                orchestration_backend,
                orchestration_args,
                processing_backend,
                processing_args):
        return datasource_commit_create.DatasourceCommitCreate(
            message=message,
            used_schema=schema,
            orchestration_backend=orchestration_backend,
            orchestration_args=orchestration_args,
            processing_backend=processing_backend,
            processing_args=processing_args)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class Function(function.Function):
    @classmethod
    def creator(cls,
                name: Text,
                function_type: Text,
                udf_path: Text,
                message: Text,
                file_contents):
        return function_create.FunctionCreate(
            name=name,
            function_type=function_type,
            udf_path=udf_path,
            message=message,
            file_contents=file_contents)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class FunctionVersion(function_version.FunctionVersion):
    @classmethod
    def creator(cls,
                udf_path: Text,
                message: Text,
                file_contents):
        return function_version_create.FunctionVersionCreate(
            udf_path=udf_path,
            message=message,
            file_contents=file_contents)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class Pipeline(pipeline.Pipeline):
    @classmethod
    def creator(cls,
                name: Text,
                pipeline_config: Dict[Text, Any],
                workspace_id: Text,
                pipeline_type: Text = 'normal'):
        return pipeline_create.PipelineCreate(
            name=name,
            pipeline_config=pipeline_config,
            workspace_id=workspace_id,
            pipeline_type=pipeline_type)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)

    def train(self,
              client,
              datasource_id: Text = None,
              datasource_commit_id: Text = None,
              orchestration_backend: Text = None,
              orchestration_args: Dict = None,
              processing_backend: Text = None,
              processing_args: Dict = None,
              training_backend: Text = None,
              training_args: Dict = None,
              serving_backend: Text = None,
              serving_args: Dict = None):
        client.train_pipeline(pipeline_id=self.id,
                              datasource_id=datasource_id,
                              datasource_commit_id=datasource_commit_id,
                              orchestration_backend=orchestration_backend,
                              orchestration_args=orchestration_args,
                              processing_backend=processing_backend,
                              processing_args=processing_args,
                              training_backend=training_backend,
                              training_args=training_args,
                              serving_backend=serving_backend,
                              serving_args=serving_args)

    def test(self,
             client,
             datasource_id: Text = None,
             datasource_commit_id: Text = None,
             orchestration_backend: Text = None,
             orchestration_args: Dict = None,
             processing_backend: Text = None,
             processing_args: Dict = None,
             training_backend: Text = None,
             training_args: Dict = None,
             serving_backend: Text = None,
             serving_args: Dict = None):
        client.test_pipeline(pipeline_id=self.id,
                             datasource_id=datasource_id,
                             datasource_commit_id=datasource_commit_id,
                             orchestration_backend=orchestration_backend,
                             orchestration_args=orchestration_args,
                             processing_backend=processing_backend,
                             processing_args=processing_args,
                             training_backend=training_backend,
                             training_args=training_args,
                             serving_backend=serving_backend,
                             serving_args=serving_args)

    def infer(self,
              client,
              pipeline_run_id: Text = None,
              datasource_id: Text = None,
              datasource_commit_id: Text = None,
              orchestration_backend: Text = None,
              orchestration_args: Dict = None,
              processing_backend: Text = None,
              processing_args: Dict = None):
        client.infer_pipeline(pipeline_id=self.id,
                              pipeline_run_id=pipeline_run_id,
                              datasource_id=datasource_id,
                              datasource_commit_id=datasource_commit_id,
                              orchestration_backend=orchestration_backend,
                              orchestration_args=orchestration_args,
                              processing_backend=processing_backend,
                              processing_args=processing_args)


class PipelineRun(pipeline_run.PipelineRun):
    @classmethod
    def creator(cls,
                pipeline_run_type,
                datasource_commit_id,
                orchestration_backend,
                orchestration_args,
                processing_backend,
                processing_args,
                additional_args):
        return pipeline_run_create.PipelineRunCreate(
            pipeline_run_type=pipeline_run_type,
            datasource_commit_id=datasource_commit_id,
            orchestration_backend=orchestration_backend,
            orchestration_args=orchestration_args,
            processing_backend=processing_backend,
            processing_args=processing_args,
            additional_args=additional_args)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)


class Workspace(workspace.Workspace):
    @classmethod
    def creator(cls,
                name=Text,
                provider_id=Text):
        return workspace_create.WorkspaceCreate(
            name=name,
            provider_id=provider_id)

    def __str__(self):
        return to_pretty_string(self.to_dict())

    def __repr__(self):
        return to_pretty_string(self.to_dict(), style=PrintStyles.PPRINT)
