import base64
import inspect
import logging

import ce_api
from ce_standards.enums import FunctionTypes
from ce_standards.standard_experiment import GlobalKeys, SplitKeys, MethodKeys, \
    PreProcessKeys, TrainerKeys, DefaultKeys, TimeSeriesKeys, EvaluatorKeys
from cengine import models
from cengine.utils import api_utils
from cengine.utils.config_utils import serialize
from cengine.utils.lint_utils import validate_split_config
from cengine.utils.print_utils import to_pretty_string, PrintStyles


class PipelineConfig:
    @classmethod
    def from_datasource(cls,
                        client,
                        datasource_id: str,
                        commit_id: str = None):

        ds_api = ce_api.DatasourcesApi(client.client)

        if commit_id is None:
            commits = api_utils.api_call(
                ds_api.get_commits_api_v1_datasources_ds_id_commits_get,
                datasource_id)
            commits.sort(key=lambda x: x.created_at)
            commit_id = commits[0].id

        schema = api_utils.api_call(
            ds_api.get_datasource_commit_schema_api_v1_datasources_ds_id_commits_commit_id_schema_get,
            ds_id=datasource_id,
            commit_id=commit_id)

        config = cls()
        config.features = [f for f in schema]
        return config

    def __init__(self,
                 split=None,
                 features=None,
                 labels=None,
                 evaluator=None,
                 trainer=None,
                 preprocessing=None,
                 timeseries=None,
                 version=None,
                 ):
        self.__version = 1
        self.__split = Split()
        self.__features = Features()
        self.__labels = Features()
        self.__evaluator = Evaluator()
        self.__trainer = Trainer()
        self.__preprocessing = Preprocessing()
        self.__timeseries = Timeseries()

        if split is not None:
            self.split = split
        if features is not None:
            self.features = features
        if labels is not None:
            self.labels = labels
        if evaluator is not None:
            self.evaluator = evaluator
        if trainer is not None:
            self.trainer = trainer
        if preprocessing is not None:
            self.preprocessing = preprocessing
        if timeseries is not None:
            self.timeseries = timeseries
        if version is not None and version != 1:
            logging.warning('Ignoring the given version {} and proceeding '
                            'with configuration version 1.'.format(version))

    @staticmethod
    def get_attributes():
        return {
            GlobalKeys.VERSION,
            GlobalKeys.SPLIT,
            GlobalKeys.FEATURES,
            GlobalKeys.LABELS,
            GlobalKeys.EVALUATOR,
            GlobalKeys.TRAINER,
            GlobalKeys.PREPROCESSING,
            GlobalKeys.TIMESERIES_,
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        return serialize(self)

    def check_completion(self):
        for attr in self.get_attributes():
            value = getattr(self, attr)
            if hasattr(value, 'check_completion'):
                value.check_completion()

    @property
    def version(self):
        return self.__version

    @property
    def split(self):
        return self.__split

    @property
    def features(self):
        return self.__features

    @property
    def labels(self):
        return self.__labels

    @property
    def evaluator(self):
        return self.__evaluator

    @property
    def trainer(self):
        return self.__trainer

    @property
    def preprocessing(self):
        return self.__preprocessing

    @property
    def timeseries(self):
        return self.__timeseries

    @version.setter
    def version(self, version):
        raise Exception("Please don't! :)")

    @split.setter
    def split(self, split):
        if isinstance(split, dict):
            self.__split = Split(**split)
        elif isinstance(split, Split):
            self.__split = split
        else:
            raise TypeError('Please provide either a cengine.Split object '
                            'or a dict instead of a {} to configure the '
                            'splitting process.'.format(type(split)))

    @features.setter
    def features(self, features):
        if isinstance(features, list):
            self.__features = Features(*features)
        elif isinstance(features, dict):
            self.__features = Features(**features)
        elif isinstance(features, Features):
            self.__features = features
        else:
            raise TypeError(
                'Please provide either a cengine.Features object '
                'or a dict instead of a {} to configure the feature '
                'selection.'.format(type(features)))

    @labels.setter
    def labels(self, labels):
        self.__labels = Features()

        if isinstance(labels, list):
            self.__labels = Features(*labels)
        elif isinstance(labels, dict):
            self.__labels = Features(**labels)
        elif isinstance(labels, Features):
            self.__labels = labels
        else:
            raise TypeError(
                'Please provide either a cengine.Features object, a list '
                'or a dict instead of a {} to configure the label '
                'selection.'.format(type(labels)))

    @evaluator.setter
    def evaluator(self, evaluator):
        if isinstance(evaluator, list):
            self.__evaluator = Evaluator(*evaluator)
        elif isinstance(evaluator, dict):
            self.__evaluator = Evaluator(**evaluator)
        elif isinstance(evaluator, Evaluator):
            self.__evaluator = evaluator
        else:
            raise TypeError(
                'Please provide either a cengine.Evaluator object '
                'or a dict instead of a {} to configure the '
                'evaluation process.'.format(type(evaluator)))

    @trainer.setter
    def trainer(self, trainer):
        if isinstance(trainer, dict):
            self.__trainer = Trainer(**trainer)
        elif isinstance(trainer, Trainer):
            self.__trainer = trainer
        else:
            raise TypeError('Please provide either a cengine.Split object '
                            'or a dict instead of a {} to configure the '
                            'splitting process.'.format(type(trainer)))

    @preprocessing.setter
    def preprocessing(self, preprocessing):
        if isinstance(preprocessing, dict):
            self.__preprocessing = Preprocessing(**preprocessing)
        elif isinstance(preprocessing, Preprocessing):
            self.__preprocessing = preprocessing
        else:
            raise TypeError('Please provide either a cengine.Defaults object '
                            'or a dict instead of a {} to configure the '
                            'default preprocessing '
                            'steps.'.format(type(preprocessing)))

    @timeseries.setter
    def timeseries(self, timeseries):
        if isinstance(timeseries, dict):
            self.__timeseries = Timeseries(**timeseries)
        elif isinstance(timeseries, Timeseries):
            self.__timeseries = timeseries
        else:
            raise TypeError(
                'Please provide either a cengine.Timeseries object '
                'or a dict instead of a {} to configure the '
                'timeseries related '
                'parameters.'.format(type(timeseries)))

    @version.deleter
    def version(self):
        raise Exception("Please don't! :)")

    @split.deleter
    def split(self):
        self.__split = Split()

    @features.deleter
    def features(self):
        self.__features = Features()

    @labels.deleter
    def labels(self):
        self.__labels = Features()

    @evaluator.deleter
    def evaluator(self):
        self.__evaluator = Evaluator()

    @trainer.deleter
    def trainer(self):
        self.__trainer = Trainer()

    @preprocessing.deleter
    def preprocessing(self):
        self.__preprocessing = Preprocessing()

    @timeseries.deleter
    def timeseries(self):
        self.__timeseries = Timeseries()


class Split:

    def __init__(self,
                 categorize=None,
                 index=None,
                 ratio=None,
                 where=None):
        self.__categorize = Categorize()
        self.__index = Index()
        self.__ratio = Ratio()
        self.__where = None

        if categorize is not None:
            self.categorize = categorize
        if index is not None:
            self.index = index
        if ratio is not None:
            self.ratio = ratio
        if where is not None:
            self.where = where

    @staticmethod
    def get_attributes():
        return {
            SplitKeys.CATEGORIZE_BY_,
            SplitKeys.INDEX_BY_,
            SplitKeys.RATIO_,
            SplitKeys.WHERE_,
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        return serialize(self)

    def check_completion(self):
        validate_split_config(self.to_serial())

    @property
    def categorize(self):
        return self.__categorize

    @property
    def index(self):
        return self.__index

    @property
    def ratio(self):
        return self.__ratio

    @property
    def where(self):
        return self.__where

    @categorize.setter
    def categorize(self, categorize):
        if isinstance(categorize, dict):
            self.__categorize = Categorize(**categorize)
        elif isinstance(categorize, Categorize):
            self.__categorize = categorize
        else:
            raise TypeError('Please provide either a cengine.Categorize '
                            'object or a dict instead of a {} to configure '
                            'the split categories.'.format(type(categorize)))

    @index.setter
    def index(self, index):
        if isinstance(index, dict):
            self.__index = Index(**index)
        elif isinstance(index, Index):
            self.__index = index
        else:
            raise TypeError('Please provide either a cengine.Index '
                            'object or a dict instead of a {} to configure '
                            'the index during splitting.'.format(type(index)))

    @ratio.setter
    def ratio(self, ratio):
        if isinstance(ratio, dict):
            self.__ratio = Ratio(**ratio)
        else:
            raise TypeError('Please provide a dict instead of a {} for the '
                            'ratio.'.format(type(ratio)))

    @where.setter
    def where(self, where):
        if isinstance(where, list):
            for w in where:
                if not isinstance(w, str):
                    raise TypeError('Please provide string values instead of '
                                    'a {} for the where '
                                    'clauses.'.format(type(w)))
            self.__where = where
        else:
            raise TypeError('Please provide a list instead of a {} for the '
                            'where clauses.'.format(type(where)))

    @categorize.deleter
    def categorize(self):
        self.__categorize = Categorize()

    @index.deleter
    def index(self):
        self.__index = Index()

    @ratio.deleter
    def ratio(self):
        self.__ratio = Ratio()

    @where.deleter
    def where(self):
        self.__where = None


class Categorize:

    def __init__(self,
                 by=None,
                 ratio=None,
                 categories=None):

        self.__by = None
        self.__ratio = Ratio()
        self.__categories = None

        if by is not None:
            self.by = by
        if ratio is not None:
            self.ratio = ratio
        if categories is not None:
            self.categories = categories

    @staticmethod
    def get_attributes():
        return {
            'by',
            'ratio',
            'categories'
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        if self.__by is not None:
            return serialize(self)

    @property
    def by(self):
        return self.__by

    @property
    def ratio(self):
        return self.__ratio

    @property
    def categories(self):
        return self.__categories

    @by.setter
    def by(self, by):
        if isinstance(by, str):
            self.__by = by
        else:
            raise TypeError('Please provide a string value instead of '
                            'a {} for the -by-.'.format(type(by)))

    @ratio.setter
    def ratio(self, ratio):
        if isinstance(ratio, dict):
            self.__ratio = Ratio(**ratio)
        else:
            raise TypeError('Please provide a dict instead of a {} for the '
                            'ratio.'.format(type(ratio)))

    @categories.setter
    def categories(self, categories):
        if isinstance(categories, list):
            self.__categories = categories
        elif isinstance(categories, dict):
            self.__categories = categories
        else:
            raise TypeError('Please provide a list or a dict instead of '
                            'a {} for the -by-.'.format(type(categories)))

    @by.deleter
    def by(self):
        self.__by = None

    @ratio.deleter
    def ratio(self):
        self.__ratio = Ratio()

    @categories.deleter
    def categories(self):
        self.__categories = None


class Index:

    def __init__(self,
                 by=None,
                 ratio=None):
        self.__by = None
        self.__ratio = Ratio()

        if by is not None:
            self.by = by
        if ratio is not None:
            self.ratio = ratio

    @staticmethod
    def get_attributes():
        return {
            'by',
            'ratio'
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        if self.__by is not None:
            return serialize(self)

    @property
    def by(self):
        return self.__by

    @property
    def ratio(self):
        return self.__ratio

    @by.setter
    def by(self, by):
        if isinstance(by, str):
            self.__by = by
        else:
            raise TypeError('Please provide a string value instead of '
                            'a {} for the -by-.'.format(type(by)))

    @ratio.setter
    def ratio(self, ratio):
        if isinstance(ratio, dict):
            self.__ratio = Ratio(**ratio)
        else:
            raise TypeError('Please provide a dict instead of a {} for the '
                            'ratio.'.format(type(ratio)))

    @by.deleter
    def by(self):
        self.__by = None

    @ratio.deleter
    def ratio(self):
        self.__ratio = Ratio()


class Features:
    def __init__(self, *args, **kwargs):

        self.__features = dict()

        for a in args:
            self.__features[a] = MethodLists()

        for k, a in kwargs.items():
            if isinstance(a, dict):
                self.__features[k] = MethodLists(**a)
            elif isinstance(a, MethodLists):
                self.__features[k] = a
            else:
                raise TypeError(
                    'Please provide either a cengine.MethodLists object '
                    'or a dict instead of a {} to configure the '
                    'preprocessing process.'.format(type(a)))

    def __delattr__(self, item):
        self.__features.pop(item)

    def __getitem__(self, item):
        return self.__features[item]

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        return serialize(self.__features)

    def check_completion(self):
        for key, value in self.__features.items():
            value.check_completion(full_check=False)

    def add(self, features):
        if isinstance(features, list):
            self.__features.update(Features(*features).__features)
        elif isinstance(features, dict):
            self.__features.update(Features(**features).__features)
        elif isinstance(features, Features):
            self.__features.update(features)
        else:
            raise TypeError(
                'Please provide either a cengine.Features object, a list '
                'or a dict instead of a {} to configure the feature '
                'selection.'.format(type(features)))


class Ratio:
    def __init__(self, **kwargs):

        self.__ratio = dict()

        for k, a in kwargs.items():
            if not isinstance(a, (int, float)) or (a < 0 or a > 1):
                raise ValueError('Please define a numerical ratio between'
                                 '0 and 1 for each split.')

            else:
                self.__ratio[k] = a

    def __delattr__(self, item):
        self.__ratio.pop(item)

    def __getitem__(self, item):
        return self.__ratio[item]

    def add_splits(self, splits):
        if isinstance(splits, dict):
            self.__ratio.update(Ratio(**splits).__ratio)
        else:
            raise TypeError('Please provide a dict instead of a {} for the '
                            'ratio.'.format(type(splits)))

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        if self.__ratio:
            return serialize(self.__ratio)


class StepList:

    def __init__(self, *args):
        self.__step_list = list()

        for a in args:
            if isinstance(a, dict):
                self.__step_list.append(Method(**a))
            elif isinstance(a, Method):
                self.__step_list.append(a)
            else:
                raise ValueError('Please use either a dict or an instance '
                                 'of cengine.Method to define a preprocessing '
                                 'method')

    def add_methods(self, methods):

        if not isinstance(methods, list):
            methods = [methods]

        for m in methods:
            if isinstance(m, dict):
                self.__step_list.append(Method(**m))
            elif isinstance(m, Method):
                self.__step_list.append(m)
            else:
                raise TypeError(
                    'Please provide either a cengine.Method object, a list '
                    'or a dict instead of a {} to configure the method '
                    'selection.'.format(type(methods)))

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def __getitem__(self, item):
        return self.__step_list[item]

    def to_serial(self):
        if self.__step_list:
            return serialize(self.__step_list)

    def check_completion(self):
        for element in self.__step_list:
            element.check_completion()


class MethodLists:

    def __init__(self,
                 filling=None,
                 transform=None,
                 label_tuning=None,
                 resampling=None):

        self.__filling = StepList()
        self.__transform = StepList()
        self.__label_tuning = StepList()
        self.__resampling = StepList()

        if filling is not None:
            self.filling = filling

        if transform is not None:
            self.transform = transform

        if label_tuning is not None:
            self.label_tuning = label_tuning

        if resampling is not None:
            self.resampling = resampling

    @staticmethod
    def get_attributes():
        return {
            PreProcessKeys.FILLING,
            PreProcessKeys.TRANSFORM,
            PreProcessKeys.LABEL_TUNING,
            PreProcessKeys.RESAMPLING,
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        return serialize(self)

    def check_completion(self, full_check=False):
        for p in [self.filling, self.resampling, self.label_tuning]:

            if full_check:
                if len(p.to_serial()) != 1:
                    raise ValueError('While defining default preprocessing '
                                     'steps, please make sure that exactly '
                                     'one method for is defined for '
                                     'resampling, filling and label_tuning.')
            else:
                if len(p.to_serial()) < 0:
                    raise ValueError('While defining preprocessing steps for, '
                                     'specific features please make sure '
                                     'that you define either 1 or 0 method for '
                                     'resampling, filling and label_tuning.')

            for m in p.to_serial():
                m.check_completion()

        if full_check:
            if len(self.transform.to_serial()) < 1:
                raise ValueError('While defining default preprocessing '
                                 'steps, please make sure that at least '
                                 'one method for is defined for '
                                 'transform')
        for m in self.transform.to_serial():
            m.check_completion()

    @property
    def filling(self):
        return self.__filling

    @property
    def transform(self):
        return self.__transform

    @property
    def label_tuning(self):
        return self.__label_tuning

    @property
    def resampling(self):
        return self.__resampling

    @filling.setter
    def filling(self, filling):
        self.__filling = StepList()
        self.__filling.add_methods(filling)

    @transform.setter
    def transform(self, transform):
        self.__transform = StepList()
        self.__transform.add_methods(transform)

    @label_tuning.setter
    def label_tuning(self, label_tuning):
        self.__label_tuning = StepList()
        self.__label_tuning.add_methods(label_tuning)

    @resampling.setter
    def resampling(self, resampling):
        self.__resampling = StepList()
        self.__resampling.add_methods(resampling)

    @filling.deleter
    def filling(self):
        self.__filling = StepList()

    @transform.deleter
    def transform(self):
        self.__transform = StepList()

    @label_tuning.deleter
    def label_tuning(self):
        self.__label_tuning = StepList()

    @resampling.deleter
    def resampling(self):
        self.__resampling = StepList()


class Method:

    def __init__(self,
                 method,
                 parameters=None):
        self.__method = None
        self.__parameters = dict()

        if method is not None:
            self.method = method
        if parameters is not None:
            self.parameters = parameters

    @classmethod
    def from_callable(cls, client, fn, params):

        path = inspect.getfile(fn)
        name = fn.__name__
        message = 'Automatic message used by the Python SDK'

        with open(path, 'rb') as file:
            data = file.read()
        encoded_file = base64.b64encode(data).decode()

        fn_api = ce_api.FunctionsApi(client.client)
        fn_list = api_utils.api_call(
            func=fn_api.get_functions_api_v1_functions_get)

        matching_fn_list = [fn for fn in fn_list if fn.name == name]
        if len(matching_fn_list) == 0:
            logging.info('No matching functions found! Pushing a new '
                         'function!')

            func = api_utils.api_call(
                func=fn_api.create_function_api_v1_functions_post,
                body=models.Function.creator(
                    name=name,
                    function_type=FunctionTypes.transform.name,
                    udf_path=name,
                    message=message,
                    file_contents=encoded_file))

            version = func.function_versions[0]

        elif len(matching_fn_list) == 1:
            logging.info('Matching functions found! Pushing a new '
                         'function version!')

            func = matching_fn_list[0]

            version = api_utils.api_call(
                func=fn_api.create_function_version_api_v1_functions_function_id_versions_post,
                function_id=func.id,
                body=models.FunctionVersion.creator(
                    udf_path=name,
                    message=message,
                    file_contents=encoded_file))
        else:
            raise ValueError('Too many functions with a matching name')

        fn = '@'.join([func.id, version.id])
        params = params
        return cls(method=fn, parameters=params)

    @staticmethod
    def get_attributes():
        return {
            MethodKeys.METHOD,
            MethodKeys.PARAMETERS
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        if self.__method is not None:
            return serialize(self)

    def check_completion(self):
        if self.method is None:
            raise ValueError('Please provide a proper value "method" for the '
                             'method name.')

    @property
    def method(self):
        return self.__method

    @property
    def parameters(self):
        return self.__parameters

    @method.setter
    def method(self, method):
        if isinstance(method, str):
            self.__method = method
        else:
            raise TypeError('Please provide a string value instead of '
                            'a {} for method name.'.format(type(method)))

    @parameters.setter
    def parameters(self, parameters):
        if isinstance(parameters, dict):
            self.__parameters = parameters
        else:
            raise TypeError('Please provide a dict instead of '
                            'a {} for method '
                            'parameters.'.format(type(parameters)))

    @method.deleter
    def method(self):
        self.__method = None

    @parameters.deleter
    def parameters(self):
        self.__parameters = dict()


class Trainer:

    def __init__(self,
                 fn=None,
                 params=None):
        self.__fn = None
        self.__params = dict()

        if fn is not None:
            self.fn = fn
        if params is not None:
            self.params = params

    @staticmethod
    def get_attributes():
        return {
            TrainerKeys.FN,
            TrainerKeys.PARAMS
        }

    @classmethod
    def from_callable(cls, client, fn, params):

        path = inspect.getfile(fn)
        name = fn.__name__
        message = 'Automatic message used by the Python SDK'

        with open(path, 'rb') as file:
            data = file.read()
        encoded_file = base64.b64encode(data).decode()

        fn_api = ce_api.FunctionsApi(client.client)
        fn_list = api_utils.api_call(
            func=fn_api.get_functions_api_v1_functions_get)

        matching_fn_list = [fn for fn in fn_list if fn.name == name]
        if len(matching_fn_list) == 0:
            logging.info('No matching functions found! Pushing a new '
                         'function!')

            func = api_utils.api_call(
                func=fn_api.create_function_api_v1_functions_post,
                body=models.Function.creator(
                    name=name,
                    function_type=FunctionTypes.model.name,
                    udf_path=name,
                    message=message,
                    file_contents=encoded_file))

            version = func.function_versions[0]

        elif len(matching_fn_list) == 1:
            logging.info('Matching functions found! Pushing a new '
                         'function version!')

            func = matching_fn_list[0]

            version = api_utils.api_call(
                func=fn_api.create_function_version_api_v1_functions_function_id_versions_post,
                function_id=func.id,
                body=models.FunctionVersion.creator(
                    udf_path=name,
                    message=message,
                    file_contents=encoded_file))
        else:
            raise ValueError('Too many functions with a matching name')

        fn = '@'.join([func.id, version.id])
        params = params
        return cls(fn=fn, params=params)

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        if self.__fn is not None:
            return serialize(self)

    def check_completion(self):
        if self.fn is None:
            raise ValueError('Please provide a value for "fn" for the '
                             'trainer')

    @property
    def fn(self):
        return self.__fn

    @property
    def params(self):
        return self.__params

    @fn.setter
    def fn(self, fn):
        if isinstance(fn, str):
            self.__fn = fn
        else:
            raise TypeError('Please provide a string value instead of '
                            'a {} for trainer name.'.format(type(fn)))

    @params.setter
    def params(self, params):
        if isinstance(params, dict):
            self.__params = params
        else:
            raise TypeError('Please provide a dict instead of '
                            'a {} for trainer '
                            'parameters.'.format(type(params)))

    @fn.deleter
    def fn(self):
        self.__fn = None

    @params.deleter
    def params(self):
        self.__params = dict()


class Preprocessing:

    def __init__(self,
                 string=None,
                 integer=None,
                 float=None,
                 boolean=None):

        self.__string = MethodLists(
            filling=[Method(method='custom', parameters={'custom_value': ''})],
            resampling=[Method(method='mode', parameters={})],
            transform=[
                Method(method='compute_and_apply_vocabulary', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )
        self.__integer = MethodLists(
            filling=[Method(method='max', parameters={})],
            resampling=[Method(method='mean', parameters={})],
            transform=[Method(method='scale_to_z_score', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )
        self.__float = MethodLists(
            filling=[Method(method='max', parameters={})],
            resampling=[Method(method='mean', parameters={})],
            transform=[Method(method='scale_to_z_score', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )
        self.__boolean = MethodLists(
            filling=[Method(method='max', parameters={})],
            resampling=[Method(method='mode', parameters={})],
            transform=[Method(method='no_transform', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )

        if string is not None:
            self.string = string
        if integer is not None:
            self.integer = integer
        if float is not None:
            self.float = float
        if boolean is not None:
            self.boolean = boolean

    @staticmethod
    def get_attributes():
        return {
            DefaultKeys.STRING,
            DefaultKeys.BOOLEAN,
            DefaultKeys.FLOAT,
            DefaultKeys.INTEGER
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        return serialize(self)

    def check_completion(self):
        for key in self.get_attributes():
            getattr(self, key).check_completion(full_check=True)

    @property
    def string(self):
        return self.__string

    @property
    def integer(self):
        return self.__integer

    @property
    def float(self):
        return self.__float

    @property
    def boolean(self):
        return self.__boolean

    @string.setter
    def string(self, string_):
        if isinstance(string_, dict):
            self.__string = MethodLists(**string_)
        elif isinstance(string_, MethodLists):
            self.__string = string_
        else:
            raise TypeError('Please provide either a cengine.Preprocessing '
                            'object or a dict instead of a {} to configure '
                            'the default preprocessing of string '
                            'values.'.format(type(string_)))

    @integer.setter
    def integer(self, integer_):
        if isinstance(integer_, dict):
            self.__integer = MethodLists(**integer_)
        elif isinstance(integer_, MethodLists):
            self.__integer = integer_
        else:
            raise TypeError('Please provide either a cengine.Preprocessing '
                            'object or a dict instead of a {} to configure '
                            'the default preprocessing of integer '
                            'values.'.format(type(integer_)))

    @float.setter
    def float(self, float_):
        if isinstance(float_, dict):
            self.__float = MethodLists(**float_)
        elif isinstance(float_, MethodLists):
            self.__float = float_
        else:
            raise TypeError('Please provide either a cengine.Preprocessing '
                            'object or a dict instead of a {} to configure '
                            'the default preprocessing of float '
                            'values.'.format(type(float_)))

    @boolean.setter
    def boolean(self, boolean_):
        if isinstance(boolean_, dict):
            self.__boolean = MethodLists(**boolean_)
        elif isinstance(boolean_, MethodLists):
            self.__boolean = boolean_
        else:
            raise TypeError('Please provide either a cengine.Preprocessing '
                            'object or a dict instead of a {} to configure '
                            'the default preprocessing of boolean '
                            'values.'.format(type(boolean_)))

    @string.deleter
    def string(self):
        self.__string = MethodLists(
            filling=[Method(method='custom', parameters={'custom_value': ''})],
            resampling=[Method(method='mode', parameters={})],
            transform=[
                Method(method='compute_and_apply_vocabulary', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )

    @integer.deleter
    def integer(self):
        self.__integer = MethodLists(
            filling=[Method(method='max', parameters={})],
            resampling=[Method(method='mean', parameters={})],
            transform=[Method(method='scale_to_z_score', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )

    @float.deleter
    def float(self):
        self.__float = MethodLists(
            filling=[Method(method='max', parameters={})],
            resampling=[Method(method='mean', parameters={})],
            transform=[Method(method='scale_to_z_score', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )

    @boolean.deleter
    def boolean(self):
        self.__boolean = MethodLists(
            filling=[Method(method='max', parameters={})],
            resampling=[Method(method='max', parameters={})],
            transform=[Method(method='no_transform', parameters={})],
            label_tuning=[Method(method='no_tuning', parameters={})],
        )


class Timeseries:

    def __init__(self,
                 resampling_rate_in_secs=None,
                 trip_gap_threshold_in_secs=None,
                 process_sequence_w_timestamp=None,
                 process_sequence_w_category=None,
                 sequence_shift=None,
                 sequence_length=None):

        self.__resampling_rate_in_secs = None
        self.__trip_gap_threshold_in_secs = None
        self.__process_sequence_w_timestamp = None
        self.__process_sequence_w_category = None
        self.__sequence_shift = None
        self.__sequence_length = None

        if resampling_rate_in_secs is not None:
            self.resampling_rate_in_secs = resampling_rate_in_secs
        if trip_gap_threshold_in_secs is not None:
            self.trip_gap_threshold_in_secs = trip_gap_threshold_in_secs
        if process_sequence_w_timestamp is not None:
            self.process_sequence_w_timestamp = process_sequence_w_timestamp
        if process_sequence_w_category is not None:
            self.process_sequence_w_category = process_sequence_w_category
        if sequence_shift is not None:
            self.sequence_shift = sequence_shift
        if sequence_length is not None:
            self.sequence_length = sequence_length

    @staticmethod
    def get_attributes():
        return {
            TimeSeriesKeys.RESAMPLING_RATE_IN_SECS,
            TimeSeriesKeys.TRIP_GAP_THRESHOLD_IN_SECS,
            TimeSeriesKeys.PROCESS_SEQUENCE_W_TIMESTAMP,
            TimeSeriesKeys.PROCESS_SEQUENCE_W_CATEGORY_,
            TimeSeriesKeys.SEQUENCE_SHIFT,
            TimeSeriesKeys.SEQUENCE_LENGTH
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        if self.process_sequence_w_timestamp is not None:
            return serialize(self)

    def check_completion(self):
        if self.process_sequence_w_timestamp is None or \
                self.resampling_rate_in_secs is None or \
                self.sequence_shift is None:
            for attr in self.get_attributes():
                if getattr(self, attr) is not None:
                    raise ValueError('If you want to use the timeseries '
                                     'block, make sure that the parameters:'
                                     'resampling_rate, sequence_shift and'
                                     'with_timestamp are always defined.')

    @property
    def resampling_rate_in_secs(self):
        return self.__resampling_rate_in_secs

    @property
    def trip_gap_threshold_in_secs(self):
        return self.__trip_gap_threshold_in_secs

    @property
    def process_sequence_w_timestamp(self):
        return self.__process_sequence_w_timestamp

    @property
    def process_sequence_w_category(self):
        return self.__process_sequence_w_category

    @property
    def sequence_shift(self):
        return self.__sequence_shift

    @property
    def sequence_length(self):
        return self.__sequence_length

    @resampling_rate_in_secs.setter
    def resampling_rate_in_secs(self, resampling_rate):
        if isinstance(resampling_rate, (float, int)):
            self.__resampling_rate_in_secs = resampling_rate
        else:
            raise TypeError('Please provide a int or a float instead of '
                            'a {} for the resampling '
                            'rate.'.format(type(resampling_rate)))

    @trip_gap_threshold_in_secs.setter
    def trip_gap_threshold_in_secs(self, trip_gap_threshold):
        if isinstance(trip_gap_threshold, (float, int)):
            self.__trip_gap_threshold_in_secs = trip_gap_threshold
        else:
            raise TypeError('Please provide a int or a float instead of '
                            'a {} for the trip gap '
                            'threshold.'.format(type(trip_gap_threshold)))

    @process_sequence_w_timestamp.setter
    def process_sequence_w_timestamp(self, with_timestamp):
        if isinstance(with_timestamp, str):
            self.__process_sequence_w_timestamp = with_timestamp
        else:
            raise TypeError('Please provide a string value instead of '
                            'a {} for the name of the timestamp '
                            'column.'.format(type(with_timestamp)))

    @process_sequence_w_category.setter
    def process_sequence_w_category(self, with_category):
        if isinstance(with_category, str):
            self.__process_sequence_w_category = with_category
        else:
            raise TypeError('Please provide a string value instead of '
                            'a {} for the name of the category '
                            'column.'.format(type(with_category)))

    @sequence_shift.setter
    def sequence_shift(self, sequence_shift):
        if isinstance(sequence_shift, int):
            self.__sequence_shift = sequence_shift
        else:
            raise TypeError('Please provide a int value instead of '
                            'a {} for the sequence '
                            'shift.'.format(type(sequence_shift)))

    @sequence_length.setter
    def sequence_length(self, sequence_length):
        if isinstance(sequence_length, int):
            self.__sequence_length = sequence_length
        else:
            raise TypeError('Please provide a int value instead of '
                            'a {} for the sequence '
                            'length.'.format(type(sequence_length)))

    @resampling_rate_in_secs.deleter
    def resampling_rate_in_secs(self):
        self.__resampling_rate_in_secs = None

    @trip_gap_threshold_in_secs.deleter
    def trip_gap_threshold_in_secs(self):
        self.__trip_gap_threshold_in_secs = None

    @process_sequence_w_timestamp.deleter
    def process_sequence_w_timestamp(self):
        self.__process_sequence_w_timestamp = None

    @process_sequence_w_category.deleter
    def process_sequence_w_category(self):
        self.__process_sequence_w_category = None

    @sequence_shift.deleter
    def sequence_shift(self):
        self.__sequence_shift = None

    @sequence_length.deleter
    def sequence_length(self):
        self.__sequence_length = None


class Evaluator:

    def __init__(self,
                 slices=None,
                 metrics=None):

        self.__slices = None
        self.__metrics = None

        if slices is not None:
            self.slices = slices
        if metrics is not None:
            self.metrics = metrics

    @staticmethod
    def get_attributes():
        return {
            EvaluatorKeys.METRICS,
            EvaluatorKeys.SLICES,
        }

    def __str__(self):
        return to_pretty_string(self.to_serial())

    def __repr__(self):
        return to_pretty_string(self.to_serial(), style=PrintStyles.PPRINT)

    def to_serial(self):
        return serialize(self)

    def check_completion(self):
        pass

    @property
    def slices(self):
        return self.__slices

    @property
    def metrics(self):
        return self.__metrics

    @slices.setter
    def slices(self, slices):
        if isinstance(slices, list):
            for s in slices:
                if not isinstance(s, list):
                    raise TypeError('Please provide a list of lists for the '
                                    'slices.'.format(type(slices)))
            self.__slices = slices
        else:
            raise TypeError('Please provide a list of lists for the '
                            'slices.'.format(type(slices)))

    @metrics.setter
    def metrics(self, metrics):
        if isinstance(metrics, dict):
            for v in metrics.values():
                if not isinstance(v, list):
                    raise TypeError('Please provide either a dict of lists '
                                    '(multi-output) or list (single-output) '
                                    'for metrics.'.format(type(metrics)))
            self.__metrics = metrics
        elif isinstance(metrics, list):
            self.__metrics = metrics
        else:
            raise TypeError('Please provide either a dict of lists '
                            '(multi-output) or list (single-output) '
                            'for metrics.'.format(type(metrics)))

    @slices.deleter
    def slices(self):
        self.__slices = None

    @metrics.deleter
    def metrics(self):
        self.__metrics = None
