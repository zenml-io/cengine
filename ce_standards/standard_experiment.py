from ce_standards.utils import ConfigKeys


class GlobalKeys(ConfigKeys):
    SPLIT = 'split'
    FEATURES = 'features'
    LABELS = 'labels'
    TRAINER = 'trainer'
    EVALUATOR = 'evaluator'
    PREPROCESSING = 'preprocessing'
    TIMESERIES_ = 'timeseries'
    PCA_ = 'pca'
    CUSTOM_CODE_ = 'custom_code'

    # following keys are MANUALLY added to the config during pipeline creation
    VERSION = 'version'
    BQ_ARGS_ = 'bq_args'


class DataSourceKeys(ConfigKeys):
    DATA_TYPE = 'type'
    DATA_SOURCE = 'source'
    ARGS = 'args'


class BQArgsKeys(ConfigKeys):
    PROJECT = 'project'
    DATASET = 'dataset'
    TABLE = 'table'


class GCSKeys(ConfigKeys):
    PATH = 'path'
    SERVICE_ACCOUNT = 'service_account'


class TimeSeriesKeys:
    RESAMPLING_RATE_IN_SECS = 'resampling_rate_in_secs'
    TRIP_GAP_THRESHOLD_IN_SECS = 'trip_gap_threshold_in_secs'

    PROCESS_SEQUENCE_W_TIMESTAMP = 'process_sequence_w_timestamp'
    PROCESS_SEQUENCE_W_CATEGORY_ = 'process_sequence_w_category'

    SEQUENCE_SHIFT = 'sequence_shift'
    SEQUENCE_LENGTH = 'sequence_length'


class SplitKeys(ConfigKeys):
    RATIO_ = 'ratio'
    CATEGORIZE_BY_ = 'categorize'
    INDEX_BY_ = 'index'
    WHERE_ = 'where'


class PCAKeys(ConfigKeys):
    NUM_DIMENSIONS = 'num_dimensions'


class TrainerKeys(ConfigKeys):
    FN = 'fn'
    PARAMS = 'params'


class DefaultKeys(ConfigKeys):
    STRING = 'string'
    INTEGER = 'integer'
    BOOLEAN = 'boolean'
    FLOAT = 'float'


class PreProcessKeys(ConfigKeys):
    RESAMPLING = 'resampling'
    FILLING = 'filling'
    TRANSFORM = 'transform'
    LABEL_TUNING = 'label_tuning'


class CustomCodeKeys(ConfigKeys):
    TRANSFORM_ = 'transform'
    MODEL_ = 'model'


class CustomCodeMetadataKeys(ConfigKeys):
    NAME = 'name'
    UDF_PATH = 'udf_path'
    STORAGE_PATH = 'storage_path'


class MethodKeys(ConfigKeys):
    METHOD = 'method'
    PARAMETERS = 'parameters'


class EvaluatorKeys(ConfigKeys):
    METRICS = 'metrics'
    SLICES = 'slices'
