from enum import Enum


class PipelineStatusTypes(Enum):
    NotStarted = 1
    Failed = 2
    Succeeded = 3
    Running = 4


class PipelineRunTypes(Enum):
    training = 1
    datagen = 2
    infer = 3
    test = 4
    eval = 5


class FunctionTypes(Enum):
    transform = 1
    model = 2


class TrainingTypes(Enum):
    gcaip = 1
    local = 2


class ServingTypes(Enum):
    gcaip = 1
    local = 2


class GDPComponent(Enum):
    SplitGen = 1
    SplitStatistics = 2
    SplitSchema = 3
    SequenceTransform = 4
    SequenceStatistics = 5
    SequenceSchema = 6
    PreTransform = 7
    PreTransformStatistics = 8
    PreTransformSchema = 9
    Transform = 10
    Trainer = 11
    Evaluator = 12
    ResultPackager = 13
    ModelValidator = 14
    Deployer = 15
    DataGen = 16
