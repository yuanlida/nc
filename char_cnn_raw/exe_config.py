from .training_config import TrainingConfig
from .model_config import ModelConfig

class EXEConfig(object):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    #     alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

    sequenceLength = 200
    batchSize = 128

    numClasses = 5  # 二分类设置为1，多分类设置为类别的数目

    rate = 0.8  # 训练集的比例

    dataSource = "../data/preProcess/labeledCharTrain.csv"

    training = TrainingConfig()

    model = ModelConfig()