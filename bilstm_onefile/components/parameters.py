class TrainingConfig(object):
    epoches = 1
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    #     hiddenSizes = [256, 256]  # 单层LSTM结构的神经元个数
    hiddenSizes = [300, 300]  # 单层LSTM结构的神经元个数

    dropoutKeepProb = 0.5
    l2RegLambda = 0.0

    embeddingSize = 300

    # embeddings
    dim_word = 300
    dim_char = 100

    # model hyperparameters
    hidden_size_lstm = 300  # lstm on word embeddings
    hidden_size_char = 100  # lstm on chars

class Config(object):
    sequenceLength = 32  # 取了所有序列长度的均值
    batchSize = 128
    word_length = 10

    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    #     alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

    dataSource = "../data/preProcess/labeledTrain.csv"

    stopWordSource = "../data/english"

    numClasses = 5  # 二分类设置为1，多分类设置为类别的数目

    rate = 0.8  # 训练集的比例

    embeddings = None

    train_embeddings = True

    use_chars = True

    training = TrainingConfig()

    model = ModelConfig()

    char_size = 0
