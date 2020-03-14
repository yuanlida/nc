# %%

import os
import random
import time
import datetime
import csv
import json
from math import sqrt
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import config as constant
import build_data
warnings.filterwarnings("ignore")


# %%

# 参数配置

class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001


class ModelConfig(object):
    # 该列表中子列表的三个元素分别是卷积核的数量，卷积核的高度，池化的尺寸
    convLayers = [[256, 7, 4],
                  [256, 7, 4],
                  [256, 3, 4]]
    #                   [256, 3, None],
    #                   [256, 3, None],
    #                   [256, 3, 3]]
    fcLayers = [512]
    dropoutKeepProb = 0.5

    epsilon = 1e-3  # BN层中防止分母为0而加入的极小值
    decay = 0.999  # BN层中用来计算滑动平均的值


class Config(object):
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    #     alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"

    sequenceLength = 500
    batchSize = 128

    numClasses = 5  # 二分类设置为1，多分类设置为类别的数目

    rate = 0.8  # 训练集的比例

    dataSource = "../data/preProcess/labeledCharTrain.csv"

    training = TrainingConfig()

    model = ModelConfig()


config = Config()


# %%

# 数据预处理的类，生成训练集和测试集

class Dataset(object):
    def __init__(self, config):
        self._dataSource = config.dataSource

        self._sequenceLength = config.sequenceLength
        self._rate = config.rate

        self.trainReviews = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalLabels = []

        self._alphabet = config.alphabet
        self.charEmbedding = None

        self._charToIndex = {}
        self._indexToChar = {}

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        reviews = []
        labels = []
        for index, file in enumerate(build_data.train_files):
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    review = [char for char in line]
                    reviews.append(review)
                    labels.append(constant.tags[index])

        # df = pd.read_csv(filePath)
        # labels = df["sentiment"].tolist()
        # review = df["review"].tolist()
        # reviews = [[char for char in line if char != " "] for line in review]

        return reviews, labels

    def _reviewProcess(self, review, sequenceLength, charToIndex):
        """
        将数据集中的每条评论用index表示
        wordToIndex中“pad”对应的index为0
        """

        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength

        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)

        for i in range(sequenceLen):
            if review[i] in charToIndex:
                reviewVec[i] = charToIndex[review[i]]
            else:
                reviewVec[i] = charToIndex[build_data.UNK]

        return reviewVec

    def _genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """

        reviews = []
        labels = []

        # 遍历所有的文本，将文本中的词转换成index表示

        for i in range(len(x)):
            reviewVec = self._reviewProcess(x[i], self._sequenceLength, self._charToIndex)
            reviews.append(reviewVec)

            labels.append([y[i]])

        trainIndex = int(len(x) * rate)

        datas = list(zip(reviews, y))
        random.shuffle(datas)
        reviews = []
        y = []
        for review, tag in datas:
            reviews.append(review)
            y.append(tag)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels

    def _genVocabulary(self, reviews):
        """
        生成字符向量和字符-索引映射字典
        """

        # 这个地方可能要换成所有字符
        chars = [char for char in self._alphabet]

        vocab, charEmbedding = self._getCharEmbedding(chars)
        self.charEmbedding = charEmbedding

        self._charToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToChar = dict(zip(list(range(len(vocab))), vocab))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/charJson/charToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._charToIndex, f)

        with open("../data/charJson/indexToChar.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToChar, f)

    def _getCharEmbedding(self, chars):
        """
        按照one的形式将字符映射成向量
        """

        alphabet = [build_data.UNK] + [char for char in self._alphabet]
        vocab = [build_data.PAD] + alphabet
        charEmbedding = []
        charEmbedding.append(np.zeros(len(alphabet), dtype="float32"))

        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype="float32")

            # 生成每个字符对应的向量
            onehot[i] = 1

            # 生成字符嵌入的向量矩阵
            charEmbedding.append(onehot)

        return vocab, np.array(charEmbedding)

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        # 初始化词汇-索引映射表和词向量矩阵
        self._genVocabulary(reviews)

        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviews, labels, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalLabels = evalLabels


data = Dataset(config)
data.dataGen()

# %%

print("train data shape: {}".format(data.trainReviews.shape))
print("train label shape: {}".format(data.trainLabels.shape))
print("eval data shape: {}".format(data.evalReviews.shape))
print("charEmbedding shape: {}".format(data.charEmbedding.shape))


# %%

# 输出batch数据集

def nextBatch(x, y, batchSize):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]

    numBatches = len(x) // batchSize

    for i in range(numBatches):
        start = i * batchSize
        end = start + batchSize
        batchX = np.array(x[start: end], dtype="int64")
        batchY = np.array(y[start: end], dtype="float32")

        yield batchX, batchY


# %%

# 定义char-CNN分类器

class CharCNN(object):
    """
    char-CNN用于文本分类
    """

    def __init__(self, config, charEmbedding):
        # placeholders for input, output and dropuot
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.float32, [None, 1], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.isTraining = tf.placeholder(tf.bool, name="isTraining")

        self.epsilon = config.model.epsilon
        self.decay = config.model.decay

        # 字符嵌入
        with tf.name_scope("embedding"):

            # 利用one-hot的字符向量作为初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(charEmbedding, dtype=tf.float32, name="charEmbedding"), name="W")
            # 获得字符嵌入
            self.embededChars = tf.nn.embedding_lookup(self.W, self.inputX)
            # 添加一个通道维度
            self.embededCharsExpand = tf.expand_dims(self.embededChars, -1)

        for i, cl in enumerate(config.model.convLayers):
            print("开始第" + str(i + 1) + "卷积层的处理")
            # 利用命名空间name_scope来实现变量名复用
            with tf.name_scope("convLayer-%s" % (i + 1)):
                # 获取字符的向量长度
                filterWidth = self.embededCharsExpand.get_shape()[2].value

                # filterShape = [height, width, in_channels, out_channels]
                filterShape = [cl[1], filterWidth, 1, cl[0]]

                stdv = 1 / sqrt(cl[0] * cl[1])

                # 初始化w和b的值
                wConv = tf.Variable(tf.random_uniform(filterShape, minval=-stdv, maxval=stdv),
                                    dtype='float32', name='w')
                bConv = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name='b')

                #                 w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
                #                 b_conv = tf.Variable(tf.constant(0.1, shape=[cl[0]]), name="b")
                # 构建卷积层，可以直接将卷积核的初始化方法传入（w_conv）
                conv = tf.nn.conv2d(self.embededCharsExpand, wConv, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # 加上偏差
                hConv = tf.nn.bias_add(conv, bConv)
                # 可以直接加上relu函数，因为tf.nn.conv2d事实上是做了一个卷积运算，然后在这个运算结果上加上偏差，再导入到relu函数中
                hConv = tf.nn.relu(hConv)

                #                 with tf.name_scope("batchNormalization"):
                #                     hConvBN = self._batchNorm(hConv)

                if cl[-1] is not None:
                    ksizeShape = [1, cl[2], 1, 1]
                    hPool = tf.nn.max_pool(hConv, ksize=ksizeShape, strides=ksizeShape, padding="VALID", name="pool")
                else:
                    hPool = hConv

                print(hPool.shape)

                # 对维度进行转换，转换成卷积层的输入维度
                self.embededCharsExpand = tf.transpose(hPool, [0, 1, 3, 2], name="transpose")
        print(self.embededCharsExpand)
        with tf.name_scope("reshape"):
            fcDim = self.embededCharsExpand.get_shape()[1].value * self.embededCharsExpand.get_shape()[2].value
            self.inputReshape = tf.reshape(self.embededCharsExpand, [-1, fcDim])

        # 保存的是神经元的个数[34*256, 1024, 1024]
        weights = [fcDim] + config.model.fcLayers

        for i, fl in enumerate(config.model.fcLayers):
            with tf.name_scope("fcLayer-%s" % (i + 1)):
                print("开始第" + str(i + 1) + "全连接层的处理")
                stdv = 1 / sqrt(weights[i])

                # 定义全连接层的初始化方法，均匀分布初始化w和b的值
                wFc = tf.Variable(tf.random_uniform([weights[i], fl], minval=-stdv, maxval=stdv), dtype="float32",
                                  name="w")
                bFc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype="float32", name="b")

                #                 w_fc = tf.Variable(tf.truncated_normal([weights[i], fl], stddev=0.05), name="W")
                #                 b_fc = tf.Variable(tf.constant(0.1, shape=[fl]), name="b")

                self.fcInput = tf.nn.relu(tf.matmul(self.inputReshape, wFc) + bFc)

                with tf.name_scope("dropOut"):
                    self.fcInputDrop = tf.nn.dropout(self.fcInput, self.dropoutKeepProb)

            self.inputReshape = self.fcInputDrop

        with tf.name_scope("outputLayer"):
            stdv = 1 / sqrt(weights[-1])
            # 定义隐层到输出层的权重系数和偏差的初始化方法
            #             w_out = tf.Variable(tf.truncated_normal([fc_layers[-1], num_classes], stddev=0.1), name="W")
            #             b_out = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            wOut = tf.Variable(tf.random_uniform([config.model.fcLayers[-1], 1], minval=-stdv, maxval=stdv),
                               dtype="float32", name="w")
            bOut = tf.Variable(tf.random_uniform(shape=[1], minval=-stdv, maxval=stdv), name="b")
            # tf.nn.xw_plus_b就是x和w的乘积加上b
            self.predictions = tf.nn.xw_plus_b(self.inputReshape, wOut, bOut, name="predictions")
            # 进行二元分类
            self.binaryPreds = tf.cast(tf.greater_equal(self.predictions, 0.0), tf.float32, name="binaryPreds")

        with tf.name_scope("loss"):
            # 定义损失函数，对预测值进行softmax，再求交叉熵。

            losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.predictions, labels=self.inputY)
            self.loss = tf.reduce_mean(losses)

    def _batchNorm(self, x):

        gamma = tf.Variable(tf.ones([x.get_shape()[3].value]))
        beta = tf.Variable(tf.zeros([x.get_shape()[3].value]))

        self.popMean = tf.Variable(tf.zeros([x.get_shape()[3].value]), trainable=False, name="popMean")
        self.popVariance = tf.Variable(tf.ones([x.get_shape()[3].value]), trainable=False, name="popVariance")

        def batchNormTraining():
            # 一定要使用正确的维度确保计算的是每个特征图上的平均值和方差而不是整个网络节点上的统计分布值
            batchMean, batchVariance = tf.nn.moments(x, [0, 1, 2], keep_dims=False)

            decay = 0.99
            trainMean = tf.assign(self.popMean, self.popMean * self.decay + batchMean * (1 - self.decay))
            trainVariance = tf.assign(self.popVariance,
                                      self.popVariance * self.decay + batchVariance * (1 - self.decay))

            with tf.control_dependencies([trainMean, trainVariance]):
                return tf.nn.batch_normalization(x, batchMean, batchVariance, beta, gamma, self.epsilon)

        def batchNormInference():
            return tf.nn.batch_normalization(x, self.popMean, self.popVariance, beta, gamma, self.epsilon)

        batchNormalizedOutput = tf.cond(self.isTraining, batchNormTraining, batchNormInference)
        return tf.nn.relu(batchNormalizedOutput)


# %%

# 定义性能指标函数

def mean(item):
    return sum(item) / len(item)


def genMetrics(trueY, predY, binaryPredY):
    """
    生成acc和auc值
    """

    auc = roc_auc_score(trueY, predY)
    accuracy = accuracy_score(trueY, binaryPredY)
    precision = precision_score(trueY, binaryPredY, average='macro')
    recall = recall_score(trueY, binaryPredY, average='macro')

    return round(accuracy, 4), round(auc, 4), round(precision, 4), round(recall, 4)


# %%

# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

charEmbedding = data.charEmbedding

# 定义计算图
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth = True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

    sess = tf.Session(config=session_conf)

    # 定义会话
    with sess.as_default():

        cnn = CharCNN(config, charEmbedding)
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.RMSPropOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(cnn.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)

        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))

        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))

        lossSummary = tf.summary.scalar("trainLoss", cnn.loss)

        summaryOp = tf.summary.merge_all()

        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # 保存模型的一种方式，保存为pb文件
        #         builder = tf.saved_model.builder.SavedModelBuilder("../model/charCNN/savedModel")
        sess.run(tf.global_variables_initializer())


        def trainStep(batchX, batchY):
            """
            训练函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: config.model.dropoutKeepProb,
                cnn.isTraining: True
            }
            _, summary, step, loss, predictions, binaryPreds = sess.run(
                [trainOp, summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)
            timeStr = datetime.datetime.now().isoformat()
            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)
            print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(timeStr, step, loss, acc,
                                                                                               auc, precision, recall))
            trainSummaryWriter.add_summary(summary, step)


        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
                cnn.inputX: batchX,
                cnn.inputY: batchY,
                cnn.dropoutKeepProb: 1.0,
                cnn.isTraining: False
            }
            summary, step, loss, predictions, binaryPreds = sess.run(
                [summaryOp, globalStep, cnn.loss, cnn.predictions, cnn.binaryPreds],
                feed_dict)

            acc, auc, precision, recall = genMetrics(batchY, predictions, binaryPreds)

            evalSummaryWriter.add_summary(summary, step)

            return loss, acc, auc, precision, recall


        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                trainStep(batchTrain[0], batchTrain[1])

                currentStep = tf.train.global_step(sess, globalStep)
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")

                    losses = []
                    accs = []
                    aucs = []
                    precisions = []
                    recalls = []

                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, auc, precision, recall = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        aucs.append(auc)
                        precisions.append(precision)
                        recalls.append(recall)

                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {}, auc: {}, precision: {}, recall: {}".format(time_str,
                                                                                                       currentStep,
                                                                                                       mean(losses),
                                                                                                       mean(accs),
                                                                                                       mean(aucs),
                                                                                                       mean(precisions),
                                                                                                       mean(recalls)))

#                 if currentStep % config.training.checkpointEvery == 0:
#                     # 保存模型的另一种方法，保存checkpoint文件
#                     path = saver.save(sess, "../model/charCNN/model/my-model", global_step=currentStep)
#                     print("Saved model checkpoint to {}\n".format(path))

#         inputs = {"inputX": tf.saved_model.utils.build_tensor_info(cnn.inputX),
#                   "keepProb": tf.saved_model.utils.build_tensor_info(cnn.dropoutKeepProb)}

#         outputs = {"binaryPreds": tf.saved_model.utils.build_tensor_info(cnn.binaryPreds)}

#         prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
#                                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
#         legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
#         builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
#                                             signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

#         builder.save()
