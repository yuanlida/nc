import os

os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import random
import json

import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
from build_data import train_files
import build_data
import config as constant

# 数据预处理的类，生成训练集和测试集
from .parameters import Config

config = Config()

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource

        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate

        self._stopWordDict = {}

        self.trainReviews = []
        self.trainChars = []
        self.trainLabels = []

        self.evalReviews = []
        self.evalChars = []
        self.evalLabels = []

        self.wordEmbedding = None

        self._alphabet = config.alphabet
        self.charEmbedding = None

        self._charToIndex = {}
        self._indexToChar = {}

        self.labelList = []

    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """

        reviews = []
        labels = []
        for index, file in enumerate(train_files):
            # if index != 3:
            #     continue
            with open(file) as f:
                lines = f.readlines()
                for line in lines:
                    reviews.append(line.strip().split())
                    labels.append(constant.tags[index])
        return reviews, labels

    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[label] for label in labels]
        return labelIds

    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = [[word2idx.get(item, word2idx[build_data.UNK]) if not item.isdigit() else word2idx[build_data.NUM] for item in review] for review in reviews]
        # reviewIds = [[word2idx.get(item, word2idx[build_data.UNK]) for item in review] for review in reviews]
        return reviewIds

    def _genTrainEvalData(self, x, y, word2idx, rate, char_ids):
        """
        生成训练集和验证集
        """
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx[build_data.PAD]] * (self._sequenceLength - len(review)))

        char_list = []
        for ids in char_ids:
            if len(ids) < config.sequenceLength:
                for i in range(config.sequenceLength - len(ids)):
                    ids.append([self._charToIndex[build_data.PAD]])
            elif len(ids) > config.sequenceLength:
                ids = ids[:config.sequenceLength]
            temp_ids = []
            for word in ids:
                if len(word) < config.word_length:
                    for i in range(config.word_length - len(word)):
                        word.append(self._charToIndex[build_data.PAD])
                elif len(word) > config.word_length:
                    word = word[:config.word_length]
                temp_ids.append(word)
            char_list.append(temp_ids)

        trainIndex = int(len(x) * rate)

        # 随机取值
        datas = list(zip(reviews, y, char_list))
        random.shuffle(datas)
        reviews = []
        y = []
        char_ids = []
        for review, tag, chars in datas:
            reviews.append(review)
            y.append(tag)
            char_ids.append(chars)

        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainChars = np.asarray(char_ids[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")

        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalChars = np.array(char_ids[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, trainChars, evalReviews, evalLabels, evalChars

    def _gen_char_vacablulary(self):
        chars = [char for char in self._alphabet]
        vocab = self._getCharEmbedding(chars)
        self._charToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self._indexToChar = dict(zip(list(range(len(vocab))), vocab))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/charJson/charToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self._charToIndex, f)

        with open("../data/charJson/indexToChar.json", "w", encoding="utf-8") as f:
            json.dump(self._indexToChar, f)

        config.char_size = len(self._indexToChar)


    def _getCharEmbedding(self, chars):
        """
        按照one的形式将字符映射成向量
        """

        alphabet = [build_data.UNK] + [build_data.NUM] + [char for char in self._alphabet]
        vocab = [build_data.PAD] + alphabet
        # charEmbedding = []
        # charEmbedding.append(np.zeros(len(alphabet), dtype="float32"))
        #
        # for i, alpha in enumerate(alphabet):
        #     onehot = np.zeros(len(alphabet), dtype="float32")
        #
        #     # 生成每个字符对应的向量
        #     onehot[i] = 1
        #
        #     # 生成字符嵌入的向量矩阵
        #     charEmbedding.append(onehot)

        return vocab  # , np.array(charEmbedding)

    def _genVocabulary(self, reviews, labels):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """

        allWords = [word for review in reviews for word in review]

        # 去掉停用词
        # subWords = [word for word in allWords if word not in self.stopWordDict]

        wordCount = Counter(allWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)

        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 1]

        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding

        word2idx = dict(zip(vocab, list(range(len(vocab)))))

        uniqueLabel = list(set(labels))
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        self.labelList = list(range(len(uniqueLabel)))

        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f)

        with open("../data/wordJson/label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f)

        return word2idx, label2idx

    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """

        # wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []

        # 添加 "<pad>" 和 "<UNK>", "<NUM>"
        vocab.append(build_data.PAD)
        vocab.append(build_data.UNK)
        vocab.append(build_data.NUM)
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))

        for word in words:
            try:
                wordEmbedding.append(np.random.randn(self._embeddingSize))
                vocab.append(word)
            except:
                print(word + "无法生成随机矩阵")
        # for word in words:
        #     try:
        #         vector = wordVec.wv[word]
        #         vocab.append(word)
        #         wordEmbedding.append(vector)
        #     except:
        #         print(word + "不存在于词向量中")

        return vocab, np.array(wordEmbedding)

    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """

        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))

    def _char_to_ids(self, reviews, char2id):
        char_ids = []
        for setence in reviews:
            setence_ids = []
            for word in setence:
                # ids = [char2id.get(item, char2id[build_data.UNK]) for item in word]
                ids = [char2id.get(item, char2id[build_data.UNK]) if not item.isdigit() else char2id[build_data.NUM] for item in word]
                setence_ids.append(ids)
            char_ids.append(setence_ids)
        return char_ids

    def dataGen(self):
        """
        初始化训练集和验证集
        """

        # 初始化停用词
        self._readStopWord(self._stopWordSource)
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)

        self._gen_char_vacablulary()
        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)

        # 将标签和句子数值化
        labelIds = self._labelToIndex(labels, label2idx)
        reviewIds = self._wordToIndex(reviews, word2idx)
        char_ids = self._char_to_ids(reviews, self._charToIndex)

        # 初始化训练集和测试集
        trainReviews, trainLabels, train_chars, evalReviews, evalLabels, eval_chars = self._genTrainEvalData(reviewIds,
                                                                                                             labelIds,
                                                                                                             word2idx,
                                                                                                             self._rate,
                                                                                                             char_ids)
        self.trainReviews = trainReviews
        self.trainChars = train_chars
        self.trainLabels = trainLabels

        self.evalReviews = evalReviews
        self.evalChars = eval_chars
        self.evalLabels = evalLabels


data = Dataset(config)
data.dataGen()

# %%

print("train data shape: {}".format(data.trainReviews.shape))
print("train label shape: {}".format(data.trainLabels.shape))
print("eval data shape: {}".format(data.evalReviews.shape))

