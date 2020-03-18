# %%
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'

import warnings
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# from tensorflow.contrib.rnn import LSTMCell
from tensorflow.lite.experimental.examples.lstm.rnn_cell import TFLiteLSTMCell as LSTMCell

# from tensorflow.nn import bidirectional_dynamic_rnn
from tensorflow.lite.experimental.examples.lstm.rnn import bidirectional_dynamic_rnn

# 构建模型
class BiLSTM(object):
    """
    Bi-LSTM 用于文本分类
    """

    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")

        # shape = (batch size, max length of sentence, max length of word)
        self.char_ids = tf.placeholder(tf.int32, shape=[None, config.sequenceLength, config.word_length],
                                       name="char_ids")

        self.inputY = tf.placeholder(tf.int32, [None], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, [None], name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):

            with tf.variable_scope("words"):
                self.W = tf.Variable(
                    tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"),
                    name="_word_embeddings",
                    dtype=tf.float32,
                    trainable=True)

                self.word_embeddings = tf.nn.embedding_lookup(self.W,
                                                            self.inputX, name="word_embeddings")

            with tf.variable_scope("chars"):
                    # get char embeddings matrix
                _char_embeddings = tf.get_variable(
                    name="_char_embeddings",
                    dtype=tf.float32,
                    shape=[config.char_size, config.model.dim_char])
                char_embeddings = tf.nn.embedding_lookup(_char_embeddings,
                                                         self.char_ids, name="char_embeddings")

                # put the time dimension on axis=1
                s = tf.shape(char_embeddings)
                #     s[0]输入了多少行, s[1]是每行多少个词，s[-2]每个词多少个character
                char_embeddings = tf.reshape(char_embeddings,
                                             shape=[s[0] * s[1], s[-2], config.model.dim_char])
                # word_lengths = tf.reshape(self.word_lengths, shape=[s[0] * s[1]])

                char_embeddings = tf.transpose(char_embeddings, perm=[1, 0, 2])
                # 在cell层增加dropout

                cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                                        LSTMCell(config.model.hidden_size_char,
                                        state_is_tuple=True),
                                        output_keep_prob=self.dropoutKeepProb[0])
                cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                                        LSTMCell(config.model.hidden_size_char,
                                        state_is_tuple=True),
                                        output_keep_prob=self.dropoutKeepProb[0])

                _output = bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, char_embeddings,
                    # sequence_length=word_lengths,
                    dtype=tf.float32,
                    time_major=True
                )

                # # read and concat output
                #     不知道为什么这么用，是不是就是把time维度去掉了？
                _, ((_, output_fw), (_, output_bw)) = _output
                output = tf.concat([output_fw, output_bw], axis=-1)

                #     对于上述矩阵的这个转换对吗？
                output = tf.reshape(output,
                                    shape=[s[1], s[0], 2 * config.model.hidden_size_char])
                output = tf.transpose(output, perm=[1, 0, 2])
            word_embeddings = tf.concat([self.word_embeddings, output], axis=-1)

            # dropout不能在tflite上正常工作
            # self.embeddedWords = tf.nn.dropout(word_embeddings, self.dropoutKeepProb)
            self.embeddedWords = word_embeddings

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):

            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb[0])
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb[0])

                    # (? 32 500)(batch_size, max_time, cell_fw.output_size)
                    # change to (32 ? 500)(max_time, batch_size, cell_fw.output_size)

                    # e_s = tf.shape(self.embeddedWords)
                    # print(e_s)
                    if idx == 0:
                        self.embeddedWords = tf.transpose(self.embeddedWords, perm=[1, 0, 2])
                    outputs, self.current_state = bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                  self.embeddedWords, dtype=tf.float32,
                                                                                  scope="bi-lstm" + str(idx),
                                                                            time_major=True
                                                                            )

                    # (?, 32, 600) (batch_size, max_time, cell_fw.output_size)
                    self.embeddedWords = tf.concat(outputs, 2)
        self.embeddedWords = tf.transpose(self.embeddedWords, [1, 0, 2])
        # 去除最后时间步的输出作为全连接的输入
        finalOutput = self.embeddedWords[:, -1, :]

        outputSize = config.model.hiddenSizes[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        output = tf.reshape(finalOutput, [-1, outputSize])  # reshape成全连接层的输入维度

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.inputY, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

