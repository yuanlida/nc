import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
import datetime

import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
from .data import Dataset, config
from .helper import nextBatch, get_multi_metrics, get_binary_metrics, mean
from .model import BiLSTM

# %%
## Train and save concurrently

# 训练模型

# 生成训练集和验证集

def train_save():

    data = Dataset(config)
    data.dataGen()

    trainReviews = data.trainReviews
    trainChars = data.trainChars
    trainLabels = data.trainLabels

    evalReviews = data.evalReviews
    evalChars = data.evalChars
    evalLabels = data.evalLabels

    wordEmbedding = data.wordEmbedding
    labelList = data.labelList

    # 定义计算图
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率

        sess = tf.Session(config=session_conf)

        # 定义会话
        with sess.as_default():
            lstm = BiLSTM(config, wordEmbedding)

            globalStep = tf.Variable(0, name="globalStep", trainable=False)
            # 定义优化函数，传入学习速率参数
            optimizer = tf.train.AdamOptimizer(config.training.learningRate)
            # 计算梯度,得到梯度和变量
            gradsAndVars = optimizer.compute_gradients(lstm.loss)
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

            lossSummary = tf.summary.scalar("loss", lstm.loss)
            summaryOp = tf.summary.merge_all()

            trainSummaryDir = os.path.join(outDir, "train")
            trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)

            evalSummaryDir = os.path.join(outDir, "eval")
            evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)

            # 初始化所有变量
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # 保存模型的一种方式，保存为pb文件
            savedModelPath = "./model/Bi-LSTM/savedModel"
            if os.path.exists(savedModelPath):
                os.rmdir(savedModelPath)
            # builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

            sess.run(tf.global_variables_initializer())


            def trainStep(batchX, batchY, batchZ):
                """
                训练函数
                """
                feeddrop = [config.model.dropoutKeepProb] * len(batchX)

                feed_dict = {
                    lstm.inputX: batchX,
                    lstm.inputY: batchY,
                    lstm.char_ids: batchZ,
                    lstm.dropoutKeepProb: feeddrop
                }
                _, summary, step, loss, predictions = sess.run(
                    [trainOp, summaryOp, globalStep, lstm.loss, lstm.predictions],
                    feed_dict)

                timeStr = datetime.datetime.now().isoformat()

                if config.numClasses == 1:
                    acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)

                elif config.numClasses > 1:
                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                                  labels=labelList)

                trainSummaryWriter.add_summary(summary, step)

                return loss, acc, prec, recall, f_beta


            def devStep(batchX, batchY, batchZ):
                """
                验证函数
                """
                feeddrop = [1.0] * len(batchX)

                feed_dict = {
                    lstm.inputX: batchX,
                    lstm.inputY: batchY,
                    lstm.char_ids: batchZ,
                    lstm.dropoutKeepProb: feeddrop
                }
                summary, step, loss, predictions = sess.run(
                    [summaryOp, globalStep, lstm.loss, lstm.predictions],
                    feed_dict)

                if config.numClasses == 1:

                    acc, precision, recall, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
                elif config.numClasses > 1:
                    acc, precision, recall, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)

                evalSummaryWriter.add_summary(summary, step)

                return loss, acc, precision, recall, f_beta


            for i in range(config.training.epoches):
                # 训练模型
                print("start training model")
                for batchTrain in nextBatch(trainReviews, trainLabels, trainChars, config.batchSize):
                    loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1], batchTrain[2])

                    currentStep = tf.train.global_step(sess, globalStep)
                    print("trin: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        currentStep, loss, acc, recall, prec, f_beta))
                    if currentStep % config.training.evaluateEvery == 0:
                        print("\nEvaluation:")

                        losses = []
                        accs = []
                        f_betas = []
                        precisions = []
                        recalls = []

                        for batchEval in nextBatch(evalReviews, evalLabels, evalChars, config.batchSize):
                            loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1], batchEval[2])
                            losses.append(loss)
                            accs.append(acc)
                            f_betas.append(f_beta)
                            precisions.append(precision)
                            recalls.append(recall)

                        time_str = datetime.datetime.now().isoformat()
                        print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str,                                                                                      mean(recalls),
                                                                                                             mean(f_betas)))

                    if currentStep % config.training.checkpointEvery == 0:
                        # 保存模型的另一种方法，保存checkpoint文件
                        path = saver.save(sess, "./model/Bi-LSTM/model/my-model", global_step=currentStep)
                        print("Saved model checkpoint to {}\n".format(path))

            tf.compat.v1.saved_model.simple_save(sess,
                                       "./model/Bi-LSTM/savedModel",
                                       inputs={"inputX": lstm.inputX,
                                               "keepProb": lstm.dropoutKeepProb,
                                               "char_ids": lstm.char_ids
                                               },
                                       outputs={"predictions": lstm.predictions})
