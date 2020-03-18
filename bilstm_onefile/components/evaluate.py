import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
import json

import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
import build_data
from bilstm_onefile.components.data import Dataset, config


def evaluate_model():
    # x = "this movie is full of references like mad max ii the wild one and many others the ladybug´s face it´s a clear reference or tribute to peter lorre this movie is a masterpiece we´ll talk much more about in the future"
    x = 'this is 123123'

    # 注：下面两个词典要保证和当前加载的模型对应的词典是一致的
    with open("./data/wordJson/word2idx.json", "r", encoding="utf-8") as f:
        word2idx = json.load(f)

    with open("./data/wordJson/label2idx.json", "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    idx2label = {value: key for key, value in label2idx.items()}

    # word list
    xIds = [word2idx.get(item, word2idx[build_data.UNK]) for item in x.split(" ")]
    if len(xIds) >= config.sequenceLength:
        xIds = xIds[:config.sequenceLength]
    else:
        xIds = xIds + [word2idx[build_data.PAD]] * (config.sequenceLength - len(xIds))

    # character list
    with open("./data/charJson/charToIndex.json") as f:
        char2index = json.load(f)

    with open("./data/charJson/indexToChar.json") as f:
        index2char = json.load(f)

    # setence 分解成char_ids
    setence = [item for item in x.split(" ")]

    char_ids = []
    for word in setence:
        ids = [char2index.get(item, char2index[build_data.UNK]) for item in word]
        char_ids.append(ids)

    # 先补充sequece数量
    if len(char_ids) < config.sequenceLength:
        for i in range(config.sequenceLength - len(char_ids)):
            char_ids.append([char2index[build_data.PAD]])
    elif len(char_ids) > config.sequenceLength:
        char_ids = char_ids[:config.sequenceLength]


    char_list = []
    for word in char_ids:
        temp_ids = []
        # 再补充每个word内的char数量
        if len(word) < config.word_length:
            for i in range(config.word_length - len(word)):
                word.append(char2index[build_data.PAD])
        elif len(word) > config.word_length:
            word = word[:config.word_length]
        temp_ids.append(word)
        char_list.append(temp_ids)

    graph = tf.Graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            checkpoint_file = tf.train.latest_checkpoint("./model/Bi-LSTM/model_lite/")
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # 获得需要喂给模型的参数，输出的结果依赖的输入值
            inputX = graph.get_operation_by_name("inputX").outputs[0]
            dropoutKeepProb = graph.get_operation_by_name("dropoutKeepProb").outputs[0]
            char_ids = graph.get_operation_by_name("char_ids").outputs[0]

            # 获得输出的结果
            predictions = graph.get_tensor_by_name("output/predictions:0")

            pred = sess.run(predictions, feed_dict={inputX: [xIds], dropoutKeepProb: [1.0], char_ids: [char_list]})[0]

    pred = [idx2label[item] for item in pred]
    print(pred)