import tensorflow as tf
import numpy as np
import string
import json
import build_data


tf_lite_file = './model/Bi-LSTM/bi-lstm.tflite'
char2index_file = './model/Bi-LSTM/charJson/charToIndex.json'
word2id_file = './model/Bi-LSTM/wordJson/word2idx.json'
label2id_file = './model/Bi-LSTM/wordJson/label2idx.json'


def test_lite_is_useful():
    interpreter = tf.lite.Interpreter(model_path=tf_lite_file)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    # Test model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_integers(0, 0, size=input_shape), dtype=np.int32)
    print(input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    input_shape = input_details[1]['shape']
    input_data = np.array(np.random.random_integers(0, 0, size=input_shape), dtype=np.float32)
    print(input_data)
    interpreter.set_tensor(input_details[1]['index'], input_data)

    input_shape = input_details[2]['shape']
    input_data = np.array(np.random.random_integers(0, 0, size=input_shape), dtype=np.int32)
    print(input_data)
    interpreter.set_tensor(input_details[2]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)


class TFLiteModel(object):
    label2id = None
    word2id = None
    char2id = None
    sentence = None
    chars = None
    words = None
    id2label = None

    def __init__(self, sentence):
        self.sentence = sentence

    def load_label2id_json(self):
        with open(label2id_file, "r", encoding="utf-8") as f:
            self.label2id = json.load(f)
            self.id2label = {value: key for key, value in self.label2id.items()}

    def load_word2id_json(self):
        with open(word2id_file, "r", encoding="utf-8") as f:
            self.word2id = json.load(f)

    def load_char2id_json(self):
        with open(char2index_file, "r", encoding="utf-8") as f:
            self.char2id = json.load(f)

    def gen_chars(self):
        pass

    def gen_words(self):
        pass

    def generate_char_ids(self):
        pass

    def generate_word_ids(self):
        pass

    def get_class(self):
        pass

    def get_label(self):
        pass

    def analyze(self):
        self.load_char2id_json()
        self.load_label2id_json()
        self.load_word2id_json()
        return 'labels'


if __name__ == '__main__':
    # test_lite_is_useful()
    sentence = 'my USA phone number is +1-202-555-0169'
    lite_model = TFLiteModel(sentence)
    label = lite_model.analyze()
    print(label)

