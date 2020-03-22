import tensorflow as tf
import numpy as np

tflite_file = './model/Bi-LSTM/bi-lstm.tflite'


def test_lite_is_useful():
    interpreter = tf.lite.Interpreter(model_path=tflite_file)

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

    def __init__(self):
        pass

    def load_label2id_json(self):
        pass

    def load_word2id_json(self):
        pass

    def load_char2id_json(self):
        pass

    def generate_char_ids(self):
        pass

    def generate_word_ids(self):
        pass

    def get_class(self):
        pass

    def get_label(self):
        pass


if __name__ == '__main__':
    # test_lite_is_useful()
    lite_model = TFLiteModel()

