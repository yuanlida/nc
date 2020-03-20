import tensorflow as tf

interpreter=tf.lite.Interpreter(model_path='./model/Bi-LSTM/bi-lstm.tflite')

interpreter.allocate_tensors()