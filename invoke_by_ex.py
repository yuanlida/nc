import tensorflow as tf
import numpy as np

# input = np.arange(32).reshape(1, 32)
input = np.zeros(shape=(1, 32))
# print(input)

# chars = np.arange(320).reshape(1, 32, 10)
chars = np.zeros(shape=(1, 32, 10))
# print(chars)

chars = np.zeros(shape=(1, 32, 10))
# print(chars)
if __name__ == '__main__':
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./model/Bi-LSTM/savedModel")
        graph = tf.get_default_graph()

        inputX = sess.graph.get_tensor_by_name('inputX:0')
        dropoutKeepProb = sess.graph.get_tensor_by_name('dropoutKeepProb:0')
        char_ids = sess.graph.get_tensor_by_name('char_ids:0')
        predictions = sess.graph.get_tensor_by_name('output/predictions:0')
        scores = sess.run(predictions,
                          feed_dict={inputX: input.tolist(), char_ids: chars.tolist(), dropoutKeepProb: [1.0]})

        print(scores)
