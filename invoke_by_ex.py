import tensorflow as tf

if __name__ == '__main__':
    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], "./model/Bi-LSTM/savedModel")
        graph = tf.get_default_graph()

        word_ids = sess.graph.get_tensor_by_name('inputX:0')
        dropoutKeepProb = sess.graph.get_tensor_by_name('dropoutKeepProb:0')
        char_ids = sess.graph.get_tensor_by_name('char_ids:0')
        predictions = sess.graph.get_tensor_by_name('output/predictions:0')
        scores = sess.run(predictions,
                          feed_dict={word_ids: [[]], char_ids: [[[]]], dropoutKeepProb: [1.0]})

        print(scores)

