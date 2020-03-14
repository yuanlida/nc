# from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_ENABLE_CONTROL_FLOW_V2'] = '1'
import tensorflow as tf

# from tensorflow.saved_model.signature_def_utils import predict_signature_def
# from tensorflow.saved_model import tag_constants
#
#
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# print(mnist)


# converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("../model", tag_set=[tf.saved_model.SERVING])
#
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
#
# tflite_model = converter.convert()
# open("./result/edison_model.tflite", "wb").write(tflite_model)


# converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("../model_get", tag_set=[tf.saved_model.tag_constants.SERVING])
converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model("./model/textCNN/savedModel", tag_set=[tf.saved_model.tag_constants.SERVING])
# converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                         tf.lite.OpsSet.SELECT_TF_OPS]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]

tflite_model = converter.convert()
open("./convert/tf_lite.tflite", "wb").write(tflite_model)

print('ok')