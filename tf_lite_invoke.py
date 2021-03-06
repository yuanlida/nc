import tensorflow as tf
import string
import json
import build_data
import numpy as np

# copy this files to ./model/ folder.
tf_lite_file = './model/Bi-LSTM/bi-lstm.tflite'
char2index_file = './model/Bi-LSTM/charJson/charToIndex.json'
word2id_file = './model/Bi-LSTM/wordJson/word2idx.json'
label2id_file = './model/Bi-LSTM/wordJson/label2idx.json'

# tf_lite_file = './bilstm_onefile23/bi-lstm.tflite'
# char2index_file = './bilstm_onefile23/charJson/charToIndex.json'
# word2id_file = './bilstm_onefile23/wordJson/word2idx.json'
# label2id_file = './bilstm_onefile23/wordJson/label2idx.json'

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
    input_data = np.array(np.random.random_integers(0, 0, size=(input_shape[0], input_shape[1], input_shape[2])),
                          dtype=np.int32)
    # print(input_data)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    input_shape = input_details[1]['shape']
    input_data = np.array(np.random.random_integers(0, 0, size=(2, input_shape[0])), dtype=np.float32)
    # print(input_data)
    interpreter.set_tensor(input_details[1]['index'], input_data)

    input_shape = input_details[2]['shape']
    input_data = np.array(np.random.random_integers(0, 0, size=(2, input_shape[0], input_shape[1])), dtype=np.int32)
    # print(input_data)
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
    chars = []
    words = []
    id2label = None

    id_label = None
    sentence_words = []
    sentence_chars = []

    sequenceLength = 20
    word_length = 10

    def __init__(self, sentence):
        if sentence != None:
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
        for word in self.sentence_words:
            for _char in word:
                self.sentence_chars.append(_char)

    def gen_words(self):
        for item in string.punctuation:
            self.sentence = self.sentence.strip().replace(item, ' ' + item + ' ')
        self.sentence_words = self.sentence.strip().split()

    def generate_char_ids(self):
        char_ids = []
        for word in self.sentence_words:
            # use the original num
            ids = [self.char2id.get(item, self.char2id[build_data.UNK]) for item in word]
            # replace num with <num>
            # ids = [self.char2id.get(item, self.char2id[build_data.UNK])
            #        if not item.isdigit() else self.char2id[build_data.NUM]
            #        for item in word]
            char_ids.append(ids)

        # 先补充sequece数量
        if len(char_ids) < self.sequenceLength:
            for i in range(self.sequenceLength - len(char_ids)):
                char_ids.append([self.char2id[build_data.PAD]])
        elif len(char_ids) > self.sequenceLength:
            char_ids = char_ids[:self.sequenceLength]

        char_list = []
        for word in char_ids:
            # temp_ids = []
            # 再补充每个word内的char数量
            if len(word) < self.word_length:
                for i in range(self.word_length - len(word)):
                    word.append(self.char2id[build_data.PAD])
            elif len(word) > self.word_length:
                word = word[:self.word_length]
            # temp_ids.append(word)
            char_list.append(word)
        self.chars = np.asarray([char_list], dtype=np.int32)

    def generate_word_ids(self):
        words = [self.word2id.get(item, self.word2id[build_data.UNK])
                 if not item.isdigit() else self.word2id[build_data.NUM]
                 for item in self.sentence_words]
        # xIds = [word2idx.get(item, word2idx[build_data.UNK]) for item in x.split(" ")]
        if len(words) >= self.sequenceLength:
            words = words[:self.sequenceLength]
        else:
            words = words + [self.word2id[build_data.PAD]] * (self.sequenceLength - len(words))
        self.words = np.asarray([words], np.int32)

    def get_class(self):
        interpreter = tf.lite.Interpreter(model_path=tf_lite_file)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        # print(input_details)
        # print(output_details)

        # Test model on random input data.
        # [1, 32, 10] chars
        interpreter.set_tensor(input_details[0]['index'], self.chars)

        # [1.0] dropout
        dropout_array = [1.0]
        np_dropout = np.asarray(dropout_array, dtype=np.float32)
        interpreter.set_tensor(input_details[1]['index'], np_dropout)

        # [1, 32] words
        interpreter.set_tensor(input_details[2]['index'], self.words)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        self.id_label = output_data[0]

    def get_label(self):
        return self.id2label[self.id_label]

    def set_sentence(self, sentence):
        self.sentence = sentence

    def analyze(self):
        self.load_char2id_json()
        self.load_label2id_json()
        self.load_word2id_json()

        self.gen_words()
        self.gen_chars()

        self.generate_word_ids()
        self.generate_char_ids()

        self.get_class()
        return self.get_label()


if __name__ == '__main__':
    # test_lite_is_useful()
    # xs = ['Bruce Jimenez',
    #       'Growth Team',
    #       'Castle I LinkedIn',
    #       'San Francisco, CA']
    

    body = """(BRIAN) HEXTER
Ron Ben-Zeev
marjorie lau
Jeetender Chugh (जितेन्द्र चुघ)
Donato 
Christian
Lonny Cloud
Peter Barquet
dawn shrum
Nilesh Sapar
Scott Konradt
Francisco Garcia
Hanan Ben-Avi
～～～～～～～"""
    body = """DAEL OATES
Harald  Albrecht
KOWITZ
Julie Kowitz Margolies
FLAMINIANO Ronald Jonn Villamil
Evan de Barra AKA Karl
Rob Record
Alexander
Tolivar Wills
Joseph Kelly
L.J. Santillan
Doug Swift
elina scorey
CHASSEN HAYNES
Morris (Moe) Vervaecke
CHRIS LLOYD
San Jose, CA 95129
San Jose CA 95129
Los Gatos CA 95032
1559 Union Ave. Suite # 142
twitter
Sampler
1385 S. Colorado Blvd.
SN-23 Rahul Street,Pune
Suite A-220
Denver, CO 80222
15, S.C Dey Street
Vidyasagar Tower
Kolkata - 700073
"1 Fusionopolis Place
Level #03-20
Galaxis West Lobby
Singapore 138522
To manage your subscription preferences, login to your Sampler account
To manage your subscription preferences,
login to your Sampler account
Change Password, Didn't request this change?
Take care,
Have questions, suggestions, or feedback? Drop us a line at feedback@topinetworks.com
Manage notifications , Give feedback . Privacy policy
Forgot Password
Today's topic summary, I apologise,the alg obviously stops when
2225C Patrick F. Taylor Hall
Baton Rouge, LA    70803
12th Floor, Tower-B, SAS Tower, Medicity, Sector-38 Gurgaon -    122001GuruGram   -     122001
Flight Department
Owner  &  Operator
ATTORNEY AT LAW
Associate Dean for Academic Affairs
Asst. Manager - Medico Marketing
Make it happen in your career"""

    body = """Middle Schoool Track & Field Head Coach
High School Soccer Asst. Coach
IT Support
Rob Record
Carla Roppo-Owczarek
& Systems
admin
Lisa white
IT Support
Ian Hamilton
Comercial strategy office
wildmoor
Responsable Ventes Partenaires
Sr. Systems Engineer / Technology
Managing Director, Estates Division
(BRIAN) HEXTER
Jordan McDonald
PO Box 7193
Gujarat, INDIA
Legal, financial, technical translations
Senior Engineer - Projects & Services
Chargé de projets """


    xs = body.split('\n')
    lite_model = TFLiteModel(None)

    for sentence in xs:
        lite_model.set_sentence(sentence)
        label = lite_model.analyze()
        print(sentence, '：label is ', label)
