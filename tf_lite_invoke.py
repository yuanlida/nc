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

    sequenceLength = 32
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
    xs = [
        # 'Rory Webb',
        #   'Industrial Engineer',
        #   'Portfolio',
        #   'P: (415)555-6689',
        #   'C:(415)555-9879',
        #   '17a St Wich Street, City Centre, Bristol',
        #   '289 Kentish Town Road , London',
        #   '564 Market Street, Suite 700, San Francisco CA 94104',
        #   'San Francisco, CA, 94107',
        #   'Jeffrey',
        #   'Crunchbase, Inc.',
        #   '564 Market Street, Suite 700',
        #   'San Francisco, CA 94104',
        #   'San Francisco, CA, 94104',
        #   'Lida Yuan',
        #   'Lining Shi',
        #   'Lida',
        #   'Lining',
        #    'Guoqing Cai',
        #     'Xinqiang Jia',
        # 'Jing Zhang',
        # 'Jun Gu',
        # 'Dalio'
        # 'THEAGENCY',
        # 'A Global Marketing and Sales Organization',
        # 'RITA J WHITNEY',
        # 'Managing Director, Estates Division',
        # 'm: 626 755 4988',
        # 'TheAgencyRE.com',
        # 'CalBRE# 01209004',
        #   'Villa Seminia, 8, Sir Temi Zammit Avenue, Ta\'Xbiex XBX1011, Matlat',
        #   'President & CEO',
        #   'John E. Sestina CFP®, ChFC',
        #   'Ryan C. Unger',
        #   'A: 3580 Progress Drive Ste L Bensalem, PA',
        #   '3580 Progress Drive Ste L Bensalem, PA',
        #   'TEAM PENNSYLVANIA | WORKING. TOGETHER.',
        #   'Dr. Brain Fann',
        # 'Brain Fann',
        # 'Phone: (931)381-6881'

        # '+ 61 (0) 407 028 883',
        # '509.710.1943',
        # '+34 619730167',
        # '505-401-6046',
        # 'Mob : 0086-180-73155526',
        # '917.533.1773',
        # '360.632.0757',
        # '514-941-2722',
        # 'main 877-777-0363     direct 727-230-7877',
        # '0086-731-84321966',
        # '845.258.3986 ',
        # '425.507.7669',
        # '514-737-7724',

        # 'CEO',
        # 'Senior Account Executive, Team SI',
        # 'Admin',
        # 'President & CEO',
        # 'Vice President',
        # 'President & CEO',
        # 'Senior Web Front-end Developer, Web Services',
        # 'Division Director, Aquila',
        # 'FILMMAKER | PRODUCER',
        # 'Communications & Marketing',
        # 'Demonstrator',
        # 'Head of Marketing at Woodpecker.co',
        # 'Industrial Engineer ',
        # 'Commerce and B2B Solutions ',
        # 'Marketer ',
        # 'Sr. Director',
        # 'Sr. Director Strategic Data Solutions',
        # 'Director, Strategic Analytic Solutions',
        # 'Marketing & PR Manager',
        # 'SVP, Strategic Data Solutions',
        # 'SVP Strategic Data Solutions',
        # 'Director',
        # 'Marketing Associate',
        # 'Sales Manager',
        # 'Kind regards,Robert Lucas, Jr.',
        # 'Product Specialist Director',

        # 'Shannon Dique',
        # 'Lionel Baret',
        # 'Lisa White ',
        # 'Chad Sproles',
        # 'Marnich Winterbach',
        # 'Nawaf Totonchy',
        # 'RITA J WHITNEY',
        # 'Kosuke Futsukaichi',
        # 'Jerry Armstrong',
        # 'Chris Metcalfe',
        # 'Bob Phillips',
        # 'Reed Piernock',
        # 'Alex Luton',
        # 'Dr. Rakesh Kumar Sharma',
        # 'Michael Klyhn Orkild',
        # 'Brenda A Lewis, LCSW',
        # 'David Hughes',
        # 'Chad Sproles',
        # 'NICK WOOLGAR',
        # 'Brenda Tapp',
        # 'Niklas Göke',
        # 'MACKENZIE DECLARK',
        # 'Cathy Patalas',
        # 'Toby | Torbjörn Ungvall ',
        # 'Brent Butchard ',
        # 'Thomas Smith',
        # 'Marcus Athari',
        # 'Alison (Bartosik) Beaver',
        # 'Richard Alicea',
        # 'Gregory Allen-Anderson',
        # 'Malik J FERNANDO​',
        # 'Hussein Kanji ',
        # 'Harlan Donato',
        # 'Mark Damia',
        # 'Alec Wnuk ',
        # 'L\'Agnese Frank',
        # 'David Staples',
        # 'Kloudless',
        # 'Daniel Wacker',
        # 'Guoqing Cai',
        # 'Charles A. "Chip" Rabus',
        # 'Dr. Brain Fann',
        # 'A. Michael Spence',
        # 'Jeremy Monday ',
        # 'Lyrics by Jason Gray',
        # 'Daniel Wacker ',
        # 'Jayson T. Garrett',
        # 'Gregory Allen-Anderson'

        # 'A: 3580 Progress Drive Ste L Bensalem, PA',
        # '26 West 9th Street Suite 4D, New York, NY 10011',
        # '548 Market, St #64304 San Francisco CA - 94104',
        # 'Family First   5509 W Gray Street  Suite 100  Tampa,  FL   33609   United States',
        # 'FIPLAB Ltd · 1 Alfred Place · London, England WC1E 7EB · United Kingdom',
        # '1 Infinite Loop, MS 96-DM, Cupertino, CA 95014.',
        # '300 BOYLSTON AVE E SEATTLE WA 98102 USA',
        # '11 Walling Road',
        # 'San Jose, CA 95113',
        # '2054 University Ave,Berkeley,CA',
        # '101 Independence Ave. S.E. Washington, D.C. 20559-6000'

        ############0331
        # 'Phone - 0141 554 5555',
        # 'M: 0435148612',
        # '0481091121',
        # 'M: 07973 734 297',
        # '|t. +618 9387 3156',
        # 'M: +47 90 21 23 73',
        # '0729366703',
        # '+6421433473',
        # 'mobile: 07478 441488',
        # 'M: 07753 86 35 01',
        # '0481091121',
        # '+27 83 550 1444',
        # 'T 62262000 | F 62200066',
        # '919811411632',
        # 'Tel: 4087 0408',
        #
        # 'Admin',
        # 'CEO',
        # 'Responsable Ventes Partenaires',
        # 'Managing Director, Estates Division',
        # 'Technical Sales & Services',
        # 'President & CEO',
        # 'Division Director',
        # 'VC, SIMATS, Chennai and Former Director, DFRL',
        #
        # 'Shannon Dique',
        # 'Philip Williams',
        # 'Nawaf Totonchy',
        # 'Max Mathias',
        # 'Polina',
        # 'Lisa White',
        # 'RITA J WHITNEY',
        # 'Philip Williams',
        # 'Bob Phillips',
        # 'Reed Piernock',
        # 'Rakesh Kumar Sharma'
        # 'Dr. Rakesh Kumar Sharma',
        # 'Michael Klyhn Orkild',
        # 'Brenda A Lewis, LCSW',
        # 'David Hughes',
        # 'Ron Ben-Zeev',
        # 'Pavel Strigunov',
        # 'W. Mark Price',
        # 'Christine D. Price',

        ##############
        '941-735-0040',
        'Mobile: 610-564-1243',
        'Text // 202-630-4865',
        '01527 558282',
        '0470399574',
        't.  613.606.3608',
        'OFFICE: 407.545.3030 ext. 221 CELL: 407.221.7000',
        'm: 626 755 4988',
        'T 62262000 | F 62200066 ',
        '919811411632',
        'Phone: 212-691-7883',
        '919.594.6600',
        'Office: 501.553.9321 Mobile: 501.909.9243',
        'O: (215) 788-8485M: (267) 816-7268',
        '505-454-9810',
        't. 613.606.3608',
        'Phone - 0141 554 5555',
        'M: 0435148612',
        '|t. +618 9387 3156',
        'mobile: 07478 441488',
        'T: 0161 798 8117M: 07753 86 35 01',
        'office: 205.879.3282 x5548',
        '877-UNFORGETTABLE',

        'Ignition Toll Free',
        'Outreachly, Inc ',
        'Graduate Student in Communication, Culture & Technology',
        'CENTRALSCANNINGLtd',
        'Zinger Oilfield Supplies Ltd',
        'THEAGENCY',
        'ERA Realty Network Pte Ltd',
        'VictoryPlus',
        'Triangle Parent Media Group',
        'For: Doigs Ltd',
        'Clyne Media, Inc.',
        'Aus. Civil & Utilities',
        'Fresh Convenience Catering',
        'TORGY LNG',
        'N W E Waste Services Limited',

        'Tour Manager/Merch Dealer',
        'Production & Photo Assistant',
        'Union Transfer | Box Office Manager',
        'VP, Marketing ',
        'Senior Web Front-end Developer, Web Services',
        'Technical Manager',
        'PRESIDENT/CEO',
        'IT System Manager',
        'Managing Director, Estates Division',
        'Division Director, Aquila',
        'VC, SIMATS, Chennai and Former Director, DFRL',
        'LCSW',
        'Office: 501.553.9321',
        'Vice President Technical Sales & Services',
        'President & CEO',
        'Admin',
        'Responsable Ventes Partenaires',
        'Operations Manager',
        'CEO',
        'Director',

        'Chris Metcalfe',
        'Jim Morrin ♉',
        '(BRIAN) HEXTER',
        'Zoe Weber ',
        'Reed Piernock (they/them)',
        'John',
        'Rob Sweetlove',
        'Chandru',
        'Maya Maya Wysokinska',
        'Ron Ben-Zeev',
        'RITA J WHITNEY',
        'Dr. Rakesh Kumar Sharma',
        'Brenda A Lewis, LCSW',
        'David Hughes',
        'Chad Sproles',
        'Jerry Armstrong',
        'John',
        'John E. Sestina CFP®, ChFC',
        'Bob Phillips',
        'Maya',
        'Maya Wysokinska',
        'Iain Forsyth',
        'Shannon Dique',
        'Nawaf Totonchy',
        'Polina',
        'Lisa White',

        'Bodilsgade 3. 1.Th 1732 København V Danmark',
        'a: 811 W 7th St, Los Angeles, CA 90017 ',
        'Unit 9 Wildmoor Mill | Mill Lane | Wildmoor | Worcestershire | B61 0BX',
        'Address: PO Box 7193 Drayton Valley, Alberta T7A 1S4',
        'Diamond Plaza 34 Le Duan, District 1, HCMC, Vietnam.',
        'ERA@MBSQ: 229 Mountbatten Road #03-01 Singapore 398007',
        '61, New York Tower A Nr. Thaltej Circle, SG Highway, Ahmedabad - 380054 Gujarat, INDIA ',
        '26 West 9th Street Suite 4D New York, NY 10011',
        'A: 3580 Progress Drive Ste L Bensalem, PA',
        '|s. 40 Alexander Street, Wembley WA 601',
    ]
    lite_model = TFLiteModel(None)

    for sentence in xs:
        lite_model.set_sentence(sentence)
        label = lite_model.analyze()
        print(sentence, '| label is ', label)
