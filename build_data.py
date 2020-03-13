import json
import numpy as np

UNK = "<UNK>"
NUM = "<NUM>"
NONE = "O"



# train_files = ['./data/train/loc.txt',
#                './data/train/name.txt',
#                './data/train/org.txt',
#                './data/train/tel.txt',
#                './data/train/tit.txt']
#
# test_files = ['./data/test/loc.txt',
#               './data/test/name.txt',
#               './data/test/org.txt',
#               './data/test/tel.txt',
#               './data/test/tit.txt']


train_files = ['./data/test/loc.txt',
              './data/test/name.txt',
              './data/test/org.txt',
              './data/test/tel.txt',
              './data/test/tit.txt']

test_files = ['./data/test/loc.txt',
              './data/test/name.txt',
              './data/test/org.txt',
              './data/test/tel.txt',
              './data/test/tit.txt']

words_json_file = './gen_data/words2id.json'
chars_json_file = './gen_data/char2id.json'


def build_vocab_words():
    vocab_words = set()
    for item in train_files:
        with open(item) as f:
            lines = f.readlines()
        for line in lines:
            words = line.split()
            for word in words:
                vocab_words.add(word)
    print("- done. {} tokens".format(len(vocab_words)))
    vocab_words.add(UNK)
    vocab_words.add(NUM)
    return vocab_words


def build_vocab_chars():
    vocab_chars = set()
    for item in train_files:
        with open(item) as f:
            lines = f.readlines()
        for line in lines:
            for item in line.strip().split():
                for c in item:
                    vocab_chars.add(c)
    print("- done. {} tokens".format(len(vocab_chars)))
    return vocab_chars


def save_dic_json(vocab, file_path):
    dic = {}
    for index, item in enumerate(vocab):
        dic[str(item)] = index
    with open(file_path, 'w') as json_file:
        json.dump(dic, json_file)


def load_dic_json(j_file):
    with open(j_file, 'r') as json_file:
        data = json.load(json_file)
    print("length is ", len(data))
    return data


def random_embedding(vocab, embedding_dim):
    """
    :param vocab:
    :param embedding_dim:
    :return:
    """
    embedding_mat = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_dim))
    embedding_mat = np.float32(embedding_mat) ## drawing random uniform samples from embedding within range
    return embedding_mat #represents output after one-hot matrix X embedding weight matrix = hidden layer


def main():
    vocab_words = build_vocab_words()
    vocab_chars = build_vocab_chars()
    save_dic_json(vocab_words, words_json_file)
    save_dic_json(vocab_chars, chars_json_file)


if __name__ == '__main__':
    main()
