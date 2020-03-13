import json

UNK = "$UNK$"
NUM = "$NUM$"
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

words_json = './gen_data/words2id.json'
chars_json = './gen_data/char2id.json'


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
    return data


def main():
    vocab_words = build_vocab_words()
    vocab_chars = build_vocab_chars()
    save_dic_json(vocab_words, words_json)
    save_dic_json(vocab_chars, chars_json)


if __name__ == '__main__':
    main()
