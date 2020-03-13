import tensorflow as tf
import numpy as np
from build_data import random_embedding, load_dic_json
from build_data import words_json_file, chars_json_file

# the dimension of random words and char matrix
dim = 300


def main():
    print('loading words dict')
    words_dic = load_dic_json(words_json_file)
    print('loading chars dict')
    chars_dic = load_dic_json(chars_json_file)
    embeding_words = random_embedding(words_dic, dim)
    enmbeding_chars = random_embedding(chars_dic, dim)
    np.save('embeding_words', embeding_words)
    np.save('enmbeding_chars', enmbeding_chars)
    print('embeding data saved.')


if __name__ == '__main__':
    main()
