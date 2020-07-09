import nltk
words = set(nltk.corpus.words.words())

with open ('/Users/macos/thunder/ner-tf/cls_data/extra/title_skill.txt', 'r') as r,\
    open ('/Users/macos/thunder/ner-tf/cls_data/extra/eng_tit.txt', 'w') as w:
    body = r.readlines()
    for sent in body:
        word = " ".join(w for w in nltk.wordpunct_tokenize(sent) \
                 if w.lower() in words or not w.isalpha())
        w.writelines(word.title())
        w.writelines('\n')
# 'Io to the beach with my'