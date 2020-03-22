from tf_lite_invoke import TFLiteModel
import os

error_file = './model/Bi-LSTM/error/'

test_files = ['./data/test/loc.txt',
              './data/test/name.txt',
              './data/test/org.txt',
              './data/test/tel.txt',
              './data/test/tit.txt']

if __name__ == '__main__':
    if not os.path.exists(error_file):
        os.mkdir(error_file)
    lite_model = TFLiteModel(None)
    for file_name in test_files:
        filepath, tmpfilename = os.path.split(file_name)
        shotname, extension = os.path.splitext(tmpfilename)
        # print(shotname)
        with open(error_file + shotname + '.txt', 'w') as of:
            with open(file_name) as f:
                for line in f.readlines():
                    lite_model.set_sentence(line)
                    label = lite_model.analyze()
                    if label != shotname:
                        print(line)
                        of.write(line.strip())
                        of.write('\t')
                        of.write(label)
                        of.write('\n')
