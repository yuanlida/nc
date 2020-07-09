from tqdm import tqdm

in_dir = "/Users/macos/Desktop/Edison-ai/sig-extract/enron_label/body_two.txt"
out_dir = "/Users/macos/thunder/ner-tf/process_data/enron_sig.txt"
file_dir = "/Users/macos/Desktop/Edison-ai/sig-extract/enron_test2/P/"

def main():
# # extracting single text file
#     with open(in_dir, 'r') as r, open(out_dir, 'w') as w:
#         for line in r.readlines():
#             if line.startswith('#sig#'):
#                 w.writelines(line)
#                 w.writelines('\n')
#%%
# extracting multiple text files
    doc = []
    for i in range(5000):
        try:
            with open((file_dir + str(i) + "_body"), 'r') as r:
                for line in r.readlines():
                    if line.startswith('#sig#'):
                        doc.append(line)
        except:
            print(i, "does not exist")
            pass
    with open(out_dir, 'w') as w:
        for ele in doc:
            w.writelines(ele)
            w.writelines('\n')

if __name__ == '__main__':
    main()