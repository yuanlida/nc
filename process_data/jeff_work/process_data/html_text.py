import html2text
import codecs
from bs4 import BeautifulSoup
from process_data.helper import get_sig

input_file = "/Users/macos/Downloads/0a1a8608daa2556bbd2ada3efc780375.html"
input_file2= "/Users/macos/Downloads/0a1b9b14af1fe9a98d917a507dbb17ea.html"
input_file3 = "/Users/macos/Desktop/Edison-ai/data/bodis/0a0b4cee82cc19f0fe3172f5ae512b70.html"
output_txt = "./email.txt"
file = codecs.open(input_file3, "r", "utf-8")


# print(file.read())
html = file.read()
# print('## HI' ,html)
sig_lines = get_sig(html)
with open(output_txt, 'w') as w:
    # w.writelines(sig_lines)
    for line in sig_lines:
        w.writelines(line)
        w.writelines('\n')

# print(html)

## html2text method
# links = [link for link in html if link.startswith("<div class=\"zd-comment\"") and link.endswith(("""</div>
#                                 </div>"""))]
# print(links)
# result = html2text.html2text(file.read())
# split_result = result.split("---")
# # print(split_result)
#
# ## neeed extract <div class="zd-comment"
# with open(output_txt, 'w') as w:
#     for n, body in enumerate(split_result):
#         w.writelines("#######" + str(n))
#         w.writelines('\n')
#         w.writelines(body)
#         w.writelines('\n')


