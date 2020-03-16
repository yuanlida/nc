import numpy as np

# # example 1:
# data1 = [[1, 2, 1], [1, 3, 1], [1, 4, 1]]
# arr2 = np.array(data1)
# arr3 = np.asarray(data1, dtype=np.int64)
# data1[1][1] = 2
# print('data1:\n', data1)
# print('arr2:\n', arr2)
# print('arr3:\n', arr3)
# print('arr4:\n',arr3[:,-1])

a = '123123'
print(len(a))

char_ids = []

x = 'abc edf lad bcd .'
setence = [item for item in x.split(" ")]
print(setence)
char_ids = []
for word in setence:
    ids = [0 for item in word]
    char_ids.append(ids)
temp_ids = []

