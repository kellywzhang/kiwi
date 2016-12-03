import numpy as np
import re
import os
import pickle

# Get filename and filepath
dimension = str(100)
filename = "glove.6B."+dimension+"d.txt"
filepath = os.path.join(os.path.abspath(os.path.curdir), "glove.6B", filename)

vocab_dict = {}
embeddingfile = open(filepath, "r")
count = 0

# Get dictionary of all words in embedding file
for line in embeddingfile:
    values = re.split('\s+', line)
    vocab_dict[values[0]] = count
    count += 1
    if count % 1000000 == 0:
        print(count)
embeddingfile.close()

pickle.dump(vocab_dict, open("glove_vocab_dict.p", "wb"))
print(len(vocab_dict)) #400,000 words total


# embeddings = np.expand_dims(np.zeros(100), axis=0)
# for line in embeddingfile:
#     values = re.split('\s+', line)
#     if len(values[-1]) == 0:
#         del values[-1]
#     vocab_dict[values[0]] = count
#     embeddings = np.concatenate((embeddings, np.expand_dims(np.array(values[1:]), axis=0)), axis=0)
#     count += 1
#     if count % 100 == 0:
#         print(count)
#
# pickle.dump(vocab_dict, open("glove_vocab_dict.p", "wb"))
# pickle.dump(embeddings[1:], open("glove_"+dimension+"_embedding.p", "wb"))
#
# print(embeddings)
# print(embeddings[1:].shape)
# print(len(vocab_dict))

"""
line = embeddingfile.readline()


print(line)
print(np.array(line[1:]))
"""
