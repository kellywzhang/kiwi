import numpy as np
import re
import os
import pickle

dimension = str(100)
filename = "glove.6B."+dimension+"d.txt"
filepath = os.path.join(os.path.abspath(os.path.curdir), "glove.6B", filename)

vocab_dict = {}
embeddingfile = open(filepath, "r")
count = 0

embeddings = np.expand_dims(np.zeros(100), axis=0)
print(embeddings)
print(np.concatenate((embeddings, embeddings), 0))

for line in embeddingfile:
    values = re.split('\s+', line)
    if len(values[-1]) == 0:
        del values[-1]
    vocab_dict[values[0]] = count
    embeddings = np.concatenate((embeddings, np.expand_dims(np.array(values[1:]), axis=0)), axis=0)
    count += 1
pickle.dump(vocab_dict, open("glove_vocab_dict.p", "wb"))
pickle.dump(embeddings[1:], open("glove_"+dimension+"_embedding.p", "wb"))

print(embeddings)
print(embeddings[1:].shape)
print(vocab_dict)

"""
line = embeddingfile.readline()


print(line)
print(np.array(line[1:]))
"""
