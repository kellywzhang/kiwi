import numpy as np
import os
import pickle
from collections import Counter

def load_data_and_labels(filename, vocab_size=150, train=True):
    """
    Loads data from files, splits the data into chars and generates labels.
    """
    file_path = os.path.join(os.path.abspath(os.path.curdir), "data", filename)
    f = open(file_path, 'r', encoding = "ISO-8859-1")
    data = list(f.readlines())
    data = [s.strip().split() for s in data]

    string_labels = [d[0] for d in data]
    # Replace spaces with "^"
    proper_nouns_strings = ["^".join(d[1:]) for d in data]

    # LABEL Generation
    labels = []
    for l in string_labels:
        if l == "drug":
            labels.append(0)
        elif l == "person":
            labels.append(1)
        elif l == "place":
            labels.append(2)
        elif l == "movie":
            labels.append(3)
        elif l == "company":
            labels.append(4)
        else:
            print("Parsing error: no label")
            break

    # VOCABULARY (CHARACTER LEVEL)
    if train:
        char_count = Counter()
        for string in proper_nouns_strings:
            for char in list(string):
                char_count[char] += 1

        ls = char_count.most_common(vocab_size)

        # leave 0 to UNK = ~
        vocab_dict = {char[0]: index + 1 for (index, char) in enumerate(ls)}
        vocab_dict["~"] = 0
        pickle.dump(vocab_dict, open("vocabulary_dict.p", "wb"))

    else:
        vocab_dict = pickle.load(open("vocabulary_dict.p", "rb"))

    char_lists = [list(string) for string in proper_nouns_strings]
    vectorized = []
    seq_lens = []
    for char_list in char_lists:
        new_char_list = []
        for char in char_list:
            try:
                new_char_list.append(vocab_dict[char])
            except KeyError:
                new_char_list.append(vocab_dict["~"])
        seq_lens.append(len(char_list))
        vectorized.append(new_char_list)

    return (vectorized, np.array(labels), np.array(seq_lens), vocab_dict)

def batch_iter(data, batch_size, num_epochs=1, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    np.random.seed(0)
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size)
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index] # returns generator (one-use iterator)

def pad(sequences, seq_lens):
    pad_len = max(seq_lens)
    new_sequences = []
    for seq in sequences:
        while len(seq) < pad_len:
            seq.append(0)
        new_sequences.append(seq)
    return np.array(new_sequences)
