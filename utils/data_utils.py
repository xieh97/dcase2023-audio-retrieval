import os
import pickle
from ast import literal_eval

import h5py
import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

stopwords = nltk.corpus.stopwords.words("english")


class Vocabulary(object):

    def __init__(self):
        self.key2vec = {}
        self.key2id = {}
        self.id = 0

    def add_key(self, key, key_vector):
        if key not in self.key2id:
            self.key2vec[key] = key_vector
            self.key2id[key] = self.id
            self.id += 1

    def __call__(self, key):
        return self.key2id[key]

    def __len__(self):
        return len(self.key2id)


class AudioTextDataset(Dataset):

    def __init__(self, **kwargs):
        self.audio_data = kwargs["audio_data"]
        self.text_data = kwargs["text_data"]
        self.text_vocab = kwargs["text_vocab"]
        self.text_level = kwargs["text_level"]

    def __getitem__(self, index):
        item = self.text_data.iloc[index]

        audio_vec = torch.as_tensor(self.audio_data[item["fid"]][()])

        text_vec = None

        if self.text_level == "word":
            text_vec = torch.as_tensor([self.text_vocab(key) for key in item["tokens"] if key not in stopwords])

        elif self.text_level == "sentence":
            text_vec = torch.as_tensor([self.text_vocab(item["tid"])])

        return item, audio_vec, text_vec

    def __len__(self):
        return len(self.text_data)


def collate_fn(batch):
    item_batch, audio_batch, text_batch = [], [], []

    for i, a, t in batch:  # list of (item, audio_vec, text_vec)
        item_batch.append(i)
        audio_batch.append(a)
        text_batch.append(t)

    audio_batch = padding(audio_batch, dtype=torch.float)
    text_batch = padding(text_batch, dtype=torch.long)

    return item_batch, audio_batch, text_batch


def padding(tensors, dtype=torch.float):
    dims = np.array([t.shape for t in tensors])
    max_dims = np.max(dims, axis=0)

    paddings = max_dims - dims
    padded_tensors = []

    for t, pad in zip(tensors, paddings):
        pad_tuple = ()

        for p in pad[::-1]:
            pad_tuple = pad_tuple + (0, p)

        padded_tensors.append(F.pad(t, pad_tuple))

    padded_tensors = torch.stack(padded_tensors)

    return padded_tensors.to(dtype)


def load_data(conf):
    # Load audio data
    audio_fpath = os.path.join(conf["dataset"], conf["audio_data"])
    audio_data = h5py.File(audio_fpath, "r")
    print("Load", audio_fpath)

    # Load text data
    text_fpath = os.path.join(conf["dataset"], conf["text_data"])
    text_data = pd.read_csv(text_fpath, converters={"tokens": literal_eval})
    print("Load", text_fpath)

    # Load prior embeddings (e.g., word2vec, BERT, and RoBERTa)
    embed_fpath = os.path.join(conf["dataset"], conf["text_embeds"])
    with open(embed_fpath, "rb") as stream:
        text_embeds = pickle.load(stream)
    print("Load", embed_fpath)

    text_level = conf["text_level"]

    # Build embedding vocabulary
    text_vocab = Vocabulary()
    for key in text_embeds:
        if len(text_vocab) == 0:
            text_vocab.add_key("<pad>", np.zeros_like(text_embeds[key]))
        text_vocab.add_key(key, text_embeds[key])

    # Enclose data
    kwargs = {"audio_data": audio_data, "text_data": text_data, "text_vocab": text_vocab, "text_level": text_level}
    dataset = AudioTextDataset(**kwargs)

    return dataset
