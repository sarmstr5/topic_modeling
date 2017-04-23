import numpy as np
import pandas as pd
import glob, os, sys
import scipy.sparse as sps
import sklearn as sk
from sklearn.decomposition import LatentDirichletAllocation


def load_dataset(dataset_type, fn):
    npy_file = np.load(fn)
    if dataset_type == 'csc':
        return sps.csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])
    else:
        return sps.csr_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])

def load_datasets(dataset_type):
    files = glob.glob('*'+dataset_type+'.npz')
    for fn in files:
        yield load_dataset(dataset_type, fn)

def load_vocabs():
    files = glob.glob('/data/vocab*')
    print(files)
    for fn in files:
        yield load_vocab(fn)

def load_vocab(fn):
    nips_voc = []
    with open(fn) as file:
        for row in file:
            nips_voc.append(row.strip())
    return np.array(nips_voc)

def create_model(n, learning_method='online', n_jobs=-2, max_iter=10):
    lda_model = LatentDirichletAllocation(n_topics=n, doc_topic_prior=None, topic_word_prior=None,
                                          learning_method=learning_method, learning_decay=0.7, learning_offset=10.0,
                                          max_iter=max_iter, batch_size=128, evaluate_every=-1, total_samples=1000000.0,
                                          perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=n_jobs,
                                          verbose=0, random_state=None)
    return lda_model

def main():
    print(glob.glob("*"))
    # os.chdir("CS674/HW4_topic_modeling/data")
    # glob.glob("*")
    #---------model params---------------#
    n_topics = 5
    learning_method = 'online' # due to size of matrix
    n_jobs = -2 # parallel run
    max_iter = 5 # default is 10
    #-----------------------------------#

    # easy run
    fn = 'data/docword.nips.txt_csc.npz'
    fn_vocab = 'data/vocab.nips.txt'
    dataset_type = 'csc'
    csc = load_dataset(dataset_type, fn)
    vocab = load_vocab(fn_vocab)
    print(vocab)


if __name__ == "__main__":
    main()