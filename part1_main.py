import glob, os, sys, csv
import numpy as np
import pandas as pd
import scipy.sparse as sps
import sklearn as sk
from sklearn.decomposition import LatentDirichletAllocation
import GetTime as gt


def load_dataset(dataset_type, fn):
    npy_file = np.load(fn)
    if dataset_type == 'csc':
        return sps.csc_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])
    else:
        return sps.csr_matrix((npy_file['data'], npy_file['indices'], npy_file['indptr']), shape=npy_file['shape'])

def load_datasets(dataset_type):
    fns = glob.glob('data/*'+dataset_type+'.npz')
    for fn in fns:
        yield load_dataset(dataset_type, fn)

def load_vocabs():
    files = glob.glob('data/vocab*')
    print(files)
    for fn in files:
        yield load_vocab(fn)

def load_vocab(fn):
    nips_voc = []
    with open(fn) as file:
        for row in file:
            nips_voc.append(row.strip())
    return np.array(nips_voc)

def get_topic_vocab(topic_terms, document_topics, vocab, k):
    top_topic_inds = []
    top_topic_terms = []
    for top_num, topic in enumerate(topic_terms):
        topic_sorted = np.argpartition(topic, -k)[-k:]
        top_topic_terms.append(vocab[topic_sorted])
        top_topic_inds.append(topic_sorted)

    return top_topic_terms

def write_to_disk(file, fn):
    with open(fn, 'w') as results:
        for row in file:
            print(row)
            results.write('{}\n'.format(' '.join(row)))

def run_sk_lda(csc, vocab, fn, n_topics, learning_method, n_jobs, max_iter, k_top_words):
    lda = LatentDirichletAllocation(n_topics=n_topics, doc_topic_prior=None, topic_word_prior=None,
                                          learning_method=learning_method, learning_decay=0.7, learning_offset=10.0,
                                          max_iter=max_iter, batch_size=128, evaluate_every=-1, total_samples=1000000.0,
                                          perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=n_jobs,
                                          verbose=0, random_state=None)
    document_topics = lda.fit_transform(csc)
    topic_terms = lda.components_
    topic_words = get_topic_vocab(topic_terms, document_topics, vocab, k_top_words)
    topic_word_fn = 'data/topic_output/{}_topic_terms_ntopics{}_kWords{}_maxIter{}.csv'.format(fn, n_topics,
                                                                                               k_top_words, max_iter)
    write_to_disk(topic_words, topic_word_fn)

def main():
    print(glob.glob("*"))
    # os.chdir("CS674/HW4_topic_modeling/data")
    # glob.glob("*")

    #---------model params---------------#
    n_topics = 7
    learning_method = 'online' # due to size of matrix
    n_jobs = 1 # parallel run
    max_iter = 10 # default is 10
    #-----------------------------------#
    k_top_words = 20
    dataset_type = 'csc'
    full_run = False

    if full_run:
        # loops through all documents
        s_time = gt.time()
        fns = glob.glob('data/vocab*')
        csc_files = load_datasets(dataset_type)
        vocab_files = load_vocabs()
        for csc, vocab, fn in zip(csc_files, vocab_files, fns):
            print('on fn:{}\t at time:{}'.format(fn, gt.time()))
            run_sk_lda(csc, vocab, fn, n_topics, learning_method, n_jobs, max_iter, k_top_words)
            print('results found at time:{}'.format(fn, gt.time()))

    else:
        # short run
        fn = 'data/docword.nytimes.txt_csc.npz'
        fn_data = 'data/'
        fn_vocab = 'vocab.nytimes.txt'
        csc = load_dataset(dataset_type, fn)
        vocab = load_vocab(fn_data+fn_vocab)
        for n_topics in range(2, 15,1):
            print('on:{}\t at time:{}'.format(n_topics, gt.time()))
            run_sk_lda(csc,vocab, fn_vocab, n_topics, learning_method, n_jobs, max_iter, k_top_words)


if __name__ == "__main__":
    main()