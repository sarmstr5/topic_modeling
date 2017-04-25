import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio
import sklearn as sk
import GetTime as gt
from sklearn.decomposition import LatentDirichletAllocation
import sys, os, glob, re



def convert_mat_to_npz(mat_file):
    fn = 'mnist/mnist_'
    for key, value in mnist_dataset.items():
        arr = np.array(value)
        np.save(fn+key, arr)

def plot_one_img(img_row):
    imgplot = plt.imshow(np.reshape(img_row, (28,28)), cmap=plt.cm.gray)
    plt.axis('off')

def run_sk_lda(X, vocab, fn, n_topics, learning_method, n_jobs, max_iter, k_top_words):
    lda = LatentDirichletAllocation(n_topics=n_topics, doc_topic_prior=None, topic_word_prior=None,
                                    learning_method=learning_method, learning_decay=0.7, learning_offset=10.0,
                                    max_iter=max_iter, batch_size=128, evaluate_every=-1, total_samples=1000000.0,
                                    perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=n_jobs,
                                    verbose=0, random_state=None)
    document_topics = lda.fit_transform(X)
    topic_terms = lda.components_
    topic_words = get_topic_vocab(topic_terms, document_topics, vocab, k_top_words)
    topic_word_fn = 'data/topic_output/{}_topic_terms_ntopics{}_kWords{}_maxIter{}.csv'.format(fn, n_topics,
                                                                                               k_top_words, max_iter)
    write_to_disk(topic_words, topic_word_fn)

def main():
    #---------model params---------------#
    n_topics = 7
    learning_method = 'online' # due to size of matrix
    n_jobs = 1 # parallel run
    max_iter = 10 # default is 10
    #-----------------------------------#
    k_top_words = 20
    dataset_type = 'csc'
    full_run = False

    # fn = 'mnist/mnist_all/mat'
    # mnist_data = sio.loadmat(fn)
    if full_run:
        pass
    else:
        train_fn = 'mnist/mnist_train0.npy'
        train_X = np.load(train_fn)
        run_sk_lda(train_X)


if __name__ == '__main__':
    main()
