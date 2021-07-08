"""
# Zachary K. Stine, 2020-12-28
#
# Train several LDA models on the combined documents from all subreddits using the original, unmodified corpus (aside from
# some basic preprocessing).
"""
import os
import csv
import cons
import datetime
import random
import gensim
import numpy as np
import logging


class NormalCorpusFromIndices(object):

    def __init__(self, subreddit_list, corpus_sub_indices):
        self.subreddit_list = subreddit_list
        self.corpus_sub_indices = corpus_sub_indices
        self.corpus_dict = {}

        for subreddit in subreddit_list:
            subreddit_corpus_fpath = os.path.join(cons.corpora_dir, subreddit, 'normal_corpus', 'normal_corpus')
            subreddit_corpus = gensim.corpora.MmCorpus(subreddit_corpus_fpath)
            self.corpus_dict[subreddit] = subreddit_corpus

    def __iter__(self):
        for (subreddit, doc_index) in self.corpus_sub_indices:
            yield self.corpus_dict[subreddit][doc_index]


def build_subreddit_document_list(subreddit_list, sample_subs, sample_size):
    """
    Function returns subreddit names and document indices for documents that will be used in model training. For most subreddits,
	this is every document, but for those designated in sample_subs it will be a random sample with size set by
	sample_size. Document indices are randomly shuffled for model training.
    """
    sub_index_list = []

    for subreddit in subreddit_list:
        subreddit_corpus_fpath = os.path.join(cons.corpora_dir, subreddit, 'normal_corpus', 'normal_corpus')
        subreddit_corpus = gensim.corpora.MmCorpus(subreddit_corpus_fpath)

        if subreddit in sample_subs:
            random_indices = np.random.choice(len(subreddit_corpus), sample_size, replace=False)
            sub_index_list += [(subreddit, rand_i) for rand_i in random_indices]

        else:
            sub_index_list += [(subreddit, doc_i) for doc_i in range(len(subreddit_corpus))]

    random.shuffle(sub_index_list)
    return sub_index_list


def write_top_words(model, model_dir, k, num_words=500):
    """
    Function writes CSV files with the highest-probability words and their probabilities for each topic.
    """
    for topic_index in range(k):

        # Make directory to store all topic-specific data
        topic_dir = os.path.join(model_dir, 'topic_' + f'{topic_index:02d}')
        cons.makedir(topic_dir)

        # Get the highest probability terms from this topic
        topic_words = model.show_topic(topic_index, num_words)

        # Path to CSV file that will hold the topic words and probabilities:
        wordlist_fname = os.path.join(topic_dir, 'topic_' + f'{topic_index:02d}' + '_word_list.csv')

        # Write top terms to csv file.
        with open(wordlist_fname, 'w', newline='', encoding='utf-8') as ofile:
            f_writer = csv.writer(ofile)

            f_writer.writerow(['word', 'probability'])

            for (word, prob) in topic_words:
                f_writer.writerow([word, prob])


if __name__ == '__main__':

    subreddit_list = cons.subreddit_list
    k_list = [20, 100, 250]
    models_per_k = 6
    downsample_subs = ['christianity', 'islam']
    downsample_size = 75000

    # LDA arguments.
    passes = 5
    iterations = 200
    chunks = 2000
    eval_every = None
    min_prob = 1e-10

    dictionary = gensim.corpora.Dictionary.load(os.path.join(cons.corpora_dir, 'dictionary.dict'))

    combo_models_dir = cons.makedir(os.path.join(cons.lda_dir, 'combined_subs_normal'))

    # Set up logger for all of the combined, normal corpus LDA models.
    log_fpath = os.path.join(combo_models_dir, 'normal_combo_models.log')
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(log_fpath, 'w', 'utf-8')])

    # Since we are going to be sampling documents from a few subreddits, we will train models with the same # of topics
    # (k) on a few different samples to make sure results are not overly specific to a particular sample. The number of
    # samples is defined by models_per_k.
    for model_pass in range(models_per_k):
        sample_num = model_pass + 1
        sample_random_seed = np.random.choice(1000)

        current_time = datetime.datetime.now()
        print(str(current_time) + ' : starting combined corpus sample ' + str(sample_num))
        print('--------------------')

        sample_dir = cons.makedir(os.path.join(combo_models_dir, 'combo_' + str(sample_num)))

        # Create training corpus by sampling documents from the designated subreddits and using all from the others.
        # Then, randomize the order of the resulting documents for training.
        corpus_sub_indices = build_subreddit_document_list(subreddit_list, downsample_subs, downsample_size)

        combo_corpus = NormalCorpusFromIndices(subreddit_list, corpus_sub_indices)
        

        for k in k_list:
            print(str(datetime.datetime.now()) + ' starting model with ' + str(k) + ' topics.')

            model_name = 'combo_' + str(sample_num) + '-' + str(k)
            print(model_name)

            # Create model directory
            model_dir = cons.makedir(os.path.join(sample_dir, model_name))

            # Train and save LDA model.
            logging.info('START ' + model_name)
            lda_model = gensim.models.LdaModel(combo_corpus, num_topics=k, id2word=dictionary, passes=passes,
                                               iterations=iterations, chunksize=chunks,
                                               eval_every=eval_every, minimum_probability=min_prob,
                                               alpha='auto', random_state=sample_random_seed)
            logging.info('END ' + model_name)
            lda_model.save(os.path.join(model_dir, 'model_files'))

            # Write summary information about file.
            model_description = {'k': k,
                                 'random_state': sample_random_seed,
                                 'passes': passes,
                                 'iterations': iterations,
                                 'chunksize': chunks,
                                 'eval_every': eval_every}
            with open(os.path.join(model_dir, 'model_description.txt'), 'w') as outfile:
                outfile.write(str(model_description))

            # Write highest probability words to file for reference.
            write_top_words(lda_model, model_dir, k, num_words=100)

        print('-------------\n')
