"""
Zachary Kimo Stine, 2021-01-19

Train LDA models on modified corpus. Closely follows the other LDA script, but with a few changes specific to the modified corpus.
"""

import cons
import os
import csv
import gensim
import numpy as np
import logging
import datetime
import random


class ReducedCorpusFromIndices(object):

    def __init__(self, subreddit_list, corpus_sub_indices):
        self.subreddit_list = subreddit_list
        self.corpus_sub_indices = corpus_sub_indices
        self.corpus_dict = {}

        for subreddit in subreddit_list:
            subreddit_corpus_fpath = os.path.join(cons.corpora_dir, subreddit, 'reduced_corpus_001', 'reduced_corpus_001')
            subreddit_corpus = gensim.corpora.MmCorpus(subreddit_corpus_fpath)
            self.corpus_dict[subreddit] = subreddit_corpus

    def __iter__(self):
        for (subreddit, doc_index) in self.corpus_sub_indices:
            yield self.corpus_dict[subreddit][doc_index]


def get_doc_len(bow_doc):
    """
    Returns document length (ie, number of tokens).
    """
    doc_len = sum([word_count for (word_id, word_count) in bow_doc])
    return doc_len


def build_subreddit_document_list_reduced(subreddit_list, sample_subs, sample_size, min_len=50.0):
    """
    Generates list of document indices to be included in the training corpus and randomly shuffles them.
    """
    sub_index_list = []

    for subreddit in subreddit_list:

        # Only consider including a document if its reduced length has at least 50 tokens.
        subreddit_reduced_metadata_path = os.path.join(cons.corpora_dir, subreddit, 'reduced_corpus_001', 'reduced_corpus_001_metadata.csv')
        subreddit_reduced_metadata = read_corpus_metadata(subreddit_reduced_metadata_path)
        subreddit_allowed_corpus_indices = [int(doc_data['corpus_index']) for doc_data in subreddit_reduced_metadata if float(doc_data['reduced_doc_length']) >= min_len]
        print(subreddit + ' has ' + str(len(subreddit_allowed_corpus_indices)) + ' docs that are >= the minimum length.')
        
		# Sample smaller # of documents (set by sample_size) for subreddits included in sample_subs list. Otherwise, keep all allowable documents.
        if subreddit in sample_subs:
            random_indices = np.random.choice(subreddit_allowed_corpus_indices, sample_size, replace=False)
            sub_index_list += [(subreddit, rand_i) for rand_i in random_indices]
       
        else:
            sub_index_list += [(subreddit, doc_i) for doc_i in subreddit_allowed_corpus_indices]
    
	# Randomize document order before model training.
    random.shuffle(sub_index_list)
    return sub_index_list


def write_top_words(model, model_dir, k, num_words=500):
    """
    Writes highest-probability words and their probabilities within each topic to file.
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


def read_corpus_metadata(fpath):
    """
	Reads in corpus metadata.
	"""
    data_list = []

    with open(fpath, 'r', encoding='utf-8') as infile:
        freader = csv.DictReader(infile)

        for row_dict in freader:
            data_list.append(row_dict)

    return data_list


if __name__ == '__main__':

    subreddit_list = cons.subreddit_list
    k_list = [20, 100, 250]
    models_per_k = 10
    downsample_subs = ['christianity']
    downsample_size = 75000  # 75,000

    # LDA arguments.
    passes = 5
    iterations = 200
    chunks = 2000
    eval_every = None
    min_prob = 1e-10

    dictionary = gensim.corpora.Dictionary.load(os.path.join(cons.corpora_dir, 'dictionary.dict'))

    combo_models_dir = cons.makedir(os.path.join(cons.lda_dir, 'combined_subs_reduced_001'))

    # Set up logger for all of the combined, normal corpus LDA models.
    log_fpath = os.path.join(combo_models_dir, 'reduced_001_combo_models.log')
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
        corpus_sub_indices = build_subreddit_document_list_reduced(subreddit_list,
                                                                   downsample_subs,
                                                                   downsample_size,
                                                                   min_len=50.0)
        print('sample ' + str(sample_num) + ' has ' + str(len(corpus_sub_indices)) + ' documents.')

        combo_corpus = ReducedCorpusFromIndices(subreddit_list, corpus_sub_indices)

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
