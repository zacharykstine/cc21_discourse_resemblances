"""
# Zachary K. Stine, 2021-01-20
#
# Identifies exemplar documents for each topic from specified LDA models and writes summary information about those documents
# to file. This makes it possible to read source text to contextualize the high-probability topic words in order to get about
# better sense of how a topic should be interpreted.
"""
import cons
import os
import csv
import gensim
import numpy as np
import math
import measures
import functions
import datetime

from scipy.special import xlogy


def write_exemplars_metadata(exemplar_documents, corpus_name, model_dir):
    """
    Writes relevant data for topic exemplars to file.
    """
    subreddit_metadata_dict = {}
    sub_list = set()
    for topic_index in exemplar_documents.keys():
        sub_list.update(tple[0] for tple in exemplar_documents[topic_index])

    for subreddit in sub_list:
        metadata_fpath = os.path.join(cons.corpora_dir, subreddit, corpus_name, corpus_name + '_metadata.csv')
        subreddit_metadata = functions.read_corpus_metadata(metadata_fpath)
        subreddit_metadata_dict[subreddit] = subreddit_metadata

    for topic_index in exemplar_documents.keys():

        # Get the topic directory:
        topic_dir = os.path.join(model_dir, 'topic_' + f'{topic_index:02d}')
        exemplars_fpath = os.path.join(topic_dir, 'topic_' + f'{topic_index:02d}' + '_exemplar_list.csv')

        with open(exemplars_fpath, 'w', encoding='utf-8', newline='') as outfile:
            fwriter = csv.writer(outfile)

            fwriter.writerow(['rank',
                              'subreddit',
                              'topic_probability',
                              'submission_date',
                              'submission_id',
                              'corpus_index',
                              'submission_score',
                              'submission_path',
                              'submission_url',
                              'submission_url_old'])

            topic_exemplar_documents = exemplar_documents[topic_index]

            for exemp_index, (subreddit, doc_index, topic_prob) in enumerate(topic_exemplar_documents):

                doc_metadata = subreddit_metadata_dict[subreddit][doc_index]

                assert doc_index == int(doc_metadata['corpus_index'])

                rank = exemp_index + 1
                doc_metadata['topic_probability'] = topic_prob

                fwriter.writerow([rank,
                                  subreddit,
                                  topic_prob,
                                  doc_metadata['submission_date'],
                                  doc_metadata['submission_id'],
                                  doc_metadata['corpus_index'],
                                  doc_metadata['submission_score'],
                                  doc_metadata['submission_path'],
                                  doc_metadata['submission_url'],
                                  doc_metadata['submission_url_old']])


def get_topic_exemplars(training_corpus_type, sample_name, corpus_name, k):
    """
	Identifies exemplar documents of each topic.
	"""

    num_exemplars = 150

    # Read in the LDA model.
    model_dir = os.path.join(cons.lda_dir, training_corpus_type, sample_name, sample_name + '-' + str(k))
    lda_model = gensim.models.LdaModel.load(os.path.join(model_dir, 'model_files'))

    # Each topic index is a key with its corresponding value a list of tuples. Each tuple describes an exemplar document
    # for the topic and consists of (subreddit, doc_index, topic_prob, max_topic).
    global_exemplars = {t_index: [] for t_index in range(k)}

    for subreddit in cons.subreddit_list:

        #
        # 1) Determine which subreddit-specific corpus indices to use.
        #

        metadata_fpath = os.path.join(cons.corpora_dir, subreddit, corpus_name, corpus_name + '_metadata.csv')
        subreddit_metadata = functions.read_corpus_metadata(metadata_fpath)

        if corpus_name == 'normal_corpus':
            corpus_indices = [int(doc_data['corpus_index']) for doc_data in subreddit_metadata]

            if subreddit in ['christianity', 'islam']:
                # sample 75k from corpus_indices
                corpus_indices = np.random.choice(corpus_indices, 75000, replace=False)

        elif corpus_name == 'reduced_corpus_001':
            corpus_indices = [int(doc_data['corpus_index']) for doc_data in subreddit_metadata if float(doc_data['reduced_doc_length']) >= 50.0]

            if subreddit == 'christianity':
                # sample 75k indices from corpus_indices
                corpus_indices = np.random.choice(corpus_indices, 75000, replace=False)

        #
        # 2) Read in the corpus. Iterate through documents and add topic probabilities to a subreddit-
        # specific list of exemplar candidates. After iterating through all docs for a subreddit, combine this list
        # with the global exemplar list and then filter down to the top documents.
        #

        # Each topic index is a key with its corresponding value a list of tuples. Each tuple describes an exemplar
        # document for the topic and consists of (doc_index, topic_prob).
        subreddit_exemplars = {t_index: [] for t_index in range(k)}

        corpus_fpath = os.path.join(cons.corpora_dir, subreddit, corpus_name, corpus_name)
        corpus = gensim.corpora.MmCorpus(corpus_fpath)

        for doc_index in corpus_indices:

            # Draw theta for the current document.
            doc_theta = lda_model[corpus[doc_index]]

            # Add the topic probability, val, along with document index to the list of possible subreddit exemplars.
            for (topic, val) in doc_theta:
                if val >= 0.1:
                    subreddit_exemplars[topic].append((doc_index, val))

        # After iterating, sort each list of subreddit exemplars and then append them to the global lists.
        for topic in subreddit_exemplars.keys():
            final_subreddit_exemplars = sorted(subreddit_exemplars[topic], key=lambda x: x[1], reverse=True)[:num_exemplars]
            potential_exemplars = global_exemplars[topic] + [(subreddit, doc_i, t_prob) for (doc_i, t_prob) in final_subreddit_exemplars]
            global_exemplars[topic] = sorted(potential_exemplars, key=lambda x: x[2], reverse=True)[:num_exemplars]

    # At this point, we should have the final exemplars for all topics across all subreddits. We can now write
    # the exemplar files.
    write_exemplars_metadata(global_exemplars, corpus_name, model_dir)


if __name__ == '__main__':
    subreddit_list = cons.subreddit_list
    get_topic_exemplars('combined_subs_normal', 'combo_1', 'normal_corpus', 250)
	get_topic_exemplars('combined_subs_reduced_001', 'combo_3', 'reduced_corpus_001', 250)

