"""
# Zachary K. Stine, 2021-01-20
#
# Description: Given a topic model and a corpus, do the following:
#     1) Have the model draw a document-topic distribution (theta) for each document in the corpus.
#     2) Assign each token in each document to a topic.
# (see https://github.com/laurejt/authorless-tms/blob/master/estimated_topic_author_correlation.py from Thompson & Blei 2018).
#     3) Create average document-topic distribution for each subreddit.
#     4) Create topic distribution from token-topic assignments.
#
# Outputs:
#     1) For each subreddit, a numpy matrix in a .txt file.
#     2) For each subreddit, a .csv file with headings, <topic_index>, <count>, <proportion>
#     3) For each subreddit, an average document-topic distribution as a numpy array in a .txt file.
#     4) For each model, also make a summary file that has some number of words of each topic with any other info
#     helpful for understanding a topic (e.g., entropy over different subreddits).
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


def write_token_topic_freqs(fpath, topic_counts):
    """
    Writes counts and proportions for assignments of tokens to topics.
    """

    topic_props = topic_counts / np.sum(topic_counts)

    with open(fpath, 'w', newline='') as outfile:
        fwriter = csv.writer(outfile)
        fwriter.writerow(['topic', 'token_count', 'token_proportion'])

        for (t_index, (t_count, t_prop)) in enumerate(zip(topic_counts, topic_props)):
            fwriter.writerow([t_index, t_count, t_prop])


def get_subreddit_entropy(token_assignment_counts):
    """
    Get entropy of topics' spread over subreddits to see how subreddit-specific they are.
    """
    sub_entropy_counts = []
    sub_entropy_props = []

    sub_list = [sub for sub in token_assignment_counts.keys()]

    k = len(token_assignment_counts[sub_list[0]])

    for topic_index in range(k):
        topic_counts = []
        topic_props = []
        for subreddit in sub_list:
            subreddit_topic_count = token_assignment_counts[subreddit][topic_index]
            topic_counts.append(subreddit_topic_count)

            subreddit_topic_prop = subreddit_topic_count / float(sum(token_assignment_counts[subreddit]))
            topic_props.append(subreddit_topic_prop)

        topic_counts = np.array(topic_counts)
        topic_props = np.array(topic_props)

        topic_counts_dist = topic_counts / np.sum(topic_counts)
        topic_props_dist = topic_props / np.sum(topic_props)

        sub_entropy_counts.append(measures.entropy(topic_counts_dist))
        sub_entropy_props.append(measures.entropy(topic_props_dist))

    return sub_entropy_counts, sub_entropy_props


def get_subreddits_w_max_topics(token_assignment_counts):
    """
    For each topic, return which subreddit had the most tokens assigned to the topic. Do this based on count of tokens
    as well as on which subreddit had the highest proportion of its tokens assigned to the topic.
    """
    max_topic_counts = []
    max_topic_props = []

    sub_list = [sub for sub in token_assignment_counts.keys()]

    k = len(token_assignment_counts[sub_list[0]])

    for topic_index in range(k):
        sub_topic_counts = []
        sub_topic_props = []

        for subreddit in sub_list:

            # Number of tokens from subreddit assigned to topic.
            subreddit_topic_count = token_assignment_counts[subreddit][topic_index]

            # Count of all tokens from the subreddit (sums over how many tokens from the subreddit were assigned to each
            # topic).
            subreddit_count = sum(token_assignment_counts[subreddit])

            subreddit_topic_prop = subreddit_topic_count / subreddit_count

            sub_topic_counts.append((subreddit, subreddit_topic_count))
            sub_topic_props.append((subreddit, subreddit_topic_prop))

        # Sort the tuples of (subreddit, topic count) from highest to lowest topic counts. Then take the top 3. Do the
        # same for proportions.
        top_3_counts = sorted(sub_topic_counts, key=lambda x: x[1], reverse=True)[:3]
        top_3_props = sorted(sub_topic_props, key=lambda x: x[1], reverse=True)[:3]

        max_topic_counts.append(top_3_counts)
        max_topic_props.append(top_3_props)

    return max_topic_counts, max_topic_props


def write_topics_summary_file(fpath,
                              sub_entropy_counts, sub_entropy_props, topic_entropies, max_sub_count, max_sub_prop,
                              lda_model,
                              k,
                              num_words=20):
    """
    Write summary information for each topic in the model.
    """

    alpha = lda_model.alpha
    norm_alpha = alpha / np.sum(alpha)

    with open(fpath, 'w', encoding='utf-8', newline='') as outfile:
        fwriter = csv.writer(outfile)

        fwriter.writerow(['topic_index',
                          'alpha_posterior',
                          'word_entropy', 'subreddit_entropy_n', 'subreddit_entropy_p',
                          '1st_subreddit_topic_count', '2nd_subreddit_topic_count', '3rd_subreddit_topic_count',
                          '1st_subreddit_topic_prop', '2nd_subreddit_topic_prop', '3rd_subreddit_topic_prop']
                         + ['word_' + str(i + 1) for i in range(num_words)])

        for t_i in range(k):
            topic_words = lda_model.show_topic(t_i, num_words)
            fwriter.writerow([t_i,
                              norm_alpha[t_i],
                              topic_entropies[t_i], sub_entropy_counts[t_i], sub_entropy_props[t_i],
                              max_sub_count[t_i][0], max_sub_count[t_i][1], max_sub_count[t_i][2],
                              max_sub_prop[t_i][0], max_sub_prop[t_i][1], max_sub_prop[t_i][2]]
                             + [w + ' (' + str(round(wp, 4)) + ')' for (w, wp) in topic_words])


def initialize_model_counters(model_info, subreddit_list):
    """
	Creates and returns a dictionary for each LDA model with relevant model data and initialized places to store 
	counts for each model.
	"""
    model_counts_dict = {}

    for training_corpus_type, sample_name, corpus_name, k_list in model_info:
        for k in k_list:
            model_name = sample_name + '-' + str(k)
            model_dir = os.path.join(cons.lda_dir, training_corpus_type, sample_name, model_name)
            model_tdists_dir = cons.makedir(os.path.join(model_dir, 'topic_analysis'))
            model_path = os.path.join(model_dir, 'model_files')

            lda_model = gensim.models.LdaModel.load(model_path)

            # Get model phi as a topic-word matrix where each row is the word distribution that constitutes the topic.
            phis = lda_model.get_topics()
            # Thompson & Blei (2018) do this, but I think phis is already normalized when calling .get_topics().
            phis = phis / phis.sum(axis=1, keepdims=True)
            # This is from Thompson & Blei (2018). Gives log of each phi value, but if phi value is zero, keeps it zero.
            nz_phis = phis > 0
            log_phis = xlogy(nz_phis, phis)

            model_dict = {'tdists_dir': model_tdists_dir,
                          'model': lda_model,
                          'nz_phis': nz_phis,
                          'log_phis': log_phis,
                          'token_topic_counts': {subreddit: np.zeros(k) for subreddit in subreddit_list},
                          'doc_topic_sums': {subreddit: np.zeros(k) for subreddit in subreddit_list},
                          'doc_topic_tokens_sums': {subreddit: np.zeros(k) for subreddit in subreddit_list},
                          'doc_counts': {subreddit: 0 for subreddit in subreddit_list}}

            model_counts_dict[(training_corpus_type, sample_name, corpus_name, k)] = model_dict

            print(training_corpus_type + '\\' + sample_name + '-' + str(k))
    return model_counts_dict


def get_topic_distributions(model_info, corpus_name, subreddit_list):
    """
    Function probabilistically assigns each token to a topic from specified models and writes assignment statistics to file.
    """
	
    # initialize where topic counts will be stored for each model indicated in model_info
    model_dict = initialize_model_counters(model_info, subreddit_list)
    print()

    # iterate through each subreddit, each of its documents, and each word type in its documents to get counts.
    for subreddit in subreddit_list:

        current_time = datetime.datetime.now()
        print(str(current_time) + ' : starting ' + subreddit)
        print('--------------------')

        corpus_fpath = os.path.join(cons.corpora_dir, subreddit, corpus_name, corpus_name)
        corpus_metadata_fpath = os.path.join(cons.corpora_dir, subreddit, corpus_name, corpus_name + '_metadata.csv')
        corpus = gensim.corpora.MmCorpus(corpus_fpath)

        for doc in corpus:
            if len(doc) < 25:
                continue

            # For each model, get theta for the document.
            model_theta_dict = {}
            for model_key in model_dict.keys():
                doc_dist_gensim = model_dict[model_key]['model'][doc]
                k = model_key[3]
                doc_dist_numpy = np.zeros(k, dtype='float64')
                for (topic, val) in doc_dist_gensim:
                    doc_dist_numpy[topic] = val

                # now that we have this document's theta, add it to the sum.
                model_dict[model_key]['doc_topic_sums'][subreddit] += doc_dist_numpy

                # From Thompson & Blei (2018):
                nz_theta_d = doc_dist_numpy > 0
                log_theta_d = xlogy(nz_theta_d, doc_dist_numpy)

                model_theta_dict[model_key] = {'nz_theta_d': nz_theta_d, 'log_theta_d': log_theta_d}

            # For each word type that occurs in doc, iterate through each model to make topic assignments.
            model_doc_token_topics = {model_key: np.zeros(model_key[3]) for model_key in model_dict}
            for (word_id, word_count) in doc:

                # Estimate topics for each model.
                for model_key in model_dict:
                    k = model_key[3]
                    #topic_assingments = assign_type_to_topic()

                    # From Thompson & Blei (2018). Basically for the current word, get its
                    # probability in each topic (nz_phis.T[word_id]). Multiply each element in this k-dimensional
                    # vector by the corresponding elements in the document's nonzero theta vector. For each element
                    # that is nonzero, return exponent(log phi values of the word in each topic + log theta values
                    # of the document. Otherwise, return 0. Not sure why the .ravel() at the end--it seems that
                    # this will return a k-dimensional vector with or without it. The resulting distribution
                    # provides the distribution p(topic | word) from which we can make an assignment of the token
                    # to a topic.
                    topic_dist = np.where(model_dict[model_key]['nz_phis'].T[word_id] * model_theta_dict[model_key]['nz_theta_d'] != 0,
                                          np.exp(model_dict[model_key]['log_phis'].T[word_id] + model_theta_dict[model_key]['log_theta_d']),
                                          0.0).ravel()

                    # Normalize distribution p(topic | word, phi, theta):
                    topic_dist = topic_dist / topic_dist.sum()

                    # Draw a topic from topic_dist for however many times the word occurs in the document.
                    topics = np.random.choice(k, size=int(word_count), p=topic_dist)

                    for topic_i in topics:
                        model_doc_token_topics[model_key][topic_i] += 1

            # now we have token-topic assingment counts for each word type present in the current document.
            # START HERE -->
            # update token-topic assignment counts
            for model_key in model_dict:
                model_doc_topic_counts = model_doc_token_topics[model_key]

                model_dict[model_key]['token_topic_counts'][subreddit] += model_doc_topic_counts

                # also make the token-topic distribution and add it to ongoing count
                model_doc_token_dist = model_doc_topic_counts / model_doc_topic_counts.sum()
                model_dict[model_key]['doc_topic_tokens_sums'][subreddit] += model_doc_token_dist

                model_dict[model_key]['doc_counts'][subreddit] += 1

        # Now we are done with all documents in a subreddit. Summary stats for the subreddit can now be calculated
        # including the average theta distribution, the distribution of token-topic assignments, & the average
        # token-topic document distribution.
        for model_key in model_dict.keys():

            # All token-topic assignments have been counted for this subreddit, so store those counts in
            # token_assignment_counts for later use and write them to file.
            token_topic_freqs_fpath = os.path.join(model_dict[model_key]['tdists_dir'],
                                                   'token_topic_freqs_' + subreddit + '.csv')
            write_token_topic_freqs(token_topic_freqs_fpath,
                                    model_dict[model_key]['token_topic_counts'][subreddit])

            # Find average theta distribution by dividing the summed thetas by the number of documents.
            avg_doc_topic_fpath = os.path.join(model_dict[model_key]['tdists_dir'],
                                               'avg_doc_topic_' + subreddit + '.txt')
            avg_doc_topic = model_dict[model_key]['doc_topic_sums'][subreddit] / float(model_dict[model_key]['doc_counts'][subreddit])
            np.savetxt(avg_doc_topic_fpath, avg_doc_topic)

            # Find the average topic distribution of each document from token-topic assignments by dividing the sum of the
            # document distributions by the number of documents.
            avg_doc_topic_tokens_fpath = os.path.join(model_dict[model_key]['tdists_dir'],
                                                      'avg_doc_topic_from_tokens_' + subreddit + '.txt')
            avg_doc_topic_from_tokens = model_dict[model_key]['doc_topic_tokens_sums'][subreddit] / float(model_dict[model_key]['doc_counts'][subreddit])
            np.savetxt(avg_doc_topic_tokens_fpath, avg_doc_topic_from_tokens)

    # topic model summary files can now be written
    # Topic summary file. Possible things to include:
    #   - entropy of the topic's word distribution (what does this really tell us that is useful?)
    #   - entropy of topic over subreddits
    #   - top N words & probabilities OR top words & probabilities up to some cumulative probability (eg, the
    #     topic words needed to account for at least 50% of the topic's word distribution.
    #   - number of tokens assigned to each subreddit. Can also do as a proportion of a subreddit's tokens
    #     assigned to each topic.
    for model_key in model_dict:
        subreddit_entropy_counts, subreddit_entropy_props = get_subreddit_entropy(model_dict[model_key]['token_topic_counts'])

        phis = model_dict[model_key]['model'].get_topics()
        k = model_key[3]
        topic_entropies = [measures.entropy(phis[topic_i]) for topic_i in range(k)]

        max_subreddit_count, max_subreddit_prop = get_subreddits_w_max_topics(model_dict[model_key]['token_topic_counts'])

        # model_key = (training_corpus_type, sample_name, corpus_name, k)
        topic_summary_fpath = os.path.join(cons.lda_dir,
                                           model_key[0],
                                           model_key[1],
                                           model_key[1] + '-' + str(k),
                                           'topics_summary.csv')
        write_topics_summary_file(topic_summary_fpath,
                                  subreddit_entropy_counts, subreddit_entropy_props,
                                  topic_entropies,
                                  max_subreddit_count, max_subreddit_prop,
                                  model_dict[model_key]['model'],
                                  k)


if __name__ == '__main__':
    subreddit_list = cons.subreddit_list
	
	# Note: depending on available computer memory, it may be convenient to do the K=250 topic models separately as is done below.
	
	normal_models_to_do_20_100 = [('combined_subs_normal', 'combo_1', 'normal_corpus', [20, 100]),
                                  ('combined_subs_normal', 'combo_2', 'normal_corpus', [20, 100]),
                                  ('combined_subs_normal', 'combo_3', 'normal_corpus', [20, 100]),
                                  ('combined_subs_normal', 'combo_4', 'normal_corpus', [20, 100]),
                                  ('combined_subs_normal', 'combo_5', 'normal_corpus', [20, 100])]
    get_topic_distributions(normal_models_to_do_20_100,
                            'normal_corpus',
                            subreddit_list)

    normal_models_to_do_250 = [('combined_subs_normal', 'combo_1', 'normal_corpus', [250]),
                               ('combined_subs_normal', 'combo_2', 'normal_corpus', [250]),
                               ('combined_subs_normal', 'combo_3', 'normal_corpus', [250]),
                               ('combined_subs_normal', 'combo_4', 'normal_corpus', [250]),
                               ('combined_subs_normal', 'combo_5', 'normal_corpus', [250])]
    get_topic_distributions(normal_models_to_do_250,
                               'normal_corpus',
                               subreddit_list)

    reduced_models_to_do_20_100 = [('combined_subs_reduced_001', 'combo_1', 'reduced_corpus_001', [20, 100]),
                                   ('combined_subs_reduced_001', 'combo_2', 'reduced_corpus_001', [20, 100]),
                                   ('combined_subs_reduced_001', 'combo_3', 'reduced_corpus_001', [20, 100]),
                                   ('combined_subs_reduced_001', 'combo_4', 'reduced_corpus_001', [20, 100]),
                                   ('combined_subs_reduced_001', 'combo_5', 'reduced_corpus_001', [20, 100])]
    get_topic_distributions(reduced_models_to_do_20_100,
                            'reduced_corpus_001',
                            subreddit_list)

    reduced_models_to_do_250 = [('combined_subs_reduced_001', 'combo_1', 'reduced_corpus_001', [250]),
                                ('combined_subs_reduced_001', 'combo_2', 'reduced_corpus_001', [250]),
                                ('combined_subs_reduced_001', 'combo_3', 'reduced_corpus_001', [250]),
                                ('combined_subs_reduced_001', 'combo_4', 'reduced_corpus_001', [250]),
                                ('combined_subs_reduced_001', 'combo_5', 'reduced_corpus_001', [250])]
    get_topic_distributions(reduced_models_to_do_250,
                            'reduced_corpus_001',
                            subreddit_list)


