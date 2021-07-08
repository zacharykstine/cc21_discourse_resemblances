"""
# Zachary Kimo Stine, 2021-01-26
#
# Compare subreddits as represented by distributions of LDA topics. There are several ways we have
# constructed a subreddit's topic distribution:
#    1) Distribution of token-topic assignments. File name form: token_topic_freqs_<subreddit>.csv
#    2) Mean theta distribution. File name form: avg_doc_topic_<subreddit>.txt
#    3) Mean document-topic distribution from token-topic assignments. File name form: avg_doc_topic_from_tokens_<subreddit>.txt
#
# Results used in the corresponding paper only consider option 1 based on probabilistic assignments of each token to a topic.
"""
import cons
import os
import csv
import numpy as np
import itertools
import measures
import gensim


def topic_distribution_divergences(sub_1_dist, sub_2_dist):
    """
	For sub1 and sub2, returns JSD(sub1 || sub2), KLD(sub1 || sub2), and KLD(sub2 || sub1).
	"""

    jsd = measures.js_divergence(sub_1_dist, sub_2_dist)
    kld_sub_1 = measures.kl_divergence(sub_1_dist, sub_2_dist)
    kld_sub_2 = measures.kl_divergence(sub_2_dist, sub_1_dist)

    return jsd, kld_sub_1, kld_sub_2


def per_topic_comparisons(sub_1_name, sub_1_dist, sub_2_name, sub_2_dist):
    """
	Calculates topic-level divergences to identify salient topics for subreddits.
    """
    per_topic_divergences = {'jsd': measures.per_word_js_divergence(sub_1_dist, sub_2_dist),
                             'kld_' + sub_1_name: measures.per_word_kl_divergence(sub_1_dist, sub_2_dist),
                             'kld_' + sub_2_name: measures.per_word_kl_divergence(sub_2_dist, sub_1_dist)}
    return per_topic_divergences


def get_top_words(lda_model, k, num_words=3):
    """
    Returns a small number of high-probaility words for each topic as summary information.
    """
    topic_words = {}
    for t_index in range(k):
        top_words = lda_model.show_topic(t_index, num_words)
        topic_words[t_index] = [w for (w, w_prob) in top_words]
    return topic_words


def read_token_topic_dist(topic_dist_fpath, k):
    """
    Reads in topic counts & proportions based on token assingments to topics.
    """
    topic_freqs = np.zeros(k)
    topic_props = np.zeros(k)

    with open(topic_dist_fpath, 'r') as infile:
        freader = csv.DictReader(infile)

        for row in freader:
            topic_index = int(row['topic'])
            token_count = float(row['token_count'])
            token_prop = float(row['token_proportion'])

            topic_freqs[topic_index] = token_count
            topic_props[topic_index] = token_prop

    return topic_freqs, topic_props


def topic_comparisons(training_corpus_type, corpus_name, k_list, num_samples, subreddit_list):
    """
    For each model and for each pair of subreddits, calculates divergences between subreddits based on topic distributions
	and writes all relevant data to files.
    """
    model_sample_dir = os.path.join(cons.lda_dir, training_corpus_type)

    for sample_i in range(num_samples):
        sample_name = 'combo_' + str(sample_i + 1)
        sample_dir = os.path.join(model_sample_dir, sample_name)

        for k in k_list:
            model_dir = os.path.join(sample_dir, sample_name + '-' + str(k))

            lda_model = gensim.models.LdaModel.load(os.path.join(model_dir, 'model_files'))
            topic_top_words = get_top_words(lda_model, k, num_words=3)

            # Each kind of subreddit-topic distribution from the current model is stored in the following directory.
            topic_dists_dir = os.path.join(model_dir, 'topic_analysis')

            # Create directory for storing model-specific topic comparisons.
            topic_comps_dir = cons.makedir(os.path.join(model_dir, 'topic_comparisons'))

            subreddit_divergence_dict = {}

            # Iterate through each pair of subreddits and compare each version of their global topic distributions.
            for sub_1, sub_2 in itertools.combinations(subreddit_list, 2):

                # ---------------------------------------------------------------------------------------------------- #
                # 1) Carry out comparisons on subreddit-topic distributions from token-topic assignments.              #
                # ---------------------------------------------------------------------------------------------------- #

                # Read in subreddit-topic distributions calculated from all token-topic assignments in the subreddit.
                sub_1_token_topic_dist_fpath = os.path.join(topic_dists_dir, 'token_topic_freqs_' + sub_1 + '.csv')
                sub_1_token_topic_freqs, sub_1_token_topic_props = read_token_topic_dist(sub_1_token_topic_dist_fpath,
                                                                                         k)

                sub_2_token_topic_dist_fpath = os.path.join(topic_dists_dir, 'token_topic_freqs_' + sub_2 + '.csv')
                sub_2_token_topic_freqs, sub_2_token_topic_props = read_token_topic_dist(sub_2_token_topic_dist_fpath,
                                                                                         k)

                # Calculate divergences between distributions.
                token_topics_jsd, token_topics_kld_sub_1, token_topics_kld_sub_2 = topic_distribution_divergences(sub_1_token_topic_props, sub_2_token_topic_props)
                subreddit_divergence_dict[(sub_1, sub_2)] = {'token_topics_jsd': token_topics_jsd,
                                                             'token_topics_kld_' + sub_1: token_topics_kld_sub_1,
                                                             'token_topics_kld_' + sub_2: token_topics_kld_sub_2}

                # Calculate per-topic divergence contributions.
                token_topic_dist_comparisons = per_topic_comparisons(sub_1, sub_1_token_topic_props,
                                                                     sub_2, sub_2_token_topic_props)

                assert round(sum(token_topic_dist_comparisons['jsd']), 6) == round(token_topics_jsd, 6)
                assert round(sum(token_topic_dist_comparisons['kld_' + sub_1]), 6) == round(token_topics_kld_sub_1, 6)
                assert round(sum(token_topic_dist_comparisons['kld_' + sub_2]), 6) == round(token_topics_kld_sub_2, 6)

                # ---------------------------------------------------------------------------------------------------- #
                # 2) Carry out comparisons on subreddit-topic distributions from mean theta distributions.             #
                # ---------------------------------------------------------------------------------------------------- #
                sub_1_mean_theta = np.loadtxt(os.path.join(topic_dists_dir, 'avg_doc_topic_' + sub_1 + '.txt'))
                sub_2_mean_theta = np.loadtxt(os.path.join(topic_dists_dir, 'avg_doc_topic_' + sub_2 + '.txt'))

                # Calculate divergences between distributions.
                mean_theta_jsd, mean_theta_kld_sub_1, mean_theta_kld_sub_2 = topic_distribution_divergences(sub_1_mean_theta, sub_2_mean_theta)
                subreddit_divergence_dict[(sub_1, sub_2)]['mean_theta_jsd'] = mean_theta_jsd
                subreddit_divergence_dict[(sub_1, sub_2)]['mean_theta_kld_' + sub_1] = mean_theta_kld_sub_1
                subreddit_divergence_dict[(sub_1, sub_2)]['mean_theta_kld_' + sub_2] = mean_theta_kld_sub_2

                # Calculate per-topic divergence contributions.
                mean_theta_comparisons = per_topic_comparisons(sub_1, sub_1_mean_theta,
                                                               sub_2, sub_2_mean_theta)
                
                assert round(sum(mean_theta_comparisons['jsd']), 6) == round(mean_theta_jsd, 6)
                assert round(sum(mean_theta_comparisons['kld_' + sub_1]), 6) == round(mean_theta_kld_sub_1, 6)
                assert round(sum(mean_theta_comparisons['kld_' + sub_2]), 6) == round(mean_theta_kld_sub_2, 6)

                # ---------------------------------------------------------------------------------------------------- #
                # 3) Carry out comparisons on subreddit-topic distributions from mean document-topic distribution from #
                # token-topic assignments.                                                                             #
                # ---------------------------------------------------------------------------------------------------- #
                sub_1_mean_token_topics = np.loadtxt(os.path.join(topic_dists_dir, 'avg_doc_topic_from_tokens_'+ sub_1 + '.txt'))
                sub_2_mean_token_topics = np.loadtxt(os.path.join(topic_dists_dir, 'avg_doc_topic_from_tokens_' + sub_2 + '.txt'))

                # Calculate divergences between distributions.
                mean_tokens_jsd, mean_tokens_kld_sub_1, mean_tokens_kld_sub_2 = topic_distribution_divergences(sub_1_mean_token_topics, sub_2_mean_token_topics)
                subreddit_divergence_dict[(sub_1, sub_2)]['mean_token_topics_jsd'] = mean_tokens_jsd
                subreddit_divergence_dict[(sub_1, sub_2)]['mean_token_topics_kld_' + sub_1] = mean_tokens_kld_sub_1
                subreddit_divergence_dict[(sub_1, sub_2)]['mean_token_topics_kld_' + sub_2] = mean_tokens_kld_sub_2

                # Calculate per-topic divergence contributions.
                mean_tokens_comparisons = per_topic_comparisons(sub_1, sub_1_mean_token_topics,
                                                                sub_2, sub_2_mean_token_topics)

                assert round(sum(mean_tokens_comparisons['jsd']), 6) == round(mean_tokens_jsd, 6)
                assert round(sum(mean_tokens_comparisons['kld_' + sub_1]), 6) == round(mean_tokens_kld_sub_1, 6)
                assert round(sum(mean_tokens_comparisons['kld_' + sub_2]), 6) == round(mean_tokens_kld_sub_2, 6)

                #
                # Write per-topic comparison data to file
                #
                per_topic_comp_fpath = os.path.join(topic_comps_dir, sub_1 + '_' + sub_2 + '_topic_divergences.csv')
                with open(per_topic_comp_fpath, 'w', encoding='utf-8', newline='') as outfile:
                    fwriter = csv.writer(outfile)

                    fwriter.writerow(['topic_index',
                                      'word_1', 'word_2', 'word_3',
                                      'tokens_count_' + sub_1, 'tokens_proportion_' + sub_1,
                                      'tokens_count_' + sub_2, 'tokens_proportion_' + sub_2,
                                      'tokens_jsd',
                                      'tokens_kld_' + sub_1,
                                      'tokens_kld_' + sub_2,
                                      'mean_theta_' + sub_1, 'mean_theta_' + sub_2,
                                      'mean_theta_jsd',
                                      'mean_theta_kld' + sub_1,
                                      'mean_theta_kld_' + sub_2,
                                      'mean_tokens_' + sub_1, 'mean_tokens_' + sub_2,
                                      'mean_tokens_jsd',
                                      'mean_tokens_kld_' + sub_1,
                                      'mean_tokens_kld_' + sub_2])

                    for topic_index in range(k):

                        fwriter.writerow([topic_index,
                                          topic_top_words[topic_index][0], topic_top_words[topic_index][1], topic_top_words[topic_index][2],
                                          sub_1_token_topic_freqs[topic_index], sub_1_token_topic_props[topic_index],
                                          sub_2_token_topic_freqs[topic_index], sub_2_token_topic_props[topic_index],
                                          token_topic_dist_comparisons['jsd'][topic_index],
                                          token_topic_dist_comparisons['kld_' + sub_1][topic_index],
                                          token_topic_dist_comparisons['kld_' + sub_2][topic_index],
                                          sub_1_mean_theta[topic_index], sub_2_mean_theta[topic_index],
                                          mean_theta_comparisons['jsd'][topic_index],
                                          mean_theta_comparisons['kld_' + sub_1][topic_index],
                                          mean_theta_comparisons['kld_' + sub_2][topic_index],
                                          sub_1_mean_token_topics[topic_index], sub_2_mean_token_topics[topic_index],
                                          mean_tokens_comparisons['jsd'][topic_index],
                                          mean_tokens_comparisons['kld_' + sub_1][topic_index],
                                          mean_tokens_comparisons['kld_' + sub_2][topic_index]])

            # At this point, each subreddit pair has had its topic distributions compared. Now to write the overall
            # divergence values between each subreddit.
            subreddit_divergences_fpath = os.path.join(model_dir, 'subreddit_topic_divergences.csv')
            with open(subreddit_divergences_fpath, 'w', newline='') as outfile:
                fwriter = csv.writer(outfile)

                fwriter.writerow(['subreddit_1',
                                  'subreddit_2',
                                  'token_topics_jsd',
                                  'token_topics_kld_sub_1',
                                  'token_topics_kld_sub_2',
                                  'mean_theta_jsd',
                                  'mean_theta_kld_sub_1',
                                  'mean_theta_kld_sub_2',
                                  'mean_tokens_jsd',
                                  'mean_tokens_kld_sub_1',
                                  'mean_tokens_kld_sub_2'])

                for sub_key in subreddit_divergence_dict.keys():
                    sub_1 = sub_key[0]
                    sub_2 = sub_key[1]
                    print(subreddit_divergence_dict[sub_key])

                    fwriter.writerow([sub_1,
                                      sub_2,
                                      subreddit_divergence_dict[sub_key]['token_topics_jsd'],
                                      subreddit_divergence_dict[sub_key]['token_topics_kld_' + sub_1],
                                      subreddit_divergence_dict[sub_key]['token_topics_kld_' + sub_2],
                                      subreddit_divergence_dict[sub_key]['mean_theta_jsd'],
                                      subreddit_divergence_dict[sub_key]['mean_theta_kld_' + sub_1],
                                      subreddit_divergence_dict[sub_key]['mean_theta_kld_' + sub_2],
                                      subreddit_divergence_dict[sub_key]['mean_token_topics_jsd'],
                                      subreddit_divergence_dict[sub_key]['mean_token_topics_kld_' + sub_1],
                                      subreddit_divergence_dict[sub_key]['mean_token_topics_kld_' + sub_2]])


if __name__ == '__main__':
    subreddit_list = cons.subreddit_list

    # Specify which LDA models are to be used in comparisons:
	# Original corpus models:
	topic_comparisons('combined_subs_normal',
                      'normal_corpus',
                      [250],
                      5,
                      subreddit_list)
	
	# Modified corpus models:
    topic_comparisons('combined_subs_reduced_001',
                      'reduced_corpus_001',
                      [250],
                      5,
                      subreddit_list)