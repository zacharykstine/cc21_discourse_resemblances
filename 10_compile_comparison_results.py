"""
# Zachary K. Stine, 2021-01-27
#
# Description: Once comparisons have been done across all models with different training corpora configurations,
# different corpus samples, and different values of k, it will be helpful to collect all of that disparate information
# into a single place for review and for making plots. This script gathers all of this information and places it in
# single location.
"""
import cons
import os
import itertools
import csv
import numpy as np
import functions
import measures
import gensim
import math


def get_reduced_word_dist(original_dist, remove_indices):
    """
    Returns word distribution after filtering out words specified by remove_indices.
    """
    return np.array([p if i not in remove_indices else 0.0 for i, p in enumerate(original_dist)])


def read_word_ids_from_csv(fpath):
    """
    Reads in word IDs from specified file.
    """
    ids = []
    with open(fpath, 'r', encoding='utf-8') as infile:
        freader = csv.DictReader(infile)

        for row in freader:
            ids.append(int(row['word_id']))

    return set(ids)


def collect_lexical_comparison_data(fpath):
    """
    Calculates JSD between word distributions of each subreddit under the following vocabulary configurations:
      a) original
      b) noncommon words removed
      c) noncommon + kld threshold of 0.0010 removed
      d) noncommon + kld threshold of 0.0005 removed 
	  e) noncommon + kld threshold of 0.0001 removed 
    """
    subreddit_list = cons.subreddit_list

    lexical_comparisons = {}

    # a) original vocabulary JSD has already been computed and just needs to be read from file.
    original_vocab_jsd_fpath = os.path.join(cons.lexcomp_dir, 'subreddit_divergences.csv')
    with open(original_vocab_jsd_fpath, 'r') as infile:
        freader = csv.DictReader(infile)

        for row in freader:
            sub_1 = row['subreddit_1']
            sub_2 = row['subreddit_2']
            jsd = float(row['jsd'])

            lexical_comparisons[(sub_1, sub_2)] = {'original': jsd}

    # For the remaining vocabulary configurations, read in the word distribution for each subreddit then read in
    # the word IDs of the removed word types. For those word IDs, set the corresponding values to zero in the
    # distributions and calculate the JSD.
    dictionary = gensim.corpora.dictionary.Dictionary.load(os.path.join(cons.corpora_dir, 'dictionary.dict'))
    original_vocab_size = len(dictionary)

    sub_word_distributions = {}
    for subreddit in subreddit_list:
        sub_freqs_fpath = os.path.join(cons.corpora_dir, subreddit, 'word_freqs.csv')
        w_freqs, w_props = functions.read_word_frequencies(sub_freqs_fpath, dictionary)

        sub_word_distributions[subreddit] = w_props

    # b) vocabulary with noncommon words removed requires us to read in the word IDs for words to be removed from the
    # vocabulary, then set the corresponding values to zero in the distributions to calculate remaining JSD.
    noncommon_words_fpath = os.path.join(cons.lexremoval_dir, '1_uncommon_words.csv')
    noncommon_word_ids = read_word_ids_from_csv(noncommon_words_fpath)


    # c) vocabulary with noncommon words and words over the per-word kld value of 0.001 in pairwise comparisons.
    pairwise_words_001_fpath = os.path.join(cons.lexremoval_dir, '2_pairwise_kld_001_words.csv')
    pairwise_words_001_ids = read_word_ids_from_csv(pairwise_words_001_fpath)

    # d)
    pairwise_words_0005_fpath = os.path.join(cons.lexremoval_dir, '2_pairwise_kld_0005_words.csv')
    pairwise_words_0005_ids = read_word_ids_from_csv(pairwise_words_0005_fpath)

    # e)
    pairwise_words_0001_fpath = os.path.join(cons.lexremoval_dir, '2_pairwise_kld_0001_words.csv')
    pairwise_words_0001_ids = read_word_ids_from_csv(pairwise_words_0001_fpath)

    print('original vocabulary size  : ' + str(original_vocab_size) + ' types.')
    print('common  vocabulary size   : ' + str(original_vocab_size - len(noncommon_word_ids)) + ' types.')
    print('kld 0.001 vocabulary size : ' + str(original_vocab_size - len(noncommon_word_ids) - len(pairwise_words_001_ids)) + ' types.')
    print('kld 0.0005 vocabulary size: ' + str(original_vocab_size - len(noncommon_word_ids) - len(pairwise_words_0005_ids)) + ' types.')
    print('kld 0.0001 vocabulary size: ' + str(original_vocab_size - len(noncommon_word_ids) - len(pairwise_words_0001_ids)) + ' types.')

    for sub_key in lexical_comparisons.keys():
        print(sub_key)
        sub_1 = sub_key[0]
        sub_2 = sub_key[1]

        per_word_comparison_fpath = os.path.join(cons.lexcomp_dir,
                                                 'word_divergences_pairwise',
                                                 sub_1 + '_' + sub_2 + '_word_divergences.csv')
												 
        # read in comparison file
        per_word_comparisons = functions.read_lexical_comparison_file(per_word_comparison_fpath,
                                                                      original_vocab_size,
                                                                      sub_1, sub_2)
        per_word_jsd_vector = per_word_comparisons['jsd']

        # sum all per-word JSD values.
        per_word_jsd = np.sum(per_word_jsd_vector)
        lexical_comparisons[sub_key]['per_word'] = per_word_jsd
        print('jsd: ' + str(round(lexical_comparisons[sub_key]['original'], 4)) + ' vs ' + str(round(per_word_jsd, 4)))

        # b) remaining jsd after removing noncommon words.
        remove_ids = noncommon_word_ids
        common_jsd = measures.js_divergence(get_reduced_word_dist(sub_word_distributions[sub_1], remove_ids),
                                            get_reduced_word_dist(sub_word_distributions[sub_2], remove_ids),
                                            checksum=False)
        lexical_comparisons[sub_key]['common_words'] = common_jsd

        per_word_jsd_common = sum([word_jsd for i, word_jsd in enumerate(per_word_jsd_vector) if i not in remove_ids])
        lexical_comparisons[sub_key]['per_word_common'] = per_word_jsd_common
        print('common words jsd: ' + str(round(common_jsd, 4)) + ' vs ' + str(round(per_word_jsd_common, 4)))

        # c) remaining jsd after removing words over the per-word kld threshold of 0.0010
        remove_ids = noncommon_word_ids.union(pairwise_words_001_ids)
        pairwise_001_jsd = measures.js_divergence(get_reduced_word_dist(sub_word_distributions[sub_1], remove_ids),
                                                  get_reduced_word_dist(sub_word_distributions[sub_2], remove_ids),
                                                  checksum=False)
        lexical_comparisons[sub_key]['kld_001'] = pairwise_001_jsd

        per_word_jsd_001 = sum([word_jsd for i, word_jsd in enumerate(per_word_jsd_vector) if i not in remove_ids])
        lexical_comparisons[sub_key]['per_word_001'] = per_word_jsd_001
        print('001 words jsd: ' + str(round(pairwise_001_jsd, 4)) + ' vs ' + str(round(per_word_jsd_001, 4)))

        # d) remaining jsd after removing words over the per-word kld threshold of 0.0005
        remove_ids = noncommon_word_ids.union(pairwise_words_0005_ids)
        pairwise_0005_jsd = measures.js_divergence(get_reduced_word_dist(sub_word_distributions[sub_1], remove_ids),
                                                   get_reduced_word_dist(sub_word_distributions[sub_2], remove_ids),
                                                   checksum=False)
        lexical_comparisons[sub_key]['kld_0005'] = pairwise_0005_jsd

        per_word_jsd_0005 = sum([word_jsd for i, word_jsd in enumerate(per_word_jsd_vector) if i not in remove_ids])
        lexical_comparisons[sub_key]['per_word_0005'] = per_word_jsd_0005
        print('0005 words jsd: ' + str(round(pairwise_0005_jsd, 4)) + ' vs ' + str(round(per_word_jsd_0005, 4)))

        # c) remaining jsd after removing words over the per-word kld threshold of 0.0001
        remove_ids = noncommon_word_ids.union(pairwise_words_0001_ids)
        pairwise_0001_jsd = measures.js_divergence(get_reduced_word_dist(sub_word_distributions[sub_1], remove_ids),
                                                   get_reduced_word_dist(sub_word_distributions[sub_2], remove_ids),
                                                   checksum=False)
        lexical_comparisons[sub_key]['kld_0001'] = pairwise_0001_jsd

        per_word_jsd_0001 = sum([word_jsd for i, word_jsd in enumerate(per_word_jsd_vector) if i not in remove_ids])
        lexical_comparisons[sub_key]['per_word_0001'] = per_word_jsd_0001
        print('0001 words jsd: ' + str(round(pairwise_0001_jsd, 4)) + ' vs ' + str(round(per_word_jsd_0001, 4)))

        print()

    with open(fpath, 'w', encoding='utf-8', newline='') as outfile:
        fwriter = csv.writer(outfile)

        fwriter.writerow(['subreddit_1',
                          'subreddit_2',
                          'original_jsd',
                          'common_words_jsd',
                          'kld_001_jsd',
                          'kld_0005_jsd',
                          'kld_0001_jsd',
                          'original_per_word_jsd',
                          'common_per_word_jsd',
                          'kld_001_per_word_jsd',
                          'kld_0005_per_word_jsd',
                          'kld_0001_per_word_jsd'])

        for sub_key in lexical_comparisons.keys():

            fwriter.writerow([sub_key[0],
                              sub_key[1],
                              lexical_comparisons[sub_key]['original'],
                              lexical_comparisons[sub_key]['common_words'],
                              lexical_comparisons[sub_key]['kld_001'],
                              lexical_comparisons[sub_key]['kld_0005'],
                              lexical_comparisons[sub_key]['kld_0001'],
                              lexical_comparisons[sub_key]['per_word'],
                              lexical_comparisons[sub_key]['per_word_common'],
                              lexical_comparisons[sub_key]['per_word_001'],
                              lexical_comparisons[sub_key]['per_word_0005'],
                              lexical_comparisons[sub_key]['per_word_0001']])

def read_token_topic_dist(topic_dist_fpath, k):
    """
    Reads in topic counts and proportions from file.
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


def collect_topic_comparison_data(results_dir, training_corpus_type, sample_num, k):
    """
    Calculates summary statistics about relationships between subreddits based on divergences between their topic distributions.
    """

    subreddit_list = cons.subreddit_list

    divergences = {}

    # Initialize dictionaries.
    for sub_1, sub_2 in itertools.combinations(subreddit_list, 2):
        divergences[(sub_1, sub_2)] = {'jsd': [],
                                       'kld_' + sub_1: [],
                                       'kld_' + sub_2: []}

    # Iterate through each corpus sample and model:
    for sample_i in range(sample_num):
        sample_name = 'combo_' + str(sample_i + 1)
        model_name = sample_name + '-' + str(k)

        # Directory for topic distributions from model trained on the specified corpus:
        topic_analysis_dir = os.path.join(cons.lda_dir,
                                          training_corpus_type,
                                          sample_name,
                                          model_name,
                                          'topic_analysis')

        # Read in each subreddit's topic distribution for the current model:
        topic_dists = {}
        for subreddit in subreddit_list:
            topic_tokens_fpath = os.path.join(topic_analysis_dir, 'token_topic_freqs_' + subreddit + '.csv')
            token_topic_counts, token_topic_props = read_token_topic_dist(topic_tokens_fpath, k)
            assert round(np.sum(token_topic_props), 6) == 1.0

            topic_dists[subreddit] = token_topic_props

        # Calculate JSD and KLD between each pair of subreddits. Append results to the lists in divergences.
        for sub_key in divergences.keys():
            sub_1 = sub_key[0]
            sub_2 = sub_key[1]

            divergences[sub_key]['jsd'].append(measures.js_divergence(topic_dists[sub_1],
                                                                      topic_dists[sub_2]))

            divergences[sub_key]['kld_' + sub_1].append(measures.kl_divergence(topic_dists[sub_1],
                                                                              topic_dists[sub_2]))

            divergences[sub_key]['kld_' + sub_2].append(measures.kl_divergence(topic_dists[sub_2],
                                                                              topic_dists[sub_1]))

    # At this point, we have divergences between all pairs of subreddits across each model that shares the same number
    # of topics, k, and is trained from the same type of corpus. We can write them to a csv along with the mean
    # divergence of each quantity.
    with open(os.path.join(results_dir, 'k-' + str(k) + '_token_topics_divergences_dict.txt'), 'w') as outfile:
        outfile.write(str(divergences))

    with open(os.path.join(results_dir, 'k-' + str(k) + '_token_topic_mean_divergences.csv'), 'w', newline='') as outfile:
        fwriter = csv.writer(outfile)
        fwriter.writerow(['subreddit_1', 'subreddit_2', 'mean_jsd', 'margin_of_error',
                          'lo_90_ci_mean_jsd', 'hi_90_ci_mean_jsd', 'std_jsd',
                          'mean_kld_sub_1', 'mean_kld_sub_2'])

        for sub_key in divergences.keys():
            sub_1 = sub_key[0]
            sub_2 = sub_key[1]

            mean_jsd = np.mean(np.array(divergences[sub_key]['jsd']), dtype=np.float64)
            sample_std_jsd = np.std(np.array(divergences[sub_key]['jsd']), ddof=1, dtype=np.float64)

            # For N=5 (models), use T-score of 2.132 for 90% confidence interval.
            moe = 2.132 * (sample_std_jsd / math.sqrt(len(divergences[sub_key]['jsd'])))

            hi_90_jsd = mean_jsd + moe
            lo_90_jsd = mean_jsd - moe

            mean_kld_sub_1 = np.mean(np.array(divergences[sub_key]['kld_' + sub_1]), dtype=np.float64)
            mean_kld_sub_2 = np.mean(np.array(divergences[sub_key]['kld_' + sub_2]), dtype=np.float64)

            fwriter.writerow([sub_1, sub_2, mean_jsd, moe,
                              lo_90_jsd, hi_90_jsd, sample_std_jsd,
                              mean_kld_sub_1, mean_kld_sub_2])


if __name__ == '__main__':
    results_dir = cons.makedir(os.path.join(cons.project_dir, '5_all_results'))

    # 1) Lexical comparisons include the following:
    #    a) JSD between word distributions of original vocabulary.
    #    b) JSD between word distributions of reduced vocabulary (0.001 threshold).
    #    c) OPTIONAL: same as b but for threshold of 0.0005.
    lexical_comparison_fpath = os.path.join(results_dir, '1_lexical_comparisons.csv')
    collect_lexical_comparison_data(lexical_comparison_fpath)

    # 2) Topic comparisons from token-topic distributions.
    #    a) From models trained on the normal corpus
    #    b) From models trained on the lexically reduced corpus.
    # In both cases, we definitely want the JSD between them, but also the KLD between them.
	topic_tokens_normal_dir = cons.makedir(os.path.join(results_dir, 'topic_tokens_comparisons_normal'))
    collect_topic_comparison_data(topic_tokens_reduced_dir, 'combined_subs_normal', 5, 20)
    collect_topic_comparison_data(topic_tokens_reduced_dir, 'combined_subs_normal', 5, 100)
    collect_topic_comparison_data(topic_tokens_reduced_dir, 'combined_subs_normal', 5, 250)
	
    topic_tokens_reduced_dir = cons.makedir(os.path.join(results_dir, 'topic_tokens_comparisons_reduced_001'))
    collect_topic_comparison_data(topic_tokens_reduced_dir, 'combined_subs_reduced_001', 5, 20)
    collect_topic_comparison_data(topic_tokens_reduced_dir, 'combined_subs_reduced_001', 5, 100)
    collect_topic_comparison_data(topic_tokens_reduced_dir, 'combined_subs_reduced_001', 5, 250)
