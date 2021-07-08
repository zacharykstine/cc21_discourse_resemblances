"""
# Zachary K. Stine
# updated 2021-01-06
#
# Description:
#   1) Count word frequencies from corpus for each subreddit.
#   2) Calculate the JS divergence between the relative word frequencies of each pair of subreddits.
#   3) Calculate the word-level divergence contributions for each word between each pair of subreddits. Word-level divergences 
#      include JS, KL, and partial KL (i.e., KL using the mean).
"""
import os
import cons
import functions
import csv
import numpy as np
import gensim
import measures
import itertools
import numpy as np


def lexical_comparison(fpath,
                       dictionary,
                       target_name, target_freqs, target_props,
                       comparison_name, comparison_freqs, comparison_props):
	"""
	Function calculates three word-level divergences and writes them to file: JS, KL, and partial KL (ie KL to mean).
	"""

    # a) Jensen-Shannon divergence
    per_word_jsds = measures.per_word_js_divergence(target_props,
                                                    comparison_props)

    # b) Kullback-Leibler divergence in both directions.
    per_word_klds_target = measures.per_word_kl_divergence(target_props,
                                                           comparison_props)

    per_word_klds_comparison = measures.per_word_kl_divergence(comparison_props,
                                                               target_props)

    # c) Partial Kullback-Leibler divergence (uses the mean distribution to ensure the quantity is defined).
    mean_props = np.mean([target_props, comparison_props], axis=0, dtype=np.float64)
    per_word_partial_kls_target = measures.per_word_kl_divergence(target_props, mean_props)
    per_word_partial_kls_comparison = measures.per_word_kl_divergence(comparison_props, mean_props)

    with open(fpath, 'w', encoding='utf-8', newline='') as outfile:
        fwriter = csv.writer(outfile)

        fwriter.writerow(['word_id',
                          'word',
                          'count_' + target_name, 'proportion_' + target_name,
                          'count_' + comparison_name, 'proportion_' + comparison_name,
                          'jsd',
                          'kld_' + target_name,
                          'kld_' + comparison_name,
                          'kld_to_mean_' + target_name,
                          'kld_to_mean_' + comparison_name])

        for word_id in dictionary.keys():
            fwriter.writerow([word_id,
                              dictionary[word_id],
                              target_freqs[word_id], target_props[word_id],
                              comparison_freqs[word_id], comparison_props[word_id],
                              per_word_jsds[word_id],
                              per_word_klds_target[word_id],
                              per_word_klds_comparison[word_id],
                              per_word_partial_kls_target[word_id],
                              per_word_partial_kls_comparison[word_id]])


if __name__ == '__main__':
    subreddit_list = cons.subreddit_list
    dictionary_name = 'dictionary.dict'
    remake_word_freqs = False

    dictionary = gensim.corpora.dictionary.Dictionary.load(os.path.join(cons.corpora_dir, dictionary_name))

    # 1) For each subreddit, put frequencies and relative frequencies of each word in vocabulary into dictionary.
    word_dists = {}
    for subreddit in subreddit_list:

        # Check if a word frequency file already exists.
        sub_freqs_fpath = os.path.join(cons.corpora_dir, subreddit, 'word_freqs.csv')

        if os.path.exists(sub_freqs_fpath) and not remake_word_freqs:
            # Read word freqs into word_dists
            w_freqs, w_props = functions.read_word_frequencies(sub_freqs_fpath, dictionary)

        else:
            # Create CSV files with word frequencies.
            sub_corpus = gensim.corpora.MmCorpus(os.path.join(cons.corpora_dir, subreddit, 'normal_corpus', 'normal_corpus'))
            w_freqs, w_props = functions.write_word_freqs_and_props(sub_freqs_fpath, dictionary, sub_corpus)


    # 2) Calculate divergences between word distributions of each each subreddit.
    pairwise_word_divergences = {}
    pairwise_divs_dir = cons.makedir(os.path.join(cons.lexcomp_dir, 'word_divergences_pairwise'))
    for sub_1, sub_2 in itertools.combinations(subreddit_list, 2):
	
	    # Calculate JSD between word proportions of each pair of subreddits.
        jsd = measures.js_divergence(word_dists[sub_1]['props'],
                                     word_dists[sub_2]['props'])
        pairwise_word_divergences[(sub_1, sub_2)] = {'jsd': jsd}

        # Also, do all per-word divergence measures between each pair of subreddits and write to CSV.
        word_divergences_fpath = os.path.join(pairwise_divs_dir,
                                              sub_1 + '_' + sub_2 + '_word_divergences.csv')
        lexical_comparison(word_divergences_fpath,
                           dictionary,
                           sub_1, word_dists[sub_1]['freqs'], word_dists[sub_1]['props'],
                           sub_2, word_dists[sub_2]['freqs'], word_dists[sub_2]['props'])

    # Write pairwise JS divergences to file.
    pairwise_divergences_fpath = os.path.join(cons.lexcomp_dir, 'subreddit_divergences.txt')

    with open(pairwise_divergences_fpath, 'w') as outfile:
        outfile.write(str(pairwise_word_divergences))

    with open(pairwise_divergences_fpath[:-3] + 'csv', 'w', newline='') as outfile:
        fwriter = csv.writer(outfile)
        fwriter.writerow(['subreddit_1', 'subreddit_2', 'jsd'])
        for (sub_1, sub_2) in pairwise_word_divergences.keys():
            fwriter.writerow([sub_1, sub_2, pairwise_word_divergences[(sub_1, sub_2)]['jsd']])