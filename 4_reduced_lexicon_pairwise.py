"""
# Zachary K. Stine, 2021-01-12
# 
# This script identifies which words meet the specified criteria for being "distinguishing." This includes
# words that do not occur at all in at least one subreddit and words that meet a defined word-level KL divergence. 
# Multiple thresholds can be set to see how they differ.
"""

import os
import cons
import functions
import csv
import gensim
import numpy as np
import itertools


def get_uncommon_words(fpath, subreddit_list, dictionary):
    """
	Identifies words that do not occur at all in at least one subreddit. These words are written to a file
	and returned as a set of word IDs.
	"""
    noncommon_words = set()

    for subreddit in subreddit_list:
        word_freqs_fpath = os.path.join(cons.corpora_dir, subreddit, 'word_freqs.csv')
        word_freqs, word_props = functions.read_word_frequencies(word_freqs_fpath, dictionary)

        noncommon_words.update(word_id for word_id, word_count in enumerate(word_freqs) if word_count == 0.0)

    with open(fpath, 'w', encoding='utf-8', newline='') as outfile:
        fwriter = csv.writer(outfile)
        fwriter.writerow(['word_id', 'word'])

        for word_id in noncommon_words:
            fwriter.writerow([word_id, dictionary[word_id]])

    return noncommon_words


def read_lexical_comparison_file(fpath, vocab_size, subreddit_name, comparison_name):
    """
    Function reads lexical comparison file (see 3_lexical_comparison.py) and returns a dictionary of all word-level data.
    """

    sub_count_array = np.zeros(vocab_size, dtype=np.int)
    sub_prop_array = np.zeros(vocab_size, dtype=np.float64)

    comp_count_array = np.zeros(vocab_size, dtype=np.int)
    comp_prop_array = np.zeros(vocab_size, dtype=np.float64)

    jsd_array = np.zeros(vocab_size, dtype=np.float64)

    sub_kld_array = np.zeros(vocab_size, dtype=np.float64)
    comp_kld_array = np.zeros(vocab_size, dtype=np.float64)

    sub_kld_mean_array = np.zeros(vocab_size, dtype=np.float64)
    comp_kld_mean_array = np.zeros(vocab_size, dtype=np.float64)

    with open(fpath, 'r', encoding='utf-8') as infile:
        freader = csv.DictReader(infile)

        for row in freader:
            wid = int(row['word_id'])
            word = row['word']

            sub_count_array[wid] = float(row['count_' + subreddit_name])
            sub_prop_array[wid] = np.float64(row['proportion_' + subreddit_name])

            comp_count_array[wid] = float(row['count_' + comparison_name])
            comp_prop_array[wid] = np.float64(row['proportion_' + comparison_name])

            jsd_array[wid] = np.float64(row['jsd'])

            sub_kld_array[wid] = np.float64(row['kld_' + subreddit_name])
            comp_kld_array[wid] = np.float64(row['kld_' + comparison_name])

            sub_kld_mean_array[wid] = np.float64(row['kld_to_mean_' + subreddit_name])
            comp_kld_mean_array[wid] = np.float64(row['kld_to_mean_' + comparison_name])

    comparison_dict = {'counts_' + subreddit_name: sub_count_array,
                       'props_' + subreddit_name: sub_prop_array,
                       'counts_' + comparison_name: comp_count_array,
                       'props_' + comparison_name: comp_prop_array,
                       'jsd': jsd_array,
                       'kld_' + subreddit_name: sub_kld_array,
                       'kld_' + comparison_name: comp_kld_array,
                       'kld_to_mean_' + subreddit_name: sub_kld_mean_array,
                       'kld_to_mean_' + comparison_name: comp_kld_mean_array}
    return comparison_dict


def lexicon_analysis_kld_threshold(analysis_dir, noncommon_words, dictionary, kld_threshold):
    """
	Function identifies words that meet the KL divergence threshold and writes them to file.
	"""
    union_all_remove_ids = set()
    remaining_jsd_dict = {}

    for sub_1, sub_2 in itertools.combinations(subreddit_list, 2):
        print(sub_1 + ' and ' + sub_2 + ' comparison')
        print('----------------------------')

        comp_fpath = os.path.join(cons.lexcomp_dir,
                                  'word_divergences_pairwise',
                                  sub_1 + '_' + sub_2 + '_word_divergences.csv')

        comp_dict = read_lexical_comparison_file(comp_fpath, original_vocab_size, sub_1, sub_2)

        print('jsd(' + sub_1 + ', ' + sub_2 + ') = ' + str(round(sum(comp_dict['jsd']), 6)))
        print()

        sub_1_remove_ids = set([word_id for word_id, kld in enumerate(comp_dict['kld_' + sub_1]) if word_id not in noncommon_words and kld > kld_threshold])
        sub_2_remove_ids = set([word_id for word_id, kld in enumerate(comp_dict['kld_' + sub_2]) if word_id not in noncommon_words and kld > kld_threshold])
        all_remove_ids = sub_1_remove_ids.union(sub_2_remove_ids)

        union_all_remove_ids = union_all_remove_ids | all_remove_ids

        print(sub_1 + ' has ' + str(len(sub_1_remove_ids)) + ' words with KLD > ' + str(kld_threshold))
        print(sub_2 + ' has ' + str(len(sub_2_remove_ids)) + ' words with KLD > ' + str(kld_threshold))
        print()

        remaining_jsd = sum([jsd for word_id, jsd in enumerate(comp_dict['jsd']) if word_id not in noncommon_words and word_id not in all_remove_ids])
        remaining_jsd_dict[(sub_1, sub_2)] = remaining_jsd
        print('remaining jsd between ' + sub_1 + ' and ' + sub_2 + ' is ' + str(round(remaining_jsd, 6)))

        sub_comparison_fpath = os.path.join(analysis_dir, sub_1 + '_' + sub_2 + '_kld_threshold_' + str(kld_threshold)[2:] + '.csv')
        with open(sub_comparison_fpath, 'w', encoding='utf-8', newline='') as outfile:
            fwriter = csv.writer(outfile)
            fwriter.writerow(['word_id', 'word', 'subreddit', 'kld_subreddit', 'jsd'])

            for word_id in sub_1_remove_ids:
                fwriter.writerow([word_id, dictionary[word_id], sub_1, comp_dict['kld_' + sub_1][word_id], comp_dict['jsd'][word_id]])

            for word_id in sub_2_remove_ids:
                fwriter.writerow([word_id, dictionary[word_id], sub_2, comp_dict['kld_' + sub_2][word_id], comp_dict['jsd'][word_id]])
        print('\n')

    with open(os.path.join(cons.lexremoval_dir, '2_pairwise_kld_' + str(kld_threshold)[2:] + '_words.csv'),
              'w', encoding='utf-8', newline='') as outfile:
        fwriter = csv.writer(outfile)
        fwriter.writerow(['word_id', 'word'])
        for wid in union_all_remove_ids:
            fwriter.writerow([wid, dictionary[wid]])

    with open(os.path.join(cons.lexremoval_dir, 'remaining_jsd_pairwise_kld_' + str(kld_threshold)[2:] + '_words.csv'),
              'w', encoding='utf-8',
              newline='') as outfile:
        fwriter = csv.writer(outfile)
        fwriter.writerow(['subreddit_1', 'subreddit_2', 'jsd'])
        for sub_1, sub_2 in remaining_jsd_dict.keys():
            fwriter.writerow([sub_1, sub_2, remaining_jsd_dict[(sub_1, sub_2)]])

    return union_all_remove_ids


if __name__ == '__main__':
    subreddit_list = cons.subreddit_list

    dictionary = gensim.corpora.dictionary.Dictionary.load(os.path.join(cons.corpora_dir, 'dictionary.dict'))
    original_vocab_size = len(dictionary)
    print('original post-processed vocabulary size is ' + str(original_vocab_size) + ' types.')

    # Determine which words from the vocabulary do not occur in every subreddit's corpus. To do this, go through
    # word counts from each subreddit. If a word ever has a count of 0, then it's out.
    noncommon_words_fpath = os.path.join(cons.lexremoval_dir, '1_uncommon_words.csv')
    noncommon_words = get_uncommon_words(noncommon_words_fpath, subreddit_list, dictionary)
    available_word_ids = set(word_id for word_id in dictionary.keys() if word_id not in noncommon_words)

    # Look at highest per-word KLD values between each pair of subreddits.
    kld_threshold_list = [0.0001, 0.0005, 0.001]
    for kld_threshold in kld_threshold_list:
        pairwise_dir = cons.makedir(os.path.join(cons.lexremoval_dir,
                                                 'pairwise_kld_threshold_' + str(kld_threshold)[2:]))
        remove_ids = lexicon_analysis_kld_threshold(pairwise_dir,
                                                    noncommon_words,
                                                    dictionary,
                                                    kld_threshold)
        print('with threshold of ' + str(kld_threshold) + ', ' + str(len(remove_ids)) + ' words removed.')

