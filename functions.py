"""
# Zachary K. Stine
# Updated 2020-12-10
#
# Collection of functions that are used across multiple scripts in this project.
"""

import tldextract
import re
import modified_casual_tokenizer
import cons
import os
import csv
import ast
import string
import gensim
import numpy as np

# Regular expression pattern for URLs taken from NLTK's TweetTokenizer().
URL_RE = re.compile(modified_casual_tokenizer.URLS, re.VERBOSE | re.I | re.UNICODE)

# Modified version of NLTK's TweetTokenizer() that preserves Reddit usernames and subreddit names.
REDDIT_TKNZR = modified_casual_tokenizer.RedditTokenizer(preserve_case=False)

PUNCT = set([punct_string for punct_string in string.punctuation] + ["‘", "’", '“', '”', "...", "..", "—", "–"])

# These are special characters that follow certain emoji, which are blank yet not treated as blank space by the
# tokenizer nor with the built-in function, strip(), and so must be dealt with directly otherwise they will get
# glued onto the front of the next token.
VARIATION_SELECTORS = [b'\xef\xb8\x80',
                       b'\xef\xb8\x81',
                       b'\xef\xb8\x82',
                       b'\xef\xb8\x83',
                       b'\xef\xb8\x84',
                       b'\xef\xb8\x85',
                       b'\xef\xb8\x86',
                       b'\xef\xb8\x87',
                       b'\xef\xb8\x88',
                       b'\xef\xb8\x89',
                       b'\xef\xb8\x8a',
                       b'\xef\xb8\x8b',
                       b'\xef\xb8\x8c',
                       b'\xef\xb8\x8d',
                       b'\xef\xb8\x8e',
                       b'\xef\xb8\x8f']
VAR_SELECTORS_PATTERN = r'[{}]'.format(''.join(var_sel.decode('utf-8') for var_sel in VARIATION_SELECTORS))
PROBLEM_CHARS = ['\udc20']


def format_url(match_object):
    """
    :param match_object: URL regular expression match.
    :return: The domain and suffix of the URL. E.g., 'https://github.com' becomes 'github.com'.
    """
    #print(match_object)
    ext = tldextract.extract(match_object.group(0))
    return '.'.join([ext.domain, ext.suffix])


def tokenize(text, stops=None, min_chars=1):
    """
    :param text: String to be tokenized
    :param stops: List of strings that are removed (stopped) from the tokenized text.
    :param min_chars: Minimum number of characters a token must have, otherwise it is removed.
    :return: List of strings (tokens)
    """
    # First, check text for any URLs so they can be formatted in such a way that their domains are preserved as tokens.
    url_safe_text = re.sub(URL_RE, format_url, text)

    # NLTK's TweetTokenizer has some useful stuff for social media text in general, so it's used here in a slightly
    # modified form. See modified_casual_tokenizer.py for specification.
    tokens = REDDIT_TKNZR.tokenize(url_safe_text)

    # Remove single-character non-alphabetic tokens. Serves to remove punctuation that is not part of emoji nor
    # combined with other characters.
    no_single_punct = [t.strip() for t in tokens if t.strip() not in PUNCT]

    # Keep tokens that meet the length criteria specified by min_chars.
    clean_tokens = [t for t in no_single_punct if len(t) >= min_chars and t not in PROBLEM_CHARS]

    no_times = []
    for t in clean_tokens:
        if len(t) >= 4 and t[0].isdigit() and t[1] == ':' and t[2].isdigit() and t[3].isdigit():
            pass
        else:
            no_times.append(t)

    # If a list of stopwords is provided, remove them.
    if stops is not None:
        return [t for t in no_times if t not in stops]
    else:
        return no_times


def read_submission(subm_fpath):
    """
    :param subm_fpath: File path of a submisison.
    :return: Returns list of strings with some basic formatting adjustments.
    """
    #print('read_submission: ' + str(subm_fpath))
    subm_thread = []
    with open(subm_fpath, 'r', encoding='utf-8') as infile:
        dict_reader = csv.DictReader(infile)

        for comm_dict in dict_reader:
            converted_text = ast.literal_eval(comm_dict['text']).decode('utf-8')
            no_curly_quotes = converted_text.replace("‘", "'").replace("’", "'").replace('“', '"').replace('”', '"')
            no_variation_selectors = re.sub(VAR_SELECTORS_PATTERN, ' ', no_curly_quotes)

            comm_dict['text'] = no_variation_selectors

            subm_thread.append(comm_dict)

    return subm_thread


def read_word_frequencies(fpath, dictionary):
    """
    Reads in word frequencies and proportions from specified file path.
    """
    word_counts = np.zeros(len(dictionary))
    word_props = np.zeros(len(dictionary))

    with open(fpath, 'r', encoding='utf-8') as infile:
        freader = csv.DictReader(infile)

        for row in freader:

            wid = int(row['word_id'])
            wcount = float(row['word_count'])
            wprop = float(row['word_proportion'])

            word_counts[wid] = wcount
            word_props[wid] = wprop

    return word_counts, word_props


def write_word_freqs_and_props(fpath, dictionary, corpus):
    """
    Writes word frequencies and proportions.
    """
    word_counts = np.zeros(len(dictionary))

    for bow_doc in corpus:
        for (word_id, count) in bow_doc:
            word_counts[word_id] += count

    total_tokens = np.sum(word_counts)
    word_proportions = np.divide(word_counts, total_tokens, dtype=np.float64)

    with open(fpath, 'w', newline='', encoding='utf-8') as ofile:
        fwriter = csv.writer(ofile)
        fwriter.writerow(['word_id', 'word', 'word_count', 'word_proportion'])

        for word_id in dictionary:
            fwriter.writerow([word_id, dictionary[word_id], word_counts[word_id], word_proportions[word_id]])

    return word_counts, word_proportions


def read_corpus_metadata(fpath):
    """
	Reads in corpus metadata from the specified file path.
	"""
    data_list = []

    with open(fpath, 'r', encoding='utf-8') as infile:
        freader = csv.DictReader(infile)

        for row_dict in freader:
            data_list.append(row_dict)

    return data_list


def read_lexical_comparison_file(fpath, vocab_size, subreddit_name, comparison_name):
    """
    Reads in word-level comparison statistics from specified file path.
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



