"""
# Zachary Kimo Stine, 2021-01-19
# 
# Remake the original corpus, but exclude a selection of word IDs that act as signals of one subreddit over another.
# In this script, this is done using a word-level KL divergence threshold of 0.001 bits.
"""

import cons
import os
import csv
import gensim


class LexicallyReducedCorpus(object):

    def __init__(self, original_corpus_fpath, remove_word_ids):
        self.original_corpus_fpath = original_corpus_fpath
        self.remove_word_ids = remove_word_ids
        self.doc_lengths = []

    def __iter__(self):
        original_corpus = gensim.corpora.MmCorpus(self.original_corpus_fpath)

        for old_bow_doc in original_corpus:
            new_bow_doc = [(w_id, w_count) for (w_id, w_count) in old_bow_doc if w_id not in self.remove_word_ids]
            self.doc_lengths.append(sum([w_count for (w_id, w_count) in new_bow_doc]))
            yield new_bow_doc


def get_removal_words_from_files(file_list):
    """
    Reads words to be removed from the relevant files specified in file_list.
    """
    removal_word_set = set()

    for fpath in file_list:
        with open(fpath, 'r', encoding='utf-8') as infile:
            freader = csv.DictReader(infile)
            for row in freader:
                removal_word_set.add(int(row['word_id']))
    return removal_word_set


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

    # Read in files to create list of "cultural lexicon" terms.
    remove_words_files = [os.path.join(cons.lexremoval_dir, '1_uncommon_words.csv'),
                          os.path.join(cons.lexremoval_dir, '2_pairwise_kld_001_words.csv')]

    remove_words = get_removal_words_from_files(remove_words_files)
    print(str(len(remove_words)) + ' words will be removed from the vocabulary.')

    # For each subreddit, create a new corpus file which is the same as the original, except that any word in the
    # list of words to be removed, is taken out.
    for subreddit in subreddit_list:
        normal_corpus_fpath = os.path.join(cons.corpora_dir, subreddit, 'normal_corpus', 'normal_corpus')
        normal_metadata_fpath = os.path.join(cons.corpora_dir, subreddit, 'normal_corpus', 'normal_corpus_metadata.csv')

        reduced_corpus_dir = cons.makedir(os.path.join(cons.corpora_dir, subreddit, 'reduced_corpus_001'))
        reduced_corpus_fpath = os.path.join(reduced_corpus_dir, 'reduced_corpus_001')
        reduced_metadata_fpath = os.path.join(reduced_corpus_dir, 'reduced_corpus_001_metadata.csv')

        # Create Gensim corpus object for the modified corpus and write to file.
        reduced_corpus = LexicallyReducedCorpus(normal_corpus_fpath, remove_words)
        gensim.corpora.MmCorpus.serialize(reduced_corpus_fpath, reduced_corpus)

        # Write metadata file for the modified corpus.
        normal_corpus_metadata = read_corpus_metadata(normal_metadata_fpath)
        reduced_corpus_doc_lengths = reduced_corpus.doc_lengths
        with open(reduced_metadata_fpath, 'w', encoding='utf-8', newline='') as outfile:
            fwriter = csv.DictWriter(outfile,
                                     ['corpus_index', 'subreddit', 'submission_id', 'submission_date',
                                      'submission_score', 'doc_length', 'reduced_doc_length', 'doc_length_difference',
                                      'num_split_docs', 'submission_path', 'submission_url', 'submission_url_old'])
            fwriter.writeheader()

            for doc_index, doc_data in enumerate(normal_corpus_metadata):
                assert doc_index == int(doc_data['corpus_index'])

                reduced_doc_length = reduced_corpus_doc_lengths[doc_index]
                doc_length_difference = int(doc_data['doc_length']) - reduced_doc_length

                doc_data['reduced_doc_length'] = reduced_doc_length
                doc_data['doc_length_difference'] = doc_length_difference

                fwriter.writerow(doc_data)