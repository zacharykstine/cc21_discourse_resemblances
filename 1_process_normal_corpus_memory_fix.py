"""
# Zachary K. Stine, 2020-12-17
# 
# This script contains preprocessing steps used. There are two preprocessing schemes present here: First, a straightforward 
# scheme in which one document is defined as the submission text and all accompanying comments, typically referred to in the code
# as the "normal corpus." Second, a complicated scheme in which long comment threads are broken up into multiple documents in such a way
# that mostly preserves the branching structure of comment threads so that all comments in the same document should be relevant to each
# other, which is referred to in the code as the "split corpus." Note that this second preprocessing scheme was not included in the 
# final paper. 
#
# The following need to be in place prior to running this script:
#   1) a list of stopwords (see cons.py)
#   2) a list of users whose text will be ignored (see cons.py)
"""

import os
import cons
import functions
import csv
import gensim
from collections import OrderedDict
import ast
import datetime
import pickle


class NormalCorpus(object):

    def __init__(self, corpus_data_fpath, dictionary, stoplist, ignore_users):
        self.corpus_data_fpath = corpus_data_fpath
        #self.submission_fpaths = submission_fpaths
        self.dictionary = dictionary
        self.stoplist = stoplist
        self.ignore_users = ignore_users

    def __iter__(self):
        with open(self.corpus_data_fpath, 'r', encoding='utf-8') as infile:
            freader = csv.DictReader(infile)
            expected_index = 0

            for subm_data in freader:
                if expected_index != int(subm_data['corpus_index']):
                    print('expected corpus index of ' + str(expected_index) + ' but file has index ' + subm_data['corpus_index'] + '.')

                subm_fpath = subm_data['submission_path']

                comment_list = functions.read_submission(subm_fpath)

                submission_tokens = [tokens for comment in comment_list for tokens in
                                     functions.tokenize(comment['text'], stops=self.stoplist, min_chars=3) if
                                     comment['author_name'].lower() not in self.ignore_users]

                yield self.dictionary.doc2bow(submission_tokens)

                expected_index += 1


class NormalSplitCorpus(object):

    def __init__(self, split_data_path, dictionary, stoplist, ignore_users):
        self.split_data_path = split_data_path
        self.dictionary = dictionary
        self.stoplist = stoplist
        self.ignore_users = ignore_users

    def __iter__(self):

        # Second, read in dictionary for split data
        with open(self.split_data_path, 'r', encoding='utf-8') as infile:
            freader = csv.DictReader(infile)

            n_docs = 0

            for split_doc_dict in freader:

                subm_fpath = split_doc_dict['submission_path']
                #print(subm_fpath)
                comment_list = functions.read_submission(subm_fpath)

                doc_list = ast.literal_eval(split_doc_dict['split_docs'])

                for docs in doc_list:
                    n_docs += 1

                    split_doc_tokens = [tokens for comment in comment_list for tokens in
                                        functions.tokenize(comment['text'], stops=self.stoplist, min_chars=3) if
                                        comment['author_name'].lower() not in self.ignore_users and comment['id'] in docs]

                    yield self.dictionary.doc2bow(split_doc_tokens)
        #print('total number of split documents: ' + str(n_docs))


def make_dictionary(submission_fpaths, stoplist, ignore_users, n_below=5, n_above=100.0, n_keep=15000):
    """
    From a collection of documents, construct a vocabulary in the from of gensim's Dictionary object with some light preprocessing.
    """
    dictionary = gensim.corpora.Dictionary()

    for subm_fpath in submission_fpaths:

        comment_list = functions.read_submission(subm_fpath)

        submission_tokens = [tokens for comment in comment_list for tokens in
                             functions.tokenize(comment['text'], stops=stoplist, min_chars=3) if
                             comment['author_name'].lower() not in ignore_users]

        dictionary.add_documents([submission_tokens])

    for stop_word in stoplist:
        if stop_word in dictionary.token2id:
            #print(stop_word + ' is in dictionary.')
            dictionary.filter_tokens(bad_ids=[dictionary.token2id[stop_word]])

    dictionary.filter_extremes(no_below=n_below, no_above=n_above, keep_n=n_keep)

    return dictionary


def merge_dictionaries(dict_list):
    """
    dict_list is a list of Gensim dictionary objects to be merged into a single dictionary, which is then returned. This becomes the 
	final vocabulary for the whole corpus.
    """

    # Use the first dictionary object in dictionary_list as the dictionary that all others will merge into.
    merged_dict = dict_list[0]

    # Iterate through the remaining dictionaries and merge each into merged_dictionary.
    for other_dictionary in dictionary_list[1:]:

        # No need to do anything with the transformer since we haven't made any corpus objects so far.
        transformer = merged_dict.merge_with(other_dictionary)

    return merged_dict


def get_thread_tree(raw_doc, stoplist, ignore_users, dictionary):
    """ Used in the preprocessing scheme where comment threads are split up"""
	
    thread_dict = {}
    submission_id = raw_doc[0]['id']
    thread_size = 0

    unfinished_comment_ids = []
    for comment_dict in raw_doc:
        comment_id = comment_dict['id']
        parent_id = comment_dict['parent_id'][3:]  # get rid of prefix (e.g., t1_xyz -> xyz).

        # If this is a bot we want to ignore, then keep it to preserve tree structure, but don't include its tokens.
        if comment_dict['author_name'].lower() in ignore_users:
            comment_tokens = []
        else:
            comment_tokens = functions.tokenize(comment_dict['text'], stoplist, min_chars=3)

        comment_num_tokens = sum(token_count for (token_id, token_count) in dictionary.doc2bow(comment_tokens))
        thread_size += comment_num_tokens

        thread_dict[comment_id] = {'parent_id': parent_id,
                                   'len': comment_num_tokens,
                                   'children_ids': [],
                                   'depth': -1}

        if comment_dict['type'] == 'comment':
            # Add this comment's id to its parent's list of children ids.
            # First check if parent_id is already in thread_dict.
            if parent_id in thread_dict.keys():
                thread_dict[parent_id]['children_ids'].append(comment_id)
                thread_dict[comment_id]['depth'] = thread_dict[parent_id]['depth'] + 1
            else:
                print(comment_id + ' added to unfinished_comment_ids.')
                unfinished_comment_ids.append(comment_id)

        else:
            assert comment_id == submission_id
            thread_dict[comment_id]['depth'] = 0

    while unfinished_comment_ids:
        comment_id = unfinished_comment_ids.pop(0)
        parent_id = thread_dict[comment_id]['parent_id']

        if parent_id in thread_dict.keys():
            thread_dict[parent_id]['children_ids'].append(comment_id)
            thread_dict[comment_id]['depth'] = thread_dict[parent_id]['depth'] + 1
        else:
            # parent id may have been deleted and removed. force this to be a top-level comment.
            print(comment_id + ' has no parent and so its parent will be submisison id ' + str(submission_id))
            #print('previous parent: ' + thread_dict[])
            thread_dict[comment_id]['parent_id'] = submission_id
            thread_dict[submission_id]['children_ids'].append(comment_id)
            thread_dict[comment_id]['depth'] = thread_dict[submission_id]['depth'] + 1

    return thread_dict, submission_id, thread_size


def split_thread(thread_tree, root_id, min_size, max_size):
    """ Used in the preprocessing scheme where comment threads are split up"""
	
    # Create first set of partitions: root and immediate children.
    partitions, under_min_nodes, over_max_nodes = create_partitions(thread_tree, root_id, min_size, max_size)

    # Continue to further split any partitions that are too big and can be split.
    while over_max_nodes:
        # Get first node in line that is too big.
        big_node = over_max_nodes.pop(0)

        # Partition it further.
        fixed_partitions, new_small_nodes, new_big_nodes = create_partitions(thread_tree, big_node, min_size, max_size)

        # Add any new nodes that are too big or too small to the appropriate list.
        over_max_nodes += new_big_nodes
        under_min_nodes += new_small_nodes

        # Remove original node from partitions, now that it has been broken up into several smaller pieces.
        del partitions[big_node]

        # Add the new partitions.
        for new_p in fixed_partitions:
            partitions[new_p] = fixed_partitions[new_p]

    # Now that the thread tree has been broken down into smallest necessary pieces, combine pieces so that they are all
    # above minimum size.
    # First, sort remaining nodes so that the deepest are first in line.
    depth_sorted_open_nodes = sorted(under_min_nodes, key=lambda n: thread_tree[n]['depth'], reverse=True)

    while depth_sorted_open_nodes:
        #print('open nodes: ' + str(depth_sorted_open_nodes) + '\n')

        small_node = depth_sorted_open_nodes.pop(0)
        #print('current small node: ' + small_node + '\n')

        # if the only remaining node is the root (ie, submission ID), then it should be merged with its smallest child).
        if small_node == root_id:
            smallest_children = sorted([n for n in thread_tree[root_id]['children_ids'] if n in partitions.keys()],
                                       key=lambda n: partitions[n]['len'])
            if len(smallest_children) > 0:
                merge(small_node, smallest_children[0], partitions)
            else:
                #print('submission ' + str(small_node) + ' does not appear to have a child available? ')
                # stick the remaining node with the shallowest and smallest of any node in partitions.
                all_sorted_partition_nodes = sorted(partitions.keys(),
                                                    key=lambda n: (thread_tree[n]['depth'], partitions[n]['len']))
                merge(small_node, all_sorted_partition_nodes[0], partitions)
            break

        parent_node = thread_tree[small_node]['parent_id']

        if parent_node in depth_sorted_open_nodes:
            # merge small_node and parent_node
            merge(parent_node, small_node, partitions)

            if partitions[parent_node]['len'] >= min_size:
                # safe to remove parent from depth_sorted_open_nodes
                depth_sorted_open_nodes.remove(parent_node)

        else:
            # If node doesn't have an open parent, check for open siblings.
            siblings = [n for n in depth_sorted_open_nodes if thread_tree[n]['parent_id'] == parent_node]

            # If no open siblings, just stick it with parent. Otherwise, combine with siblings.
            if len(siblings) == 0:
                merge(parent_node, small_node, partitions)

            else:
                while siblings:
                    # Get first sibling and merge.
                    sib_node = siblings.pop(0)
                    merge(sib_node, small_node, partitions)

                    # If the current small_node isn't the original one which was popped from the front of
                    # depth_sorted_open_nodes, then it will need to be removed here.
                    if small_node in depth_sorted_open_nodes:
                        depth_sorted_open_nodes.remove(small_node)

                    # For next iteration in while loop, small_node will be updated to the current sib_node.
                    small_node = sib_node

                    # If the merged partition is big enough, remove it from the list of open nodes.
                    if partitions[sib_node]['len'] >= min_size:
                        depth_sorted_open_nodes.remove(sib_node)
                        break

    return [partitions[node]['nodes'] for node in partitions.keys()]


def create_partitions(tree, root, min_size, max_size):
    """ Used in the preprocessing scheme where comment threads are split up"""
	
    small_nodes = []
    big_nodes = []
    parts = {root: {'nodes': [root], 'len': tree[root]['len']}}
    if tree[root]['len'] < min_size:
        small_nodes.append(root)

    for child in tree[root]['children_ids']:
        # add every subbranch to this one
        branch_size = 0
        all_branch_ids = []
        id_queue = [child]

        while id_queue:
            comm_id = id_queue.pop(0)
            all_branch_ids.append(comm_id)
            id_queue += tree[comm_id]['children_ids']
            branch_size += tree[comm_id]['len']

        parts[child] = {'nodes': all_branch_ids, 'len': branch_size}
        if branch_size > max_size and len(all_branch_ids) > 1:
            big_nodes.append(child)
        elif branch_size < min_size:
            small_nodes.append(child)

    return parts, small_nodes, big_nodes


def merge(node_1, node_2, partitions):
    """ Used in the preprocessing scheme where comment threads are split up"""

    partitions[node_1]['nodes'] += partitions[node_2]['nodes']
    partitions[node_1]['len'] += partitions[node_2]['len']

    del partitions[node_2]


def write_metadata_and_split_data(submission_paths, dictionary, stoplist, ignore_users, subreddit_name,
                 metadata_fpath, split_data_fpath,
                 min_tokens=20, min_split_size=100, max_split_size=350):
    """
	Function writes metadata for all processed documents, which includes the following for each submission:
    1) index within the corpus
    2) subreddit
    3) submission_id
    4) submission date
    5) submission score
    6) document length (# of tokens)
    7) submission URL
    8) submission URL using Reddit's old UI.
    9) how the comment IDs should be grouped for split documents used in model training.
    """
	
    current_index = 0

    with open(metadata_fpath, 'w', encoding='utf-8', newline='') as metadata_outfile:
        fwriter_metadata = csv.DictWriter(metadata_outfile, ['corpus_index',
                                                             'subreddit',
                                                             'submission_id',
                                                             'submission_date',
                                                             'submission_score',
                                                             'doc_length',
                                                             'num_split_docs',
                                                             'submission_path',
                                                             'submission_url',
                                                             'submission_url_old'])
        fwriter_metadata.writeheader()

        with open(split_data_fpath, 'w', encoding='utf-8', newline='') as splitdata_outfile:
            fwriter_splitdata = csv.DictWriter(splitdata_outfile,
                                               ['corpus_index',
                                                'subreddit',
                                                'submission_id',
                                                'submission_path',
                                                'split_docs'])
            fwriter_splitdata.writeheader()

            for subm_fpath in submission_paths:
                comment_list = functions.read_submission(subm_fpath)
                thread_tree, submission_id, document_length = get_thread_tree(comment_list, stoplist, ignore_users, dictionary)
                #print('  ' + submission_id + ' being processed in write_metatdata_and_split_data.')

                if document_length > min_tokens:
                    corpus_index = current_index
                    current_index += 1

                    # Determine if split is needed for constructing the split corpus.
                    if document_length > max_split_size:

                        # Split the submission up into smaller pieces.
                        split_docs = split_thread(thread_tree, submission_id, min_split_size, max_split_size)

                        num_split_docs = len(split_docs)

                    else:
                        # No need to figure out a split.
                        split_docs = [[comm['id'] for comm in comment_list]]
                        num_split_docs = 1

                    # Write split data to file.
                    submission_splitdata = {'corpus_index': corpus_index,
                                            'subreddit': subreddit,
                                            'submission_id': submission_id,
                                            'submission_path': subm_fpath,
                                            'split_docs': split_docs}
                    fwriter_splitdata.writerow(submission_splitdata)


                    # Write submission metadata to file
                    submission_date = int(os.path.basename(subm_fpath).split('_')[0])
                    submission_id = os.path.basename(subm_fpath).split('_')[1][:-4]
                    submission_score = comment_list[0]['score']
                    base_url = 'reddit.com/r/' + subreddit_name + '/comments/' + submission_id

                    submission_metadata = {'corpus_index': corpus_index,
                                           'subreddit': subreddit_name,
                                           'submission_id': submission_id,
                                           'submission_date': submission_date,
                                           'submission_score': submission_score,
                                           'doc_length': document_length,
                                           'num_split_docs': num_split_docs,
                                           'submission_path': subm_fpath,
                                           'submission_url': 'www.' + base_url,
                                           'submission_url_old': 'old.' + base_url}

                    
                    fwriter_metadata.writerow(submission_metadata)


if __name__ == '__main__':

    dictionary_name = 'dictionary.dict'

    subreddit_list = cons.subreddit_list

    # ---------------------------------------------------------------------------------------------------------------- #
    # 1) Define the vocabulary of each subreddit separately in the form of a Gensim dictionary object.                 #
    # ---------------------------------------------------------------------------------------------------------------- #

    # Each subreddit-specific dictionary is added to dictionary_list.

    dictionary_list = []

    print(str(cons.stoplist))
    print('\n\n')

    for subreddit in subreddit_list:
        current_time = datetime.datetime.now()
        print(str(current_time) + ' : starting dictionary for r/' + subreddit + '.')

        # Path to directory with subreddit's raw data files:
        submission_dir = os.path.join(cons.data_dir, 'r_' + subreddit, 'threads')

        # Put each document path in list if its submission was authored on or before max_date (defined in cons.py).
        submission_fnames = [os.path.join(submission_dir, f) for f in os.listdir(submission_dir) if
                             int(os.path.basename(f).split('_')[0]) <= cons.max_date]

        # Create Gensim dictionary object that defines this subreddit's vocabulary.
        subreddit_dictionary = make_dictionary(submission_fnames,
                                               cons.stoplist,
                                               cons.ignore_users,
                                               n_below=50,
                                               n_keep=10000)
        print('dictionary for ' + subreddit + ' has ' + str(len(subreddit_dictionary)) + ' words.')
        dictionary_list.append(subreddit_dictionary)

        for stop in cons.stoplist:
            if stop in subreddit_dictionary.token2id:
                print(stop + ' in dictionary. ID is ' + str(subreddit_dictionary.token2id[stop]))

        print('-'*100)

    # ---------------------------------------------------------------------------------------------------------------- #
    # 2) Merge each subreddit-specific dictionary into a single one for consistent word<-->ID mapping.                 #
    # ---------------------------------------------------------------------------------------------------------------- #
    merged_dictionary = merge_dictionaries(dictionary_list)
    merged_dictionary.save(os.path.join(cons.corpora_dir, dictionary_name))

    # ---------------------------------------------------------------------------------------------------------------- #
    # 3) Create two sets of Gensim corpus objects and metadata CSVs for each subreddit.                                #
    # ---------------------------------------------------------------------------------------------------------------- #
    merged_dictionary = gensim.corpora.Dictionary.load(os.path.join(cons.corpora_dir, dictionary_name))

    print('\n\n')

    for subreddit in subreddit_list:
        current_time = datetime.datetime.now()

        print(str(current_time) + ' : starting corpus for r/' + subreddit + '.')

        # Path to directory with subreddit's raw data files:
        submission_dir = os.path.join(cons.data_dir, 'r_' + subreddit, 'threads')

        # Put each document path in list if its submission was authored on or before max_date (defined in cons.py).
        submission_fnames = [os.path.join(submission_dir, f) for f in os.listdir(submission_dir) if
                             int(os.path.basename(f).split('_')[0]) <= cons.max_date]

        # Confirm that documents are sorted from oldest to newest.
        sorted_submission_fnames = sorted(submission_fnames, key=lambda f: int(os.path.basename(f).split('_')[0]))

        # 3a) First, create metadata CSV and Gensim corpus object.
        normal_corpus_dir = cons.makedir(os.path.join(cons.corpora_dir, subreddit, 'normal_corpus'))
        metadata_fpath = os.path.join(normal_corpus_dir, 'normal_corpus_metadata.csv')
        split_data_fpath = os.path.join(normal_corpus_dir, 'normal_corpus_split_data.csv')

        write_metadata_and_split_data(sorted_submission_fnames,
                                      merged_dictionary,
                                      cons.stoplist,
                                      cons.ignore_users,
                                      subreddit,
                                      metadata_fpath,
                                      split_data_fpath,
                                      min_tokens=35,
                                      min_split_size=100,
                                      max_split_size=350)

        subreddit_normal_corpus = NormalCorpus(metadata_fpath, merged_dictionary, cons.stoplist, cons.ignore_users)
        corpus_fpath = os.path.join(normal_corpus_dir, 'normal_corpus')
        gensim.corpora.MmCorpus.serialize(corpus_fpath, subreddit_normal_corpus)

        subreddit_split_corpus = NormalSplitCorpus(split_data_fpath,
                                                   merged_dictionary,
                                                   cons.stoplist,
                                                   cons.ignore_users)
        split_corpus_fpath = os.path.join(normal_corpus_dir, 'split_normal_corpus')
        gensim.corpora.MmCorpus.serialize(split_corpus_fpath, subreddit_split_corpus)

        print('-'*100)
