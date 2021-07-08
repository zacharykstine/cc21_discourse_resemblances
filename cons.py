"""
# Zachary K. Stine, 2020-11-24
#
# This file defines some constants that are used across the project scripts.
"""

import os


def makedir(path):
    """
    If the directory specified by path does not exist, it is created. In both cases the directory path is returned.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


# Directories used throughout the project:
project_dir = os.getcwd()
data_dir = os.path.join(os.path.dirname(project_dir), '0_data')
corpora_dir = makedir(os.path.join(project_dir, '1_corpora'))
lda_dir = makedir(os.path.join(project_dir, '2_lda'))
lexcomp_dir = makedir(os.path.join(project_dir, '3_lexical_comparisons'))
lexremoval_dir = makedir(os.path.join(project_dir, '4_lexicon_removal'))

# The target subreddit being investigated.
target_sub = 'spirituality'

# The set of subreddits that the target subreddit will be compared against:
comparison_sub_list = open(os.path.join(project_dir, 'comparison_subreddits.txt'), 'r').read().lower().splitlines()

# All subreddits in a single list:
subreddit_list = [target_sub] + comparison_sub_list

# The set of reddit users whose comments and posts will be ignored. Used here for common bot accounts.
ignore_users = open(os.path.join(project_dir, 'ignore_users.txt'), 'r').read().lower().splitlines()

# List of highly frequent words that are removed in preprocessing.
stoplist = open(os.path.join(project_dir, 'stoplist.txt'), 'r').read().lower().splitlines()

# The maximum UTC day for which data is used from each subreddit. Format is YYYYMMDD.
max_date = 20191231
