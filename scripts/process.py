import os, sys
import gzip
import json
import argparse
sys.path.append(os.getcwd())

import pandas as pd
import numpy as np

import config
import text_process


def get_df(path):
    """ Apply raw data to pandas DataFrame. """
    i = 0
    df = {}
    g = gzip.open(path, 'rb')
    for line in g:
        df[i] = eval(line)
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def extraction(meta_path, review_df, stop_words, count):
    """ Extract useful information. """
    with gzip.open(meta_path, 'rb') as g:
        categories = {}
        for line in g:
            line = eval(line)
            categories[line['asin']] = line['categories']

    # filter each user have at least 10 transactions
    r_lens = review_df.groupby('reviewerID').size()
    review_df = review_df[np.in1d(review_df.reviewerID, 
                            r_lens[r_lens >= 10].index)]
    review_df = review_df.reset_index(drop=True)

    queries, reviews = [], []
    for i in range(len(review_df)):
        asin = review_df['asin'][i]
        review = review_df['reviewText'][i]
        category = categories[asin]

        # process queries
        qs = map(text_process._remove_dup, 
                    map(text_process._remove_char, category))
        qs = [[w for w in q if w not in stop_words] for q in qs]

        # return the query with max length
        q = max(qs, key=len) 
        
        # process reviews
        review = text_process._remove_char(review)
        review = [w for w in review if w not in stop_words]

        queries.append(q)
        reviews.append(review)

    review_df['query_'] = queries 

    # filtering words counts less than count
    reviews = text_process._filter_words(reviews, count)
    review_df['reviewText'] = reviews
    return review_df


def split_data(df):
    filters = []
    user_length = df.groupby('reviewerID').size().tolist()

    for user in range(len(user_length)):
        for _ in range((user_length[user] - 1)):
            filters.append('Train')
        filters.append('Test')
    
    df['filter'] = filters
    df_train = df[df['filter'] == 'Train']
    df_test = df[df['filter'] == 'Test']

    return (df.reset_index(drop=True), 
            df_train.reset_index(drop=True), 
            df_test.reset_index(drop=True))


def get_user_bought(train_set):
    """ obtain the products each user has bought before test. """
    user_bought = {}
    for i in range(len(train_set)):
        user = train_set['reviewerID'][i]
        item = train_set['asin'][i]
        if user not in user_bought:
            user_bought[user] = []
        user_bought[user].append(item)
    return user_bought


def rm_test(df, df_test):
    """ Remove test review data and remove duplicate. """
    df = df.reset_index(drop=True)
    reviews = [df['reviewText'][i] if df['filter'][i] == 'Train'
                                else '[]' for i in range(len(df))]                               
    df['reviewText'] = reviews
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review_file', 
        type=str,
        default='reviews_Toys_and_Games_5.json.gz',
        help="5 core review file")
    parser.add_argument('--meta_file', 
        type=str,
        default='meta_Toys_and_Games.json.gz',
        help="meta data file for the corresponding review file")
    parser.add_argument('--count', 
        type=int, 
        default=5,
        help="remove the words number less than count")
    FLAGS = parser.parse_args()


    ###################################### PREPARE PATHS ####################################
    if not os.path.exists(config.processed_path):
        os.makedirs(config.processed_path)

    stop_path = config.stop_file
    meta_path = os.path.join(config.main_path, FLAGS.meta_file)
    review_path = os.path.join(config.main_path, FLAGS.review_file)

    review_df = get_df(review_path)
    stop_df = pd.read_csv(stop_path, header=None, names=['stopword'])
    stop_words = set(stop_df['stopword'].unique())


    df = extraction(meta_path, review_df, stop_words, FLAGS.count)
    df = df.drop(['reviewerName', 'reviewTime', 
                'helpful', 'summary', 'overall'], axis=1)
    print("The number of {} users is {:d}; items is {:d}; feedbacks is {:d}.".format(
        config.dataset, len(df.reviewerID.unique()), len(df.asin.unique()), len(df)))

    df = df.sort_values(by=['reviewerID', 'unixReviewTime'])
    df, df_train, df_test = split_data(df) 

    user_bought = get_user_bought(df_train)
    json.dump(user_bought, open(os.path.join(config.processed_path, 
            '{}_user_bought.json'.format(config.dataset)), 'w'))
 
    df = rm_test(df, df_test) # remove the reviews from test set.

    df = df.drop(['unixReviewTime', 'filter'], axis=1)
    df_train = df_train.drop(['unixReviewTime', 'filter'], axis=1)
    df_test = df_test.drop(['unixReviewTime', 'filter'], axis=1)

    df.to_csv(os.path.join(
        config.processed_path, '{}_full.csv'.format(config.dataset)), index=False)
    df_train.to_csv(os.path.join(
        config.processed_path, '{}_train.csv'.format(config.dataset)), index=False)
    df_test.to_csv(os.path.join(
        config.processed_path, '{}_test.csv'.format(config.dataset)), index=False)


if __name__ == "__main__":
    main()
