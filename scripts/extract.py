import pandas as pd
import numpy as np

import gzip, json
import argparse
import os, time, sys

import text_preprocess as tp


def getDF(path):
    """Apply raw data to pandas DataFrame."""
    i = 0
    df = {}
    g = gzip.open(path, 'rb')
    for line in g:
        df[i] = eval(line)
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def extraction(meta_path, reviewDF, stop_path, count):
    """ Extract item categories from meta data."""
    with gzip.open(meta_path, 'rb') as g:
        category_dic = {}
        descrip_dic = {}
        for line in g:
            line = eval(line)
            category_dic[line['asin']] = line['categories']

    # Filter each user have at least 10 transactions.
    review_lengths = reviewDF.groupby('reviewerID').size()
    reviewDF = reviewDF[np.in1d(reviewDF.reviewerID, 
                            review_lengths[review_lengths >= 10].index)]
    reviewDF = reviewDF.reset_index(drop=True)

    # For further stop words processing
    stopDF = pd.read_csv(stop_path, header=None, names=['stopword'])
    stop_set = set(stopDF['stopword'].unique())

    query_list, reviewText_list, review_with_des = [], [], []
    for i in range(len(reviewDF)):
        asin = reviewDF.asin[i]
        value_queries = category_dic[asin]

        # Remove punctuation marks, duplicated and stop words
        new_queries = []
        for value_query in value_queries:
            value_query = tp.remove_char(value_query)
            value_query = tp.remove_dup(value_query)
            value_query = tp.remove_stop(value_query, stop_set)
            new_queries.append(value_query)

        # Return the maximum length of queries
        new_query = max(new_queries, key=len) 
        
        # Process reviewText and description.
        reviewText = reviewDF.reviewText[i]
        reviewText = tp.remove_char(reviewText)
        reviewText = tp.remove_stop(reviewText, stop_set)

        query_list.append(new_query)
        reviewText_list.append(reviewText)

        # reviewText.extend(des_text) 
        review_with_des.append(reviewText)

    reviewDF['query_'] = query_list 

    # Filtering words counts less than count
    reviewText_list = tp.filter_words(reviewText_list, count)
    reviewDF['reviewText'] = reviewText_list
    review_with_des = tp.filter_words(review_with_des, count)
    reviewDF['reviewDescript'] = review_with_des

    return reviewDF


def split_data(df):
    split_filter = []
    user_length = df.groupby('reviewerID').size().tolist()

    for user in range(len(user_length)):
        for _ in range((user_length[user] - 1)):
            split_filter.append('Train')
        split_filter.append('Test')
    
    df['filter'] = split_filter
    df_train = df[df['filter'] == 'Train']
    df_test = df[df['filter'] == 'Test']

    return df, df_train, df_test

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

def removeTest(df, df_test):
    """ Remove test review data and remove duplicate."""
    df = df.reset_index(drop=True)
    reviewText = []

    for i in range(len(df)):
        if df['filter'][i] == 'Test':
            reviewText.append("[]")
        else:
            reviewText.append(df['reviewText'][i])

    df['reviewText'] = reviewText

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
        default='/media/yang/DATA/Datasets/amazon',
        help="All source files should be under this folder.")
    parser.add_argument('--review_file', type=str,
        default='reviews_Toys_and_Games_5.json.gz',
        help="5 core review file.")
    parser.add_argument('--meta_file', type=str,
        default='meta_Toys_and_Games.json.gz',
        help="Meta data file for the corresponding review file.")
    parser.add_argument('--count', type=int, default=5,
        help="Remove the words number less than count.")
    parser.add_argument('--stop_file', type=str, default='stopwords.txt',
        help="Stop words file.")
    parser.add_argument('--save_path', type=str, default='processed',
        help="Destination to save all the files.")

    FLAGS = parser.parse_args()

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    meta_path = os.path.join(FLAGS.data_dir, FLAGS.meta_file)
    review_path = os.path.join(FLAGS.data_dir, FLAGS.review_file)
    stop_path = os.path.join(FLAGS.data_dir, FLAGS.stop_file)

    reviewDF = getDF(review_path)

    df = extraction(meta_path, reviewDF, stop_path, FLAGS.count)
    print("Pre extraction done!\nThe number of users is %d.\tNumber of items is %d.\tNumber of feedbacks is %d." %(
        len(df.reviewerID.unique()), len(df.asin.unique()), len(df)))
    df = df.drop(['reviewerName', 'reviewTime', 'helpful', 'summary', 'overall'], axis=1)

    df = df.sort_values(by=['reviewerID', 'unixReviewTime'])
    df, df_train, df_test = split_data(df)   
    print(df_train); sys.exit(0)
    user_bought = get_user_bought(df_train)
    json.dump(user_bought, open(os.path.join(
                                FLAGS.save_path, 'user_bought.json'), 'w')) 
    df = removeTest(df, df_test) #Remove the reviews from test set.

    df = df.drop(['unixReviewTime', 'filter'], axis=1)
    df_train = df_train.drop(['unixReviewTime', 'filter'], axis=1)
    df_test = df_test.drop(['unixReviewTime', 'filter'], axis=1)

    df.to_csv(os.path.join(FLAGS.save_path, 'full.csv'), index=False)
    df_train.to_csv(os.path.join(FLAGS.save_path, 'train.csv'), index=False)
    df_test.to_csv(os.path.join(FLAGS.save_path, 'test.csv'), index=False)

    print("All processes done!")


if __name__ == "__main__":
    main()
