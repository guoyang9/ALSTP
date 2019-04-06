# choose dataset name
dataset = 'toys'

# paths
main_path = '/home/share/guoyangyang/amazon/'
stop_file = './stopwords.txt'

processed_path = './processed/'

full_path = processed_path + '{}_full.csv'.format(dataset)
train_path = processed_path + '{}_train.csv'.format(dataset)
test_path = processed_path + '{}_test.csv'.format(dataset)

asin_sample_path = processed_path + '{}_asin_sample.json'.format(dataset)
user_bought_path = processed_path + '{}_user_bought.json'.format(dataset)

doc2model_path = processed_path + '{}_doc2model'.format(dataset)
query_path = processed_path + '{}_query.json'.format(dataset)

# embedding size
embed_size = 512
