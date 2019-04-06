# ALSTP

A pytorch and tensorflow GPU implementation of ALSTP.

Yangyang Guo, Zhiyong Cheng, Liqiang Nie, Yinglong Wang, Junma and Mohan Kankanhalli (2018). Attentive Long Short-Term Preference Modeling for Personalized Product Search. In TOIS.

**Please cite our TOIS paper if you use our codes. Thanks!**


You can download the Amazon Dataset from http://jmcauley.ucsd.edu/data/amazon.

## The requirements are as follows:
* python==3.6

* pytorch==0.4.1

* tensorflow==1.7

* gensim==3.7.1

* tensorboardX==1.6

## Example to Run
* Make sure the raw data, meta data are in the same direction.
* Preprocessing data. Filter the review to each user having at least 10 transactions. Remove the words whose number is less than ```count```. Split the data into three sets and extract queries.
   ```
   python extract.py --review_file --meta_file --count
   ```
* We leverage the PV-DM model to convert queries and product representations to the same latent space.
   ```
   python doc2vec.py --dataset --embedding_size --window_size
   ```
* Now start training the model. 
   ```
   python main.py --dataset --lr --num_steps --alpha
   ```
