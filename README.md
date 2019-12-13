# ALSTP

A pytorch GPU implementation of ALSTP.

Yangyang Guo, Zhiyong Cheng, Liqiang Nie, Yinglong Wang, Junma and Mohan Kankanhalli (2018). Attentive Long Short-Term Preference Modeling for Personalized Product Search. In TOIS.

**Please cite our TOIS paper if you use our codes. Thanks!**


You can download the Amazon Dataset from http://jmcauley.ucsd.edu/data/amazon.

## The requirements are as follows:
	* python==3.6
	* pandas==0.24.2
	* numpy==1.16.2
	* pytorch==0.4.1
	* gensim==3.7.1
	* tensorboardX==1.6

## Example to Run
* Make sure the raw data, meta data are in the same direction.
* Preprocessing data. Filter the review to each user having at least 10 transactions. Remove the words whose number is less than ```count```. Split the data into three sets and extract queries.
   ```
   python scripts/process.py --review_file=selected_file --meta_file=selected_file --count=5
   ```
* We leverage the PV-DM model to convert queries and product representations to the same latent space.
   ```
   python scripts/doc2vec.py --window_size=3
   ```
* Start training the model. 
   ```
   python ALSTP.py --lr=0.001 --num_steps=4 --alpha=0.9
   ```
