### AdaBoost Implementation

The first part is that you need to write a function to implement the AdaBoost algorithm we discussed in class. The input of your function is the training and test sets, as well as the number of rounds of boosting \( T \). It should then run AdaBoost for \( T \) rounds, using the decision-tree algorithm as the base learner. The function should then return the predictions of the final combined classifier on the given training and test examples, as well as the training and test error rate of the combined classifier following each of the \( T \) rounds.

### Bagging Implementation

“Bagging” (short for “bootstrap aggregating”) is a different method for combining decision trees or other base classifiers. Similar to boosting, the base learning algorithm is run repeatedly in a series of rounds. However, the manner in which the base learner is called is different than in boosting. In particular, on each round, the base learner is trained on what is often called a “bootstrap replicate” of the original training set.

Suppose the training set consists of \( m \) examples. Then a bootstrap replicate is a new training set that also consists of \( m \) examples, and which is formed by repeatedly selecting uniformly at random and with replacement \( m \) examples from the original training set. This means that the same example may appear multiple times in the bootstrap replicate, or it may appear not at all.

Thus, on each of \( T \) rounds of bagging, a bootstrap replicate is created from the original training set. A base classifier is then trained on this replicate, and the process continues. After \( T \) rounds, a final combined classifier is formed which simply predicts with the majority vote of all of the base classifiers.

### Datasets

We will use three real-world datasets:

- **Letter Dataset:**  
  The first dataset is a letter dataset that contains descriptions of the characters “C” and “G”, and the goal is to distinguish between these two letters. The class label is either “C” or “G”. There are 16 attributes for things like the width of the letter and the total number of pixels turned on. There are 500 training and 1009 test examples. More detailed information about this dataset and the various attributes is available [here](https://archive.ics.uci.edu/ml/datasets/letter+recognition) (obviously, we used only the letters C and G).  
  The dataset is available at: [Letter Recognition Dataset](https://archive.ics.uci.edu/ml/datasets/letter+recognition)

- **Credit Dataset:**  
  The credit dataset classifies people described by a set of attributes as good or bad credit risks. There are 20 attributes encoding credit history, purpose of the loan, employment status, etc. There are 400 training and 600 test examples. More detailed information about this dataset and the various attributes is available [here](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) (we used the “original” dataset, and did not make use of the “cost matrix”).  
  The dataset is available at: [German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))

- **Spam Dataset:**  
  The spam dataset classifies email messages as spam or ham. The 57 attributes mainly encode the number of times that certain words or characters occur. There are 1000 training and 3601 test examples.  
  The dataset is available at: [Spambase Dataset](https://archive.ics.uci.edu/ml/datasets/spambase)

### Testing and Experimentation

Making sure that algorithms of this kind are working correctly is essential, but can be difficult since it is not always obvious what correct behavior looks like. For this reason, it is important to take extra care to test your implementations. One way of doing so is by trying small datasets on which it might be possible to compute the correct answer by hand.

Once you have your boosting and bagging implementations working, your next step is to run some experiments using them on the provided datasets. You should try out both boosting and bagging on the three provided datasets. You also should experiment both with deep trees and very shallow trees (say, of depth only one, often called “decision stumps”). Thus, there are (at least) twelve combinations to consider: there are three datasets, on each of which you can try any combination of bagging or boosting on deep or shallow trees.
