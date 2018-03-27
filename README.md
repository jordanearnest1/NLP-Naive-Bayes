# Naive-Bayes
A Naive Bayes classifier to classify tweets as offensive vs. non-offensive. 

Tokenizes each tweet by white space, trains to compute P(X = xi), P(Y = yj), and P(X = xi|Y = yi), smooths the weights, and classifies the tokenized document, returning the class with the highest posterior probability for each tweet.
