## Logistic Regression

**environment**:Python3.6 

**lib**: numpy、matplotlib、random

**dataset**:Iris

**the  model have three kind of train/test proportion.**

1. use 50% of dataset as train set,50% of dataset as test set
2. use 70% of dataset as train set,30% of dataset as test set
3. use 90% of dataset as train set,10% of dataset as test set

As i want to show my test set and classification boundary,i have to reduce 

dimensionality,so i choose **PCA** to do that.

About update parameters,i use two method.one is **SGD**,the other is **Adam**,all

of them i just use python and numpy to realize without sklearn and other lib.

At last,when i finish my train,i show the cost function and compute accuracy on my 

test set.you can see it in my code.Of course you can also run it to see.