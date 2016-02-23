This implementation of Principal Component Analysis uses an Eigendecomposition to extract the principal components from a DataSet. While this is sub-optimal for large/distributed datasets, it will serve as a starting point to eventually implement Scalable PCA (detailed [here](http://thangnguyen.us/spca.pdf)).

## Naive Principal Component Analysis Algorithm

Let's say we have a data set that is represented as a collection of same sized vectors (dimensions/features). These vectors can be represented as rows of a Matrix (N rows X D columns). The naive algorithm for PCA is as follows:

1. Calculate the mean matrix (for each item in a vector in the matrix, subtract the mean of the containing vector)
2. Calculate the covariance matrix (each item is the dot product between 2 vectors divided by the vector size - 1)
3. Eigendecomposition of the covariance matrix 
4. Filter out the top N principal components based on the eigenvalues from step 3

-- Algorithm adapted from this [paper](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf).


## Code

Implemented [here](https://github.com/nguyent/flink/blob/master/flink-libraries/flink-ml/src/main/scala/org/apache/flink/ml/pca/PCA.scala) 

Tests [here](https://github.com/nguyent/flink/blob/master/flink-libraries/flink-ml/src/test/scala/org/apache/flink/ml/pca/PCASuite.scala)
