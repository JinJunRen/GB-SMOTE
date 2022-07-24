# Grouping-Based Synthetic Minority Oversampling Technique (briefly GB-SMOTE)

We have proposed a new oversampling algorithm called GB-SMOTE, which can circumvent the deficiency of WK-SMOTE [1] (and SMOTE as well as its variants) caused by randomly selecting some minority class samples. To design this oversampling method, we first design a simple grouping scheme that can divide the minority class into different groups by using slack variables. This grouping scheme has established a theoretical basis for selecting valuable minority class samples. Moreover, it also provides a new explanation for the poor performance of SVM on imbalanced data sets. Second, a reasonable samples selection scheme has been designed, which can avoid generating new samples in the overlapping region, and an effective samples generation scheme is proposed to generate high-quality new samples. Subsequently, an effective oversampling method GB-SMOTE is proposed. The idea of GB-SMOTE (partially selecting valuable samples) can also be applied to preprocess biased learners or to modify the input distribution in cases of a limited amount of data in semi-supervised learning. The experimental results indicate the effectiveness of the sample selection scheme and sample generation scheme in GB-SMOTE. Besides, the experiment on the real-world datasets shows that compared with all of the benchmark algorithms in both homologous and heterologous groups, GB-SMOTE outperforms them, especially on the data sets with a high imbalance ratio.


# Install

Our EASE implementation requires following dependencies:
- [python](https://www.python.org/) (>=3.7)
- [numpy](https://numpy.org/) (>=1.11)
- [scipy](https://www.scipy.org/) (>=0.17)
- [scikit-learn](https://scikit-learn.org/stable/) (>=0.21)
- [imblearn](https://pypi.org/project/imblearn/) (>=0.2) (optional, for canonical resampling)


```
git clone https://github.com/JinJunRen/GB-SMOTE
```

# Usage

## Documentation
**GB-SMOTE.py**

| Parameters    | Description   |
| ------------- | ------------- |
| `clf` | *object, optional (default=`sklearn.sklearn.svm.SVC()`)* <br> Built-in `fit()`, `predict()`, `predict_proba()` methods are required. |
| `C`    | *float, optional (default=100)* <br>  Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. <br>  The penalty is a squared l2 penalty. |

----------------

| Methods    | Description   |
| ---------- | ------------- |
| `fit(self, X, y)` | Build a GB-SMOTE classifier on the training set (X, y).|
| `predict(self, X)` | Predict class for X. |
| `predict_proba(self, X)` | Predict class probabilities for X. |
| `score(self, X, y)` | Returns the average precision score on the given test data and labels. |
| `calcKxi(self,X,y)` | Calcuate the slack variables of samples X. |
| `partitionInstance(self,X,y)` | Divide each class into three sets, e.g., error set, margin set and safe set. |
| `selectInstances(self,X,in_margin,pos_safe,n)` | Select n sample-pairs from #M^+ and S^+, respectively, and create a synthesized sample set. <br> Note that: the elements in in_margin and pos_safe are the indexs of samples in X.. |
| `augmentKernelMatrix(self,X_len,dim,kernelmatrix,n)` | Augment kernelmatrix based on the dim(it equals the dimension of dataset, that is, the sum of the length of both training set and test set) and n (the number of the generated samples).  |

----------------

**demorun.py**

In this python script we provided an example of how to use our implementation of GB-SMOTE methods to perform classification.

| Parameters    | Description   |
| ------------- | ------------- |
| `data` | *String*<br> Specify a dataset. |
| `ker`  | *String,(default=`rbf`)*<br> Specify the type of kernel of SVM. |
| `n`  | *Integer,(default=`n`)*<br> Specify the number of n-fold cross-validation. |

----------------

## Examples

```python
python demorun.py -data ./dataset/moon_1000_100_2.csv -n 5 
or
python demorun.py -data ./dataset/moon_1000_200_4.csv -n 5
```

##Dataset links:
[Knowledge Extraction Based on Evolutionary Learning (KEEL)](https://sci2s.ugr.es/keel/studies.php?cat=imb).


# References
- [1] Mathew J, Pang C K, Luo M, et al. Classification of imbalanced data by oversampling in kernel space of support vector machines[J]. IEEE transactions on neural networks and learning systems, 2017, 29(9): 4065-4076.
