# Title: Protocol for "Spezielle Bioinformatik 3" Using Classical Machine Learning Against Pre-Existing Deep Learning Models in PET/CT-Based Local Prostate Cancer Recurrence

# **Abstract**:

### Background: 
As by far the most common cancer in <mark style="background: #FF5582A6;">men </mark>, Prostate Cancer (PC) is of great interest for medical research. Of the various established imaging methods PSMA-directed positron emission tomography (PET) has prove itself as a reliable method for diagnosing and detecting recurrent PC as well as metastasis. A deep learning model previously trained on [18F]-prostate specific membrane antigen (PSMA)-1007 PET-Scans to detect local PC recurrence serves as a base line for comparison of performance. Training another model with classic machine learning methods, on the same dataset, the goal was to see how close in performance we could get with as simplistic of an approach as possible. 
### Methods:
Multiple models based on three different algorithms were trained on the metadata of a dataset including 1404 [18F]-PSMA-1007 PET/CT re staging scans from patients with histologically confirmed prostate cancer. [[results]]
### Conclusion:
In the presented data we show, that using deep learning as the preferred approach for analysing this dataset of 1404 scans was at least better to some degree. As with the highly simplified approach we managed to almost match f1 and accuracy, surpassing it in some specific model and metrics. Which leads to believe that simply the addition of the prostatectomy status has a high degree of influence over prediction result. Data processing and amount may hold back both approaches from achieving clinically relevant models. 
# **1. Introduction**

The goal of this internship is to take a simplified approach to prostate cancer recurrence detection form PET/CT scans, or rather the tabular data derived from patients. 
To achieve proper documentation as well as holding up a basic scientific standard, trackable version control was required. For this reason learning basic version control via git/github in a "data carpentry" course was required.
Following this, the goal was to have a comprehensive overview across all future models and metrics, for this purpose MLFLOW was used as it offers easy to understand insights and comparison across models, as well as additional version control. As a base for all models the repository scikit-learn was utilised, it offers comprehensive documentation, making it easier to understand and adjust models for our intended use.

# **2. Materials and Methods**

## 2.1 Data processing/Study population
The data used for this project was based on the metadata obtained from patient scans using 18F-PSMA-1007 PET/CT imaging from the department of Nuclear Medicine at the University Hospital Würzburg, conducted between 2019 and 2023. The initially provided patient split was 1016 scans/patients for training, 188 for validation and 200 as a test set to be conducted at the end of the project.

The metadata used for model training consists of the data obtained from each patient that got referred to primary/follow up screening due to elevated PSA values.
Per patient there are two randomised IDs for privacy, age of the patient, gender, whether or not its the primary or restaging of said patient, status of prostatectomy px (0 = no prostatectomy, 1 = prostatectomy), levels of the PC indicative protein PSA, the label (0 = no cancer, 1 = cancer, 2 = uncertain) and the intended set with either training or validation. The actual images were not integrated into the feature selection. At later stages the consideration was made to add derived features from the image data, such as image intensities and potential lesion volumes. Inclusion of the additional features was deemed outside the scope of the initial project and to be looked at in the follow up project.

[[table]]

The patient data was  simplified as much as possible,after the initial model training, to introduce as little variables or potential issues as possible. For this, any data containing the label 2, rows containing features with N/A and primary staging patients were excluded after initial testing including all data. After this, only the relevant features of age, psa and px were included in further training. 

Val/Train masks were created and matching was done via the index, to remove both patient IDs. 
The overall data was reduced from 1205 viable rows to <mark style="background: #FFB8EBA6;">~1000</mark> rows, with X and Y being removed for the test and the validation set.


## 2.2 Data logging/documentation via GitHub/MLFLOW
For version control GitHub was used. To learn how to use git and GitHub a software carpentry course was completed. In this course the first lesson was the concept of version control based on checkpoints, as well as conflict resolution with two differing outputs. After setting up the account and setting the basic text editor, nano in our case, we finished up the config. 
The first task was to set up a repository and create basic text files to track and manipulate and inspect these changes. The best practises regarding git commit messages were explained. The concept of the staging area was explained using the example of multiple receipts. While simple in theory, the gitignore file is very important for this project, as we only wanted to track the most important metrics and the section was studied in detail. Afterwards a remote repository was created in GitHub, as this project was not being worked on by multiple people, best practises regarding branching and pushing to main were not explored. For this project a trunk based branching path was chosen. 

When creating and testing various models based on different data subsets, parameters or algorithms, a lot of different models, all with specific performance metrics, will be created. To not accidentally toss a well performing model or rely on tracking hundreds of small files via GitHub, MLFLOW was introduced. It allowed a fully local storage of any and all models including the specific parameters, metrics and environment they were created in. The tracking for the models is highly customisable and manual tracking was preferred over the easy to implement automated tracking provided within the documentation. This allowed us to specify to only track the parameters differing from the default parameters, as seen in the documentation of each method in Scikit-learn and the parameters that actually reflect model performance. The custom parameter and metric tracking allowed for standardised comparison between models by using a confusion matrix involving Accuracy, Recall, Precision and F1-score. These metrics could be visualised in a comprehensive graph view within the locally hosted UI of MLFLOW, allowing for model evaluation and comparison at a glance.
The primary issue across the project was adjusting the tracking for the different purposes, as for parameter tuning, not every run was relevant to keep track of, simply the best one or two parameter configurations. With the initial automated tracking that was implemented just a few simple grid search runs would create around 20 models with almost exact outputs, cluttering the graphs, as well as around 200 files for GitHub to keep track of, as the parameters used weren't accounted for via the gitignore file yet. When the tuning was finished, the best performing model of each used algorithm was selected and logged internally. This selection process was possible due to the evaluation function MLFLOW provides, as for some of the initial tuning runs the confusion matrix was not tracked, for proper comparison between all the models, the best performing grid search models were manually evaluated with the evaluate function MLFLWO provides. The evaluate function provides not only the confusion matrix for each model, but also other insights like a ROC-AUC, providing additional detail with which to select the best performing models. 

## 2.3 Scikit learn
Scikit learn provides all the algorithms used for this Project. For each algorithm used it provides in depth documentation and explanation about parameters and attributes used to adjust, analyse and improve the results, as well as a basic examples of code and their output. The most important sections for this work consisted of 1.10 "Decision Trees", 1.11 Ensembles: "Gradient boosting and random forests, bagging, voting, stacking" and the whole of section 3: Model selection and evaluation. The latter being used to learn how to evaluate the created models.

## 2.4 Decision tree
The decision tree is a simple, yet powerful classification algorithm, due to its simplicity it was chosen as the first model algorithm. It attempts to categorise data based on decision splits, these are based on specific attributes found within the dataset. These splits are referred to as branches, splitting the dataset into nodes with a certain purity, the purity of a node reflects the homogeneity of data characteristics of the node. This continues until the tree reaches end nodes, so called "leaves", with maximum purity. A leaf with a purity of 1 contains only one specific result or label. 

This method of looking for a pattern leading to the purest leaf often leaves to overfitting. That is, it will be specialised on the seen data, as it accounts for all the noise or outliers to achieve good results in training with near-perfect accuracy, but will struggle to handle previously unseen data. 

For this reason, a method called "pruning" is used to overcome the issue of overfitting. By adjusting certain parameters in the decision tree function, provided by scikit-learn, we can ensure that the tree doesn't develop too many leaves, "cut" off the ones that do not impact the final score too much and are most likely based on noise, or ensure that it doesn't branch up to infinity (or until all leaves are homogenous).

Another aspect that was considered for the decision tree is removing the provided set split feature altogether and scrambling the train/validation sets in different ways via k-fold cross validation to systematically split the data into smaller subsets and train based on withholding one of these sets. The specific implementation of the k-fold method was provided by scikit learn.

## 2.4 Random Forest
The Random Forest is an expansion of the decision tree, it tries to solve the prominent issue of overfitting a single tree by using a multitude of, often times, weaker trees. These come to a final prediction via majority vote. 

The most important factor for this to work is something called bootstrapping. Training every tree on the exact same data, even accounting for randomness in training and weak learners, will lead to biased/converging trees. Weak learners in this context are decison trees that do not achieve high accuracy, that can be by limiting their depth (i.e.. how many branches they are allowed to create) or heavy pruning. By splitting the training data into smaller subsets, done in a process called "bagging", each tree ends up vastly different from one another. While the individual trees might be overfitting or have a bias, across the entirety of the forest these issues are supposed to balance themselves out.

Two approaches tested to this were a bigger forest with weak learners (100+ trees, max depth 3-10) as the default approach to this problem. The other approach was using a smaller forest (20 trees, no depth limit) with strong learner trees (20 trees, no depth limit) to see if it could compete with the larger forest. 

## 2.5 XGBoost
The Extreme Gradient Boosting algorithm is also an ensemble method, similar to the random forest and based on decision trees. Originally seen in [[Greedy Function Approximation: A Gradient Boosting Machine, by Friedman]] and the further developed and enhanced for use by Tianqi C<mark style="background: #FFB8EBA6;">hen</mark>.
Instead of training multiple trees at once, unlike the Random Forest, XGBoost trains multiple weak trees sequentially to correct the mistakes of the previous trees using something called gradient descent. This process involves calculating the log loss function from the residuals, the difference between the predicted and actual value. To minimize this function different parameters are scored with changed weights to create a strong classifier.
An integral factor for determining the models performance is the learning rate, as it scales the step length of the gradient descent procedure, that is to say it influences the impact of each individual tree. To still ensure sufficient learning, the number of iterations is usually increased to account for a reduced learning rate. Combined with bagging, or subsampling as its called for XGBoost, it can provide an substantially improved result as shown by [[T. Hastie, R. Tibshirani and J. Friedman, “Elements of Statistical Learning Ed. 2”, Springer, 2009.]] ![[Pasted image 20251203101829.png]]


Another important feature of XGBoost is early stopping, this method help to determine the optimal number of iterations to build a model that is not overfitted or exhibits a great amount of bias. Early stopping occurs when the models performance on the validation set plat<mark style="background: #FFB8EBA6;">eus</mark> or worsens across a given number of iterations.

Scikit learn provides a visualised example of early stoppage and its impact on training and validation error as well as training time. 

![[sphx_glr_plot_gradient_boosting_early_stopping_001.png]]

## 2.6 Training, Optimisation and Evaluation

###### 2.6.1 Model A: Decision tree
The initial model was using the default settings of the function provided by scikit learn and included all the tabular data without any adjustments to establish a baseline. Train and validation sets were not considered yet and a simple 80/20 random split was introduced.

Following this, multiple features deemed not relevant or potentially damaging to the training process were dropped and the given splits into train and validation sets considered.

The third and final version of the decision tree model used five-fold GridSearchCV to systematically check for the best parameter combination out of the given values. From this point onwards the random state 42 was introduced as a constant, to keep the model consistent and replicable, removing another layer of randomness that could affect results in either way.

The specific parameters given for parameter tuning of the decision tree <mark style="background: #FFB8EBA6;"> include:</mark>

| Parameter (value)                         | Function                                                                                                                                                     |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Criterion<br>(gini, entropy)              | Measuring the quality of a split based on Gini impurity or shannon information gain***                                                                       |
| max_depth<br>(none,10, 20, 30)            | Maximum amount of splits the tree is allowed to make. If none, tree will continue splitting till all leaves are pure, or other parameters prevent splitting. |
| min_samples_split <br>(2, 5, 10)          | The minimum number of samples required to split a node                                                                                                       |
| min_samples_leaf <br>(1, 2, 4)            | The minimum number of samples required to be contained within a lea node, forces every split to at least contain (min_number_leaf) on both sides             |
| min_weight_fraction_leaf (0.0, 0.1, 0.2)  | The minimum fraction of samples required at a leaf note, if weight is provided, equal weight is assumed                                                      |
| random_state <br>(42)                     | Choose the seed determining the randomness of the method. Used to ensure replicable results each time.                                                       |
| max_leaf_nodes <br>(None, 10, 20, 30)     | Grow the best tree using the specified number of leaves. <br>Best tree is determined by elative reduction in impurity                                        |
| min_impurity_decrease <br>(0.0, 0.1, 0.2) | Nodes split must decrease impurity by as much or more. Decrease is calculated***                                                                             |
| ccp_alpha <br>(0.0, 0.1, 0.2)             | Parameter for minimal Cost-Complexity Pruning.<br>Subtree with largest cost complexity smaller than ccp_alpha will be chosen.***                             |
###### 2.6.2 Model B: Random Forest
<mark style="background: #FF5582A6;">The initial Random Forest was trained at default settings with the cleaned data, establishing a base for this algorithms performance.</mark>

The second iteration of model B used parameter tuning for optimisation. This time two tuning runs were conducted to establish the difference between a smaller, more robust forest and a larger, less robust forest. The included parameters are as follows:

| Parameter (value)                           | Function                                                                                                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Criterion<br>(gini, entropy)                | Measuring the quality of a split based on Gini impurity or shannon information gain.                                                                         |
| max_depth<br>(none,10, 20, 30)              | Maximum amount of splits the tree is allowed to make. If none, tree will continue splitting till all leaves are pure, or other parameters prevent splitting. |
| min_samples_split <br>(2, 3, 4)             | The minimum number of samples required to split a node.                                                                                                      |
| min_samples_leaf <br>(1, 2, 4)              | The minimum number of samples required to be contained within a lea node, forces every split to at least contain (min_number_leaf) on both sides.            |
| min_weight_fraction_leaf (0.0, 0.1, 0.2)    | The minimum fraction of samples required at a leaf note, if weight is provided, equal weight is assumed.                                                     |
| random_state <br>(42)                       | Choose the seed determining the randomness of the method. Used to ensure replicable results each time.                                                       |
| max_leaf_nodes <br>(None, 10, 20, 30)       | Grow the best tree using the specified number of leaves. <br>Best tree is determined by elative reduction in impurity.                                       |
| min_impurity_decrease <br>(0.0, 0.01, 0.1,) | Nodes split must decrease impurity by as much or more. Decrease is calculated.                                                                               |
| ccp_alpha <br>(0.0, 0.01, 0.0015, 0.01)     | Parameter for minimal Cost-Complexity Pruning.<br>Subtree with largest cost complexity smaller than ccp_alpha will be chosen.                                |
| n_estimators<br>(20, 100)                   | The amount of trees trained for the ensemble. The final result is determined by a collective vote of all estimators                                          |
| bootstrap<br>(bool)                         | Enables/Disables bootstrapping. If true, a subset of the dataset is used to train each tree, reducing bias.                                                  |



###### 2.6.3 Model C: Gradient boosting 
The first iteration of the GradientBoostingClassifier used the default settings with no subsampling, early stoppage and a low learning rate.

Due to the amount of variables to consider for each parameter, the grid search had to be done in batches, with the best performing parameters being used in the next batch. Some parameters were left out of testing, as they were shown to have limited impact in the previous optimisations. 

The options of early stoppage and warm starts were both not considered because of this.
Focus of the grid search was to optimise the Gradient Boosting regularisation and the tree itself.

| Parameter (value)                | Function                                                                                                                                                           |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| loss<br>(log_loss)               | The log loss refers to the binomial deviance in this classification problem.<br>Its the loss function to be optimised.                                             |
| learning_rate<br>(0.1, 0.2, 0.5) | Dictates the impact of each individual tree on the loss function. Combined with subsampling it can be used for regularisation.                                     |
| n_estimatiors<br>(20, 100)       | Number of iterative boosting stages to perform. Due to robustness against overfitting, higher numbers typically lead to better results.                            |
| subsample<br>(0.5, 0.8, 1.0)     | The fraction of samples used to train each boosting stage. Any value below 1 will result in Stochastic Gradient Boosting, reducing variance while increasing bias. |
| criterion<br>(friedman_mse)      | Measuring function of split quality using mean squared error with improvement score by Friedman.                                                                   |
| warm_start<br>(bool)             | Enables reuse of a previous solution as starting point, adding more estimators to the ensemble.                                                                    |
| validation_fraction<br>(float)   | Only used in case of early stoppage. Fraction of training set, set aside for validation in case of stoppage.                                                       |
| n_iter_no_change<br>(int)        | Parameter deciding early stopping when validation scores across iterations do not improve. When set to none, no early stoppage will occur.                         |
| tol<br>(float)                   | Tolerance determining the minimum improvement for the validation score across n_iter_no_change.                                                                    |

# **Results**
Model A.1, using unadulterated input with no parameter modifications achieved a F1-score of 0.53 and a balanced accuracy of 0.65. The F-score indicating a prediction only minimally better than random chance.
After removal of excess data in model A.2 led to an increased F1-score of 0.57 and a balanced accuracy of 0.57.
Performing grid search to find optimal parameter configurations yielded model A.3 with an F1-score of 0.75 and a balanced accuracy of 0.76. The only change compared to A.2 was the inclusion of the parameter "min_weight_leaf" of 0.1 compared to 0.0.

For model B.1 the initial settings of the Random Forest included the previously determined min_weight_leaf of 0.1 and a smaller estimator size of 20 estimators. This model achieved almost the exact same results of A.3 with an F1-score of 0.75 and a balanced accuracy of 0.76.
The grid search for the random forest was split into a smaller forest of strong learners (model B.2a) with:
('min_weight_fraction_leaf': [0.1], 'n_estimators': [20], 'max_depth': [None, 10, 20, 30], 'ccp_alpha': [0.0, 0.005, 0.01, 0.0015]) 
and a larger forest of weaker learners (model B.2b) with:
('min_samples_split': [2, 3, 4], 'min_weight_fraction_leaf': [0.0, 0.01, 0.1], 'min_impurity_decrease': [0.0, 0.01, 0.1], 'max_leaf_nodes': [None, 10, 20])
Model B.2a achieved better results with an F1-score of 0.74 and balanced accuracy of 0.76.
Model B.2.b performed worse with an F1-score of 0.70 and a balanced accuracy of 0.72.

The first Gradient Boosting model C.1 used the same modified data and default parameters resulting in an F1-score of 0.70 with a balanced accuracy of 0.70.
Multiple grid searches combined, while keeping the determined best parameters resulted in model C.2, these stayed same to the default and achieved an F1-score of 0.70 with a balanced accuracy of 0.70.

Model B.2b was chosen for evaluation with the hold out set, that was not used in either training nor validation in any models.
# **Discussion**
This works aim was to develop traditional machine learning model, that could predict PC recurrence form only the metadata provided from patients coming in for restaging PSMA-PET/CT Scans. 
Starting form the simplest algorithm with a naïve approach utilising no data trimming, which resulted in mostly random predictions, with only an above expected accuracy. This was due to the inclusion of primary staging patients and some patients coming in for multiple scans. In most cases these were cancerous, leading to bias. To remedy this, any primary staging as well as patient identifiers were removed, while respecting the set splitting mask, leading to a worse result in Accuracy and consequently F1, while slightly improving Precision and Recall. 
Interestingly, even after extensive testing, only one Parameter seemed to have a meaningful impact on metrics. "min_weight_fracton_leaf" of 0.1 solved the issue of overfitting and filtering out noise. Including or Excluding the parameter change from subsequent grid searches showed other parameter combinations that performed better than the default settings, but could not reach the combination of default settings with a minimum fraction.
Creating the next models based on a random forest algorithm, but keeping the previous findings in mind. The initial small forest seemed to have mostly copied the individual trees previously made, as the results only start to differ past the third decimal point. Considering the first attempt at a forest was built around a smaller forest with strong learners, this outcome could have been predicted. The parameter tuning that followed was split into two approaches. The approach to refine the "robust" forest showed no improvement over the previous model. While expected, this confirms that the additional parameters available for tuning in a forest model did not affect decision making.
[[results grid search wide]] 
[[why is leaf so good]]
As the most complex and final model, gradient boosting was expected to perform the best or at least similar to all the previous models. The results however show a 5 to 10% decrease in performance for F1, in the recall metric is where the performance drop is the most significant. While, with the given data, a singular tree created via iterative learning, might not have been better than the previous models using a multitude of trees or a single fine-tuned tree, this result was unexpected. As a cause for this might be an insufficient exploration of the parameters available for this algorithm. Due to the large quantity of parameters only a selected number of "more important" parameters were chosen and had to be run in grid search batches, as the exhaustive exploration using the previous method would have taken around 3 year based on rough estimations (using the universities provided Lenovo Thinkpad). In favour of learn set shrinkage and learn rate reduction, which was explained above, the concept of early stoppage has been neglected here. Early stopping could have allowed for more extensive parameter testing, as it can significantly reduce the train time for most runs. Other ways to more extensively test parameter combinations for gradient boosting could have been Bayesian search instead of the exhaustive grid search or the third party framework OPTUNA. These issues were brought up and considered but deemed outside the scope of this project. There may be a configuration not explored with better performance. 

The rate of true positive's could be seen as the clinically most significant metric, as it dictated the rate of actual cancer diagnoses. When looking at the individual models confusion metrics it becomes apparent that the sensitivity is significantly higher than the recall, often balancing out the data. After running a spearman's rank correlation using seaborn, the weak correlation between Age, PSA levels and label (all calculated separately) becomes apparent. The more significant finding was the comparatively stronger negative correlation between prostatectomy status and label. 
![[Pasted image 20251204152322.png]]
This, combined with the low amount of features processed in training, would be an explanation as to why all models across the board struggle with recall while still achieving acceptable numbers compared to the original deep learning model. This seems to be an issue of the original model too, as in the final test with only a 56.7% rate of correct predictions for true positives but an 88% rate for true negatives, the previous versions also have a heavy bias towards true negatives. Reinforcing our believe that prostatectomy status plays a heavy role in the prediction. ![[Pasted image 20251204152441.png]]

[[section about final test set]]
# **Conclusion**
After testing multiple algorithms with various parameter configurations and complexities, it seems that despite getting near the performance of the deep learning model used for comparison we were unable to reach the same or better performance with the available data. 
Considering the small amount of tabular data used to train the models there is potential to increase performance by carefully selecting and determining significant image features to enter into training. These results are in line with the expected outcome of the supervisor and show that, while there still is some potential improvements to be had using classical ML methods, 
choosing a deep learning model for the initial project was not a bad choice. 
[[unable to surpas baesian error?]]

# **Appendix**

Model version differences/full metrics like grid search or kfold
mathematical formulas of parameters
Packages
code/github
All tools used (obsidian, zotero, )