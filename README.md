# Online News Popularity
### Michael S. Bonetti
#### Zicklin School of Business
#### CUNY Bernard M. Baruch College
#
#### Brief Description
This project aims to apply regression models, using R, on the Online News Popularity Dataset, in order to analyze model performance, runtimes, and variable importance. The dataset is available at UCI's ML repository and summarizes a heterogeneous set of features about articles published by Mashable in a two-year period.
https://archive.ics.uci.edu/ml/datasets/online+news+popularity#

The dataset containes 39,644 observations, with 58 predictive attributes (19 categorial and 39 numerical), with 2 non-predictive and 1 target (shares). For the purposes of this project, a 25% random sample was taken, using 9,911 observations and 57 predictive attributes. However, an incomplete run of the code on the full dataset (all 39,644 observations) was done, and I make comparisons between the 25% and the full dataset throughout the PowerPoint slide deck.

#
#### Data Pre-Processing
Before running regression methods and models, some data cleaning was done in order to remove outliers, omit missing values, remove non-predictive variables, and convert binomial weekday & news type into factors. Additionally, some preliminary exploratory data analysis (EDA) boxplots were created, to better visualize attribute importance and weight. To reduce bias/influence, the target variable was isolated into a dummy variable and removed from the dataset. Additionally, the target varaible y-distribution was taken and appears to be slightly right-skewed, but otherwise normal.
#
####  Creating Statistical Learning Models

#### R<sup>2</sup>
An 80/20 split of the training and testing set was performed to fit 4 regression models (LASSO, Elastic-net (EN), Ridge, and Random Forest (RF)) 100 times for 100 samples. The R<sup>2 </sup> and runtimes were recorded, with boxplots to visualize these results. RF performed the best in both the training and testing sets, with all four methods generally performing at R<sup>2</sup> = ~ 24%, on average. Because of the field the type of data being analyzed resides in (social sciences), an R<sup>2</sup> of between 10 - 20% is considered normal.
  
#### Cross-validation Curves
For one of the 100 samples, mean-squared-error (MSE) 10-fold cross-validation (CV) curves, using LASSO, EN, and Ridge, were done with Ridge performing the best in the 25% random sample, but the worst in the full dataset. When comparing between the 25% RS and the full dataset, the CV curves, log(lambdas) and runtimes were generally the same, with LASSO and EN curves becoming more tightly defined as more observations were included.
#
####  Residuals
On observing the residuals for the training and testing set, based on one of the 100 samples, the residual means neared zero, with all methods having roughly the same residual variance, expect for the RF training. Even in the testing set, RF was slightly smaller than the other 3 methods of LASSO, EN, and Ridge.
defined as more observations were included.
#
####  Bootstrapping, Performance and Runtimes
100 bootstrapping samples were performed, with the runtimes and results tracked, followed by fitting 10-fold CVs onto LASSO, EN, and Ridge, with an RF fitting. This was only done for the 25% RS, as this procedure was incomplete on the full dataset run.

#
####  Variable Importance
Creating bootstrapping samples and fitting onto the entire 25% RS dataset allows for estimated coefficients (RF) and variable importance (LASSO, EN, Ridge) barplots to be generated. 
#
####  Results / Closing Thoughts
