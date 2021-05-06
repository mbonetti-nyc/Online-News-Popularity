# Online News Popularity
### Michael S. Bonetti
#### Zicklin School of Business
#### CUNY Bernard M. Baruch College
#
#### Brief Description

This project aims to apply regression models, using R, on the Online News Popularity Dataset, in order to analyze model performance, runtimes, and variable importance. The dataset is available at UCI's ML repository and summarizes a heterogeneous set of features about articles published by Mashable in a two-year period.
https://archive.ics.uci.edu/ml/datasets/online+news+popularity#

The dataset containes 39,644 observations, with 58 predictive attributes (19 categorial and 39 numerical), with 2 non-predictive and 1 target (𝑠ℎ𝑎𝑟𝑒𝑠). For the purposes of this project, a 25% random sample (RS) was taken, using 9,911 observations and 57 predictive attributes. However, an incomplete run of the code on the full dataset (all 39,644 observations) was done, and I make comparisons between the 25% and the full dataset throughout the PowerPoint slide deck.

#
#### Data Pre-Processing
Before running regression methods and models, some data cleaning was done in order to remove outliers, omit missing values, remove non-predictive variables, and convert binomial weekday & news type into factors. Additionally, some preliminary exploratory data analysis (EDA) boxplots were created, to better visualize attribute importance and weight. To reduce bias/influence, the target variable was isolated into a dummy variable and removed from the dataset. Additionally, the target varaible y-distribution was taken and appears to be slightly right-skewed, but otherwise normal.
#
####  Creating Statistical Learning Models

#### R<sup>2</sup>
An 80/20 split of the training and testing set was performed to fit 4 regression models (LASSO, Elastic-net (EN), Ridge, and Random Forest (RF)) 100 times for 100 samples. The R<sup>2</sup> and runtimes were recorded, with boxplots to visualize these results. RF performed the best in both the training and testing sets, with all four methods generally performing at R<sup>2</sup> = ~ 24%, on average. Because of the field the type of data being analyzed resides in (social sciences), an R<sup>2</sup> of between 10 - 20% is considered normal.\
<img src="https://user-images.githubusercontent.com/83367900/117169373-0cb83d80-ad97-11eb-8fa9-c381b2064e34.png" width="45%" height="45%">

#### Cross-validation Curves
For one of the 100 samples, mean-squared-error (MSE) 10-fold cross-validation (CV) curves, using LASSO, EN, and Ridge, were done with Ridge performing the best in the 25% random sample, but the worst in the full dataset. When comparing between the 25% RS and the full dataset, the CV curves, log(λ) and runtimes were generally the same, with LASSO and EN curves becoming more tightly defined as more observations were included.\
<img src="https://user-images.githubusercontent.com/83367900/117169320-032ed580-ad97-11eb-9379-ab3a5db9977d.png" width="30%" height="30%">
<img src="https://user-images.githubusercontent.com/83367900/117169333-04f89900-ad97-11eb-8216-6d97b1f2b64b.png" width="30%" height="30%">
<img src="https://user-images.githubusercontent.com/83367900/117169342-06c25c80-ad97-11eb-8e2d-dc36256d3297.png" width="30%" height="30%">

#
####  Residuals
On observing the residuals for the training and testing set, based on one of the 100 samples, the residual medians neared zero, with all methods having roughly the same residual variance, expect for RF training. Even in the testing set, RF was slightly smaller than the other 3 methods of LASSO, EN, and Ridge. The boxplots shrank in variance, overall, as more observations were included.\
<img src="https://user-images.githubusercontent.com/83367900/117170935-6e2cdc00-ad98-11eb-86e1-71cbd94c6dca.png" width="45%" height="45%">

#
####  Bootstrapping, Performance and Runtimes
100 bootstrapping samples were performed, with the runtimes and results tracked, followed by fitting 10-fold CVs onto LASSO, EN, and Ridge, with an RF fitting. This was only done for the 25% RS, as this procedure was incomplete on the full dataset run. 90% confidence intervals of R<sup>2</sup> were consistent at approximately ± 0.05 from the means, with Random Forest performing the best but also with the highest runttimes. \
<img src="https://user-images.githubusercontent.com/83367900/117349867-e5866c80-ae79-11eb-8754-2ae210f684f0.png" width="40%" height="40%">
<img src="https://user-images.githubusercontent.com/83367900/117350062-1bc3ec00-ae7a-11eb-9d50-1874aaa95a88.png" width="48.5%" height="48.5%">

#
####  Variable Importance
Creating bootstrapping samples and fitting onto the entire 25% RS dataset allows for estimated coefficients (RF) and variable importance (LASSO, EN, Ridge) barplots to be generated. Variables 26, 25 and 27 were the top 3 influencers for RF.\
<img src="https://user-images.githubusercontent.com/83367900/117171268-b946ef00-ad98-11eb-8fd9-e41db15915dd.png" width="45%" height="45%">

#
####  Results

#### Variable Importance
* The top 3 positive influencers are 26 - 𝑘𝑤_𝑎𝑣𝑔_𝑎𝑣𝑔 (Avg. keyword (avg. shares)), 29 - 𝑠𝑒𝑙𝑓_𝑟𝑒𝑓𝑒𝑟𝑒𝑛𝑐𝑒_𝑎𝑣𝑔_𝑠ℎ𝑎𝑟𝑒𝑠 (Avg. shares of referenced articles in Mashable), and 37 - 𝐿𝐷𝐴_00 (Closeness to LDA topic 0)
* The top 3 negative influencers are 25 - 𝑘𝑤_𝑚𝑎𝑥_𝑎𝑣𝑔 (Avg. keyword (max. shares)), 10 - 𝑎𝑣𝑒𝑟𝑎𝑔𝑒_𝑡𝑜𝑘𝑒𝑛_𝑙𝑒𝑛𝑔𝑡ℎ (Avg. length of words in content), and 13 - 𝑑𝑎𝑡𝑎_𝑐ℎ𝑎𝑛𝑛𝑒𝑙_𝑖𝑠_𝑒𝑛𝑡𝑒𝑟𝑡𝑎𝑖𝑛𝑚𝑒𝑛𝑡 (Is data channel 'Entertainment’?)
* The top two ranking parameters for RF, 𝑘𝑤_𝑎𝑣𝑔_𝑎𝑣𝑔 and 𝑘𝑤_𝑚𝑎𝑥_𝑎𝑣𝑔, exactly match what Fernandes, Vinagre, and Cortez, the original dataset authors, achieved!
<img src="https://user-images.githubusercontent.com/83367900/117169211-e8f4f780-ad96-11eb-9932-9d743b389b35.png" width="45%" height="45%">

#### 25% 
Sample vs. Full Dataset Comparison
* Results were generally the same
* Overall performance decreased, and Ridge / RF caused runtimes to substantially increase (26+ hrs)
* CV curves had same shapes, while boxplot variances shrank

#### Improvements can be made…
* Using RF (still the best performer), AdaBoost, SVM, kNN, NB, as well as CART and C5.0
#### … but human behavior is unpredictable!
* Therefore, R<sup>2</sup> between 10 – 20% for social sciences is acceptable

#
#### Closing Thoughts
While human behavior can be predicted, to a certain extent in order to ascertain some insights from the Mashable dataset, performance may increase under classification models, but not by much for regression models. The performance figures I obtained, via the 25% RS where above average at ~ 21%, where this was significantly reduced under the full dataset run. As such, the unpredictability of human behavior will "cap" performance in a range between 10 - 20%. Therefore, using the full dataset, along with Random Forest, provides only minimal performance gains, but doesn't justify the excessive runtime to create the models. LASSO / EN appeared to be the better regression model choices, having consistently solid performance and quick runtimes. Additionally, by comparing my variable importance and estimated coefficent parameters, a common trait between my results and the authors was the top-ranking variables were not specific-type attributes (e.g. day of the week or article category) but more general in nature, and related to averages. As such, although improvements can surely be made, the data _may_ be insufficent to adequately predict the number of shares for a Mashable news article, based on its popularity.
