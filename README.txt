Created 29-04-2021

This file contains relevant code for the final version of the paper "Predicting the probability of death using proteomics".

Mor_feature_ranking.ipynb  - Examining feature ranking, predicting, testing and plotting
Mor_heritability.ipynb  -  Calculate heritability, correlation between siblings and parent offspring correlations
Mor_pred_associations.ipynb  -  Associations of predicted mortality to phenotypes (HERA phenotypes and diseases.)
                       
Mor_predict_cv_results.ipynb  - Load and plot crossvalidation results
Mor_results_newdata.ipynb  -  Plot prediction results for recent datsets (few cases)
Mor_trad_20210106.ipynb  -  Other quantitative pehnotypes tried.
Mor_top_protein.ipynb  - Look at correlations of top proteins 
Mor_pred_predict_test.ipynb  - Predict in the test set using previously trained models.
Mor_predict_age.ipynb  -  Simple age prediction with the protein data.
Mor_pred2_results_paper_plots.ipynb  -  One notebook with all the plots used in the paper (or directions to where the plot was made)
Mor_pred2_results_paper_plots_final_size_pdf.ipynb  -  Plotting the main manuscript figure in the correct size. 

Mor_features.py  - Feature selection
Mor_predict_forward.py   -  Forward feature selection(ranking), bootstrapped version and more
Mor_predict_bootstrap.py  -  Feature rankning by bootstrapped Lasso model
Mor_predict_cv.py  -  Train and predict cross validation on training set
Mor_predict.py -  Train prediction models
Mor_univariate.py  -  Univariate protein phenotype associations
Mor_predict_single_protein.py  -  Train prediction models with single proteins

association_functions.py - Functions used in Mor_pred_associations.ipynb to correct and calculate associations between predictions and phenotypes.
Calculate_score.py - Functions to calculate predictions scores.
helpers_pat.py - Class for hyperparameter tuning using hyperopt for testing XGB and MLP. Code by Pat Zhang, Amgen, 2019.
Predict_functions.py - Functions with all the used prediction methods, in most cases just a wrapper around common scikit-learn functions.
R_functions.py - A wrapper using rpy2 to access R functions.
