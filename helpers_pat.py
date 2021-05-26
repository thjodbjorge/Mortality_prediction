## Code by Pat Zhang, Amgen, 2019.
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, STATUS_OK, tpe, hp
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler

class HyperoptTuner:
	def __init__(self, estimator, scoring, cv=5, n_repeats=1, random_state=42):
		""" Class to perform hyperparameter tuning using hyperopt 

		    estimator - estimator class to tune
		    scoring - sklearn scoring metric to optimize on (str)
		    cv - number of folds in stratified k-fold for the cross-validation strategy (int)
		    n_repeats - number of repeats for the cross-validation strategy (int)
		    random_state - random_state (int)
		"""

		self.model = estimator
		self.scoring = scoring
		self.cv = cv
		self.n_repeats = n_repeats
		self.random_state = random_state

	def format_space_helper(self, label, val):
		""" Helper function to format the search space 

		    label - name of the parameter (str)
		    val - values to search over (list, array, or tuple)
		"""
		
		# If val is an array or list, use hp.choice (discrete values)
		if type(val) in [list, np.ndarray]:
			return hp.choice(label, val)

		# If val is a tuple of length 2, use hp.uniform (uniform over val[0], val[1])
		# If val is a tuple of length 3, use hp.quniform (discrete between val[0] and val[1], intervals of val[2]) 
		elif type(val) == tuple:
			if len(val) == 2:
	    			return hp.uniform(label, *val)
			if len(val) == 3:
	   	 		return hp.quniform(label, *val)

	def format_space(self, space):
		""" Format the input space (dict) for hyperopt using the helper function """ 

		hp_space = {label: self.format_space_helper(label, val) for label, val in space.items()}
		return hp_space

	def cross_validation(self, model, x, y, scoring):
		""" Helper function to perform cross validation 

		    model - estimator to fit
		    x - data to fit
		    y - target labels 
		    scoring - sklearn scoring metric (str)
		"""

		scores = []
		np.random.seed(self.random_state)
		for seed in np.random.choice(100, self.n_repeats, replace=False):
			cv_score = cross_val_score(model, x, y, scoring=scoring, 
					       	   cv=StratifiedKFold(self.cv, shuffle=True, random_state=seed))
			scores.append(cv_score)
		return [np.mean(s) for s in scores]

	def objective(self, params):
		""" Objective function for hyperopt to minimize """

		clf = self.model(**params, random_state=self.random_state)
		score = self.cross_validation(clf, self.x, self.y, scoring=self.scoring)
		return 1 - np.mean(score) 

	def optimize(self, x, y, space, max_evals=100):
		""" Function to run the hyperopt optimization

		    x - data to fit
		    y - target labels
		    space - parameter space to optimize within (dict where key is parameter name, and value is list, array, or tuple)
		    max_evals - number of runs for optimizatoin (default is 100) 
		"""

		self.x = x
		self.y = y
		hp_space = self.format_space(space)

		trials = Trials()
		bst = fmin(fn=self.objective, space=hp_space, algo=tpe.suggest, trials=trials, max_evals=max_evals,
			   rstate=np.random.RandomState(self.random_state), return_argmin=False)
		return bst 
