import numpy as np
from dtree import *
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

def bootstrap(X, y):
	n = len(y)
	idx = np.random.randint(0, n, size=n)
	X_train = X[idx]
	y_train = y[idx]
	mask = np.ones(n, dtype=bool)
	mask[idx] = False
	return X_train, y_train, mask


class RandomForest:
	def __init__(self, n_estimators=10, oob_score=False):
		self.n_estimators = n_estimators
		self.oob_score = oob_score
		self.oob_score_ = np.nan

	def fit(self, X, y):
		"""
		Given an (X, y) training set, fit all n_estimators trees to different,
		bootstrapped versions of the training data.  Keep track of the indexes of
		the OOB records for each tree.  After fitting all of the trees in the forest,
		compute the OOB validation score estimate and store as self.oob_score_, to
		mimic sklearn.
		"""
		self.nunique = np.unique(y)
		oob_indices = []
		if self.name == 'regressor':
			for _ in range(self.n_estimators):
				X_train, y_train, oob_mask = bootstrap(X, y)
				t = RegressionTree(min_samples_leaf=self.min_samples_leaf, max_features = self.max_features)
				t.fit(X_train, y_train)
				self.trees.append(t)
				if self.oob_score:
					oob_indices.append(oob_mask)
		else:
			for _ in range(self.n_estimators):
				X_train, y_train, oob_mask = bootstrap(X, y)
				t = ClassifierTree(min_samples_leaf=self.min_samples_leaf, max_features = self.max_features)
				t.fit(X_train, y_train)
				self.trees.append(t)
				if self.oob_score:
					oob_indices.append(oob_mask)
		if self.oob_score:
			self.oob_score_ = self.compute_oob_score(X, y, oob_indices)


class RandomForestRegressor(RandomForest):
	def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
		super().__init__(n_estimators, oob_score=oob_score)
		self.min_samples_leaf = min_samples_leaf
		self.max_features = max_features
		self.trees = []
		self.name = 'regressor'

	def predict(self, X_test) -> np.ndarray:
		"""
		Given a 2D nxp array with one or more records, compute the weighted average
		prediction from all trees in this forest. Weight each trees prediction by
		the number of samples in the leaf making that prediction.  Return a 1D vector
		with the predictions for each input record of X_test.
		"""
		all_tree_weighted_sum = np.zeros(len(X_test))
		all_tree_Ns = np.zeros(len(X_test))
		for tree in self.trees:
		    all_leaves = tree.predict(X_test)
		    leaf_ns = np.array([leaf.n for leaf in all_leaves])
		    leaf_preds = np.array([leaf.prediction for leaf in all_leaves])
		    leaf_weighted_sum = leaf_preds*leaf_ns
		    all_tree_weighted_sum += leaf_weighted_sum
		    all_tree_Ns += leaf_ns
		return all_tree_weighted_sum/all_tree_Ns

	def score(self, X_test, y_test) -> float:
		"""
		Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
		collect the prediction for each record and then compute R^2 on that and y_test.
		"""
		y_predict = self.predict(X_test)
		return r2_score(y_test, y_predict)

	def compute_oob_score(self, X, y, oob_indices):
		all_unique_oob_idx = []
		for mask in oob_indices:
			all_unique_oob_idx.extend(list(np.where(mask==True)[0]))
		all_unique_oob_idx = np.unique(np.array(all_unique_oob_idx))
		oob_counts = np.zeros(len(X))
		oob_preds = np.zeros(len(X))
		for i, tree in enumerate(self.trees):
			X_oob = X[oob_indices[i]]
			all_leaves = tree.predict(X_oob)
			leaf_ns = np.array([leaf.n for leaf in all_leaves])
			leaf_preds = np.array([leaf.prediction for leaf in all_leaves])
			leaf_weighted_sum = leaf_preds*leaf_ns
			oob_preds[oob_indices[i]] += leaf_weighted_sum
			oob_counts[oob_indices[i]] += leaf_ns
		oob_avg_preds = oob_preds[all_unique_oob_idx] / oob_counts[all_unique_oob_idx]
		return r2_score(y[all_unique_oob_idx], oob_avg_preds)


class RandomForestClassifier(RandomForest):
	def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
		super().__init__(n_estimators, oob_score=oob_score)
		self.min_samples_leaf = min_samples_leaf
		self.max_features = max_features
		self.trees = []
		self.name = 'classifier'

	def predict(self, X_test) -> np.ndarray:
		class_sums = np.zeros((len(X_test), len(self.nunique)))
		for tree in self.trees:
			leaves = tree.predict(X_test)
			ys = [leaf.y for leaf in leaves]
			for i, col_ys in enumerate(ys):
				class_sum = np.array([len(col_ys[col_ys==class_val]) for class_val in self.nunique])
				class_sums[i] += class_sum
		return np.argmax(class_sums, axis=1)

	def score(self, X_test, y_test) -> float:
		"""
		Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
		collect the predicted class for each record and then compute accuracy between
		that and y_test.
		"""
		y_predict = self.predict(X_test)
		return accuracy_score(y_test, y_predict)

	def compute_oob_score(self, X, y, oob_indices):
		all_unique_oob_idx = []
		for mask in oob_indices:
			all_unique_oob_idx.extend(list(np.where(mask==True)[0]))
		all_unique_oob_idx = np.unique(np.array(all_unique_oob_idx))
		oob_counts = np.zeros(len(X))
		oob_preds = np.zeros((len(X), len(self.nunique)))
		for i, tree in enumerate(self.trees):
			X_oob = X[oob_indices[i]]
			all_leaves = tree.predict(X_oob)
			leaf_ns = np.array([leaf.n for leaf in all_leaves])
			leaf_preds = np.array([leaf.prediction for leaf in all_leaves])
			oob_preds[oob_indices[i], leaf_preds] += leaf_ns
			oob_counts[oob_indices[i]] += 1
		class_votes = np.argmax(oob_preds[all_unique_oob_idx], axis=1)
		return accuracy_score(y[all_unique_oob_idx], class_votes)

