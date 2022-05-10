import numpy as np
from scipy.stats import mode
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

# class DecisionNode:
#     def __init__(self, col, split, lchild, rchild):
#         self.col = col
#         self.split = split
#         self.lchild = lchild
#         self.rchild = rchild

#     def predict(self, x_test):
#         # Make decision based upon x_test[col] and split
#         if x_test[self.col] <= self.split:
#         	return self.lchild.predict(x_test)
#         return self.rchild.predict(x_test)

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild
    def predict(self, x_test):
        return self.leaf(x_test)
    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] <= self.split:
        	return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)


class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        self.y = y

    def predict(self, x_test):
        return self
        


class DecisionTree621:
	def __init__(self, min_samples_leaf=1, max_features=0.3, loss=None):
		self.min_samples_leaf = min_samples_leaf
		self.loss = loss # loss function; either np.std or gini
		self.max_features = max_features

	def bestsplit(self, X, y, loss):
		best = (-1, -1, loss(y))
		p = X.shape[1]
		col_indices = np.random.choice(range(p), int(np.round(self.max_features*p)), False)
		for col_idx in col_indices:
			if len(X[:, col_idx]) < 11:
				candidates = X[:, col_idx]
			else:
				candidates = np.random.choice(X[:, col_idx], 11, replace=False)
			for split in candidates:
				y_l = y[X[:, col_idx] <= split]
				y_r = y[X[:, col_idx] > split]
				if len(y_l) < self.min_samples_leaf or len(y_r) < self.min_samples_leaf:
					continue
				l = ((len(y_l) * loss(y_l)) + (len(y_r) * loss(y_r)))/len(y)
				if l == 0:
					return col_idx, split
				if l < best[2]:
					best = (col_idx, split, l)
		return best[0], best[1]

	def fit(self, X, y):
		"""
		Create a decision tree fit to (X,y) and save as self.root, the root of
		our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
		predict the most common class (the mode) and regressors predict the average y
		for samples in that leaf.  
		      
		This function is a wrapper around fit_() that just stores the tree in self.root.
		"""
		self.root = self.fit_(X, y)
        
	def fit_(self, X, y):
		"""
		Recursively create and return a decision tree fit to (X,y) for
		either a classifier or regressor.  This function should call self.create_leaf(X,y)
		to create the appropriate leaf node, which will invoke either
		RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
		on the type of self.

		This function is not part of the class "interface" and is for internal use, but it
		embodies the decision tree fitting algorithm.

		(Make sure to call fit_() not fit() recursively.)
		"""
		if len(X) <= self.min_samples_leaf:
			return self.create_leaf(y)
		col, split = self.bestsplit(X, y, self.loss)
		if col == -1:
			return self.create_leaf(y)
		lchild = self.fit_(X[X[:, col] <= split], y[X[:, col] <= split])
		rchild = self.fit_(X[X[:, col] > split], y[X[:, col] > split])
		return DecisionNode(col, split, lchild, rchild)
        
	def predict(self, X_test):
		"""
		Make a prediction for each record in X_test and return as array.
		This method is inherited by RegressionTree621 and ClassifierTree621 and
		works for both without modification!
		"""
		predictions = []
		for row in X_test:
			pred = self.root.predict(row)
			predictions.append(pred)
		return np.array(predictions)


class RegressionTree621(DecisionTree621):
	def __init__(self, min_samples_leaf=1, max_features=0.3):
		super().__init__(min_samples_leaf, max_features, loss=np.std)

	def score(self, X_test, y_test):
		"Return the R^2 of y_test vs predictions for each record in X_test"
		y_predict = self.predict(X_test)
		return r2_score(y_test, y_predict)

	def create_leaf(self, y):
		"""
		Return a new LeafNode for regression, passing y and mean(y) to
		the LeafNode constructor.
		"""
		return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
	def __init__(self, min_samples_leaf=1, max_features=0.3):
		super().__init__(min_samples_leaf, max_features, loss=gini)

	def score(self, X_test, y_test):
		"Return the accuracy_score() of y_test vs predictions for each record in X_test"
		y_predict = self.predict(X_test)
		return accuracy_score(y_test, y_predict)

	def create_leaf(self, y):
		"""
		Return a new LeafNode for classification, passing y and mode(y) to
		the LeafNode constructor.
		"""
		return LeafNode(y, mode(y)[0][0])

def gini(y):
	"Return the gini impurity score for values in y"
	all_p = []
	for c in np.unique(y):
		p_c = len(y[y==c]) / len(y)
		all_p.append(p_c)
	all_p_sq = (np.array(all_p))**2
	return 1 - np.sum(all_p_sq)




		


