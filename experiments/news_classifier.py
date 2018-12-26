"""
  20 News groups
"""
#from xgboost import XGBClassifier

from copy import deepcopy
import numpy as np
from scipy.stats import norm
from scipy import integrate
# Local imports
from mf_func import MFOptFunction
# For this
#from v2_news.news_classifier import get_kfold_val_score
#from v2_news import util
import sklearn
import warnings
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import math
#import xgboost as xgb

NUM_FOLDS = 5
NUM_JOBS = 2    ##### should be ideally equal to number of processors in your machine
problem_bounds_raw = [[1,2.2], [.5,1], [-6,-3],[-6,-3],[1.,5.2]] ########## Domain X for parameter tuning ########################



class NGMFOptFunction(MFOptFunction):
  """ MFOptFunction for SN data. """

  def __init__(self, fidel_bounds):
    """ Constructor. """
    self._load_data2()
    self.max_data = fidel_bounds[0,1]
    mf_func = self._mf_func
    fidel_cost_func = self._fidel_cost
    domain_bounds = (np.array(problem_bounds_raw))
    opt_fidel_unnormalised = np.array([self.max_data])
    super(NGMFOptFunction, self).__init__(mf_func, fidel_cost_func, fidel_bounds,
                                          domain_bounds, opt_fidel_unnormalised,
                                          vectorised=False)

  def _fidel_cost(self, z):
    """ cost function """
    return 0.01 + (z[0]/self.max_data)

  def _mf_func(self, z, x):
    print ("XXXXXXX", x)
    """ The MF Func. """
    #clf = SVC(C=np.exp(x[0]), kernel='rbf', gamma=np.exp(x[1]), tol=1e-20, max_iter=100)
    #clf = XGBClassifier()
    #clf = RandomForestClassifier()
    #clf = RandomForestClassifier(n_estimators=1000)
    #clf = RandomForestClassifier(max_features=85762)
    #max_depth = int(.001*z[0])
    #print("DEPTH ", max_depth)
    #clf = RandomForestClassifier(max_depth = max_depth)
    #min_samples_split = 0.0001
    #clf = RandomForestClassifier(min_samples_split=min_samples_split)
    #min_samples_leaf = 0.001
    #clf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)

    # , max_depth= int(math.log((z[0]*18846),2)), min_samples_split = .5, min_samples_leaf=0.5, n_jobs=2
    #print ("MIN SAMPLE LEAF", x[3], math.pow(10,x[3]))

    print(int(math.pow(10,x[0])), int(math.pow(10,x[1])*z[0]), math.pow(10,x[2]), math.pow(10,x[3]), int(math.pow(10,x[4])))
    clf = RandomForestClassifier(n_estimators=int(math.pow(10,x[0])), max_depth=int(math.pow(10,x[1])*z[0]) ,min_samples_split= math.pow(10,x[2]), min_samples_leaf= math.pow(10,x[3]),  max_features= int((math.pow(10,x[4]))), n_jobs=4)

    #clf = RandomForestClassifier()
    num_data_curr = int(z[0])
    feat_curr = self.features[1:num_data_curr]
    label_curr = self.labels[1:num_data_curr]
    print("STATE ", feat_curr.shape, len(label_curr), num_data_curr)
    return get_kfold_val_score(clf, feat_curr, label_curr)



  def _load_data2(self):
    print('Loading data ...')
    newsgroups_train = fetch_20newsgroups(subset='all')
    vectorizer = TfidfVectorizer()
    features = vectorizer.fit_transform(newsgroups_train.data)
    labels = newsgroups_train.target
    self.features,self.labels = deterministic_shuffle(features,labels)
    #print "NO OF FEATURE", self.features.shape
    #print "NO OF LABELS", len(self.labels)





def get_kfold_val_score(clf, X, Y, num_folds=None,random_seed = 512):
  st0 = np.random.get_state()
  np.random.seed(random_seed)
  if num_folds is None:
    num_folds = NUM_FOLDS
  max_folds = 5.0
  num_folds = int(min(max_folds, num_folds))
  Kf = KFold(n_splits = num_folds, shuffle = True, random_state = random_seed)
  acc = cross_val_score(clf,X = X,y = Y,cv=Kf,n_jobs=NUM_JOBS,scoring='accuracy')
  np.random.set_state(st0)
  return acc.mean()

def deterministic_shuffle(X, Y, random_seed=512):
  """ deterministically shuffles. """
  st0 = np.random.get_state()
  np.random.seed()
  idxs = np.random.permutation(len(Y))
  X = X[idxs]
  Y = Y[idxs]
  np.random.set_state(st0)
  print "DET SUFFLE"
  return X, Y
