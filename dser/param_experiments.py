# sample file to get best params

from sklearn.model_selection import train_test_split
from utils import *
from modelselection_conformal import ConformalRejectOptionGridSearchCV
from sklearn.neighbors import KNeighborsClassifier

knn_parameters = {'n_neighbors': [3, 5, 7, 10, 15]}

data_desc = 'heloc'
target = 'class'

# Load data
X, y = load_custom_data(data_desc, target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

model_search = ConformalRejectOptionGridSearchCV(model_class=KNeighborsClassifier, parameter_grid=knn_parameters, rejection_thresholds=reject_thresholds)
best_params = model_search.fit(X_train, y_train)
#print(**best_params["model_params"])

print(best_params["model_params"])
print(best_params["rejection_threshold"])