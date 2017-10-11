import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# Loading the training set
training_set = pd.read_csv("fashion-mnist_train.csv", dtype='int')

# Preprocessing the data: feature selection
features = training_set.columns[1:]
X_train = training_set.drop('label', axis = 1)
Y_train = training_set['label']

clf = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                             max_depth=50, random_state=0, n_jobs=-1)
clf.fit(X_train, Y_train)

# Selector to identify important features that have weight > 0.0007
sfm = SelectFromModel(clf, threshold=0.0007)
sfm.fit(X_train, Y_train)
X_selected_train = sfm.transform(X_train)

# Creating classifier with important features
clf_selected = RandomForestClassifier(n_estimators=100, max_features='sqrt',
                             max_depth=50, random_state=0, n_jobs=-1)
clf_selected.fit(X_selected_train, Y_train)


# Predicting labels on the test set
test_set = pd.read_csv("test_data.csv", dtype='int')
X_test = test_set[features]
X_selected_test = sfm.transform(X_test)
Y_sel_predicted = clf_selected.predict(X_selected_test)

# Creating the submission
rf_sel_submit = pd.DataFrame({'ids': test_set['ids'],
                             'label': Y_sel_predicted}, dtype=int)
rf_sel_submit.to_csv('rf_predictions.csv', index=False)