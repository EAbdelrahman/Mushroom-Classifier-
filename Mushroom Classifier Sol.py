import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
#importing data 

fileCSV = pd.read_csv('data/agaricus-lepiota.data')

#Test if it's read 

#print(file.head())

# Adding column names

fileCSV.columns = ['class','cap_shape','cap_surface','cap_color','bruises','odor','gill_attachment','gill_spacing','gill_size'
                ,'gill_color','stalk_shape','stalk_root','stalk_surface_above_ring','stalk_surface_below_ring',
                'stalk_color_above_ring','stalk_color_below_ring','veil_type','veil_color','ring_number','ring_type',
                'spore_print_color','population','habitat']

#H Handling Missing Data
# Eliminating the feature that has missing data

fileCSV.pop('stalk_root')

# splitting out target and features

features = fileCSV.iloc[:,1:]
target = fileCSV['class']

# Test

#print(features.head())
#print(target.head())

# Features encoding

encoder1 = OrdinalEncoder()
encoded_features = encoder1.fit_transform(features) 

#print(encoded_features)

# Target encoding

encoder2 = LabelEncoder()
encoded_target = encoder2.fit_transform(target)

#print(encoded_target)


#Splitting Data
# f for feature 
# t for target


f_train,f_test,t_train,t_test = train_test_split(encoded_features,encoded_target,test_size=0.2
                                                 ,stratify=encoded_target,random_state=1)

#Create descision tree classifier

DT_Class = DecisionTreeClassifier()

# Making grid search

criterion = ['gini','entropy']
max_depth = [2,5,6,9,12,13,14,16]
DT_parameters = dict(criterion=criterion, max_depth=max_depth)

Grid_Search1 = GridSearchCV(estimator=DT_Class,param_grid=DT_parameters,scoring='accuracy',cv=10)

print(Grid_Search1.fit(f_train,t_train))

# Get the best parameters to build our descision tree
print("best decision tree  parameters : ", Grid_Search1.best_params_)
print("best decision tree accuracy score : ", Grid_Search1.best_score_)

#in this case best parameters are  criterion (gini) &  max depth is (9)
# and accurecy is 100%

# Building our descision tree with the best parameters 

DT_Class = DecisionTreeClassifier(criterion="gini", max_depth=9)
DT_Class.fit(f_train,t_train)
t_predictDT = DT_Class.predict(f_test)

print(t_predictDT)

# creating a KNN classifier

#KNN_Class = KNeighborsClassifier(n_neighbors=5, p=2)
KNN_Class = KNeighborsClassifier()
# Doing a grid search 

near_neighbors = [3,5,7,11]
p = [2,4,5,9,10]
KNN_parameters = dict(n_neighbors=near_neighbors, p=p)
Grid_Search2 = GridSearchCV(estimator=KNN_Class,param_grid=KNN_parameters,scoring='accuracy',cv=10)
print(Grid_Search2.fit(f_train,t_train))

print("best KNN parameters :" , Grid_Search2.best_params_)
print("best KNN accuracy score :" , Grid_Search2.best_score_)

#in this case best parameters are n_neigbors (3) and p (2)
# and the accurecy of the model is 0.998922745

# Create A CLASSIFIER with the best parameters
KNN_Class = KNeighborsClassifier(n_neighbors=3, p=2)
KNN_Class.fit(f_train,t_train)
t_predictKNN = KNN_Class.predict(f_test)
print(t_predictKNN)

# Evaluating Models using confusion matrix

DT_ConfMat = confusion_matrix(t_test,t_predictDT)
'''
[[842,0]
[0,783]]
'''
KNN_ConfMat = confusion_matrix(t_test,t_predictKNN)
'''
[[841,1],
[1,782]]
'''
tot = DT_ConfMat.sum()

DT_accurecy = ((DT_ConfMat[0][0]+DT_ConfMat[1][1])/tot)*100

KNN_accurecy = ((KNN_ConfMat[0][0]+KNN_ConfMat[1][1])/tot)*100

print(DT_ConfMat)
print(KNN_ConfMat)
print(DT_accurecy)
print(KNN_accurecy)
