#!/usr/bin/python
import pprint
import sys
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
import selectKbest as kbest
plt.style.use('ggplot')
import featureCreator as fc
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
featuresList = [
					'poi','exercised_stock_options', 'total_stock_value', 'bonus',
					'salary', 'deferred_income', 'long_term_incentive',
					'restricted_stock', 'total_payments', 'shared_receipt_with_poi',
					'loan_advances', 'expenses', 'from_poi_to_this_person', 'from_this_person_to_poi'
				]

#print("Features list:\n")
pprint.pprint(featuresList)

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

# Total number of data points
#print('There are {} people in the dataset.'.format(len(data_dict)))

# Allocation across classes (POI/non-POI)
poi_counts = defaultdict(int)
for featureValues in data_dict.values():
    poi_counts[featureValues['poi']] += 1
#print('There are {} POIs and {} non-POI''s.'.format(poi_counts[True], poi_counts[False]))

# Names of features

list(list(data_dict.values())[0].keys())

# Features with missing values
nanCountPoi = defaultdict(int)
nanCountNonPoi = defaultdict(int)

#find the poi that has no data assocaited to it.

def doFilterNaNValues():
    for dataPoint in data_dict.values():
        if dataPoint['poi'] == True:
            for feature, value in dataPoint.items():
                if value == "NaN":
                    nanCountPoi[feature] += 1
        elif dataPoint['poi'] == False:
            for feature, value in dataPoint.items():
                if value == "NaN":
                    nanCountNonPoi[feature] += 1
        else:
            print('Got an uncategorized person.')

    nanCountDf = pd.DataFrame([nanCountPoi, nanCountNonPoi]).T
    nanCountDf = nanCountDf.fillna(pd.DataFrame.mean(nanCountDf))
    nanCountDf.columns = ['# NaN in POIs', '# NaN in non-POIs']
    nanCountDf['# NaN total'] = nanCountDf['# NaN in POIs'] + nanCountDf['# NaN in non-POIs']
    nanCountDf['% NaN in POIs'] = nanCountDf['# NaN in POIs'] / poi_counts[True] * 100
    nanCountDf['% NaN in non-POIs'] = nanCountDf['# NaN in non-POIs'] / poi_counts[False] * 100
    nanCountDf['% NaN total'] = nanCountDf['# NaN total'] / len(data_dict) * 100
    # visualize data
    # nanCountDf = nanCountDf.cumsum()
    # nanCountDf.plot()
    # plt.show()


doFilterNaNValues()



### Task 2: Remove outliers

data_dict.pop("TOTAL", 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)
data_dict.pop("LOCKHART EUGENE E", 0)


### Task 3: Create new feature(s)

'''
below functions creates features:
'exercised_stock_ratio',
'poi_email_ratio'

'''
featuresList = fc.CreateExercisedStockRatio(data_dict,featuresList)
featuresList = fc.CreatePoiEmailRatio(data_dict,featuresList)

### Store to my_dataset for easy export below.
myDataSet = data_dict
#print('data set has: ',my_dataset)

### Extract features and labels from dataset for local testing

data = featureFormat(myDataSet, featuresList, sort_keys = True)
#print('data extract features has : ', data)
labels, features = targetFeatureSplit(data)

'''
    Lets try scaling the features according to min/max values
    input: featuresList
        
'''
def doScaleData():
    from sklearn.preprocessing import MinMaxScaler
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    scaler = MinMaxScaler()
    minMaxfeaturesList = scaler.fit_transform(features)

    #print ('scaled data : \n',minMaxfeaturesList)
    return minMaxfeaturesList

#scaledFeatures = doScaleData()

### Task 4: Try a varity of classifiers

'''
    returns transformed pca object
    answers question: who are the 'big shots' at enron
    reduces noise - get the most 'noisy'
'''
def doPCA(data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=8)
    pca.fit(data)
    return pca

#get the pca object
pca=doPCA(data)

#define estimator
estimators = [('reduce_dim', PCA(n_components=pca.n_components)), ('clf', DecisionTreeClassifier(min_samples_split=3))]

#define clf
clf = Pipeline(estimators)

#transform data
transformed_data = pca.transform(data)
#print ('explained variance ratio : ', pca.explained_variance_ratio_)

first_pc = pca.components_[0]
second_pc = pca.components_[1]

'''
    input: clf object
    returns: decision tree classifier object
'''

'''
    input: #PCA'fied data
    returns classifier
'''


def doDecisionTree():

    clf = DecisionTreeClassifier(min_samples_split=8)

    DtFeaturesList = ['poi', 'exercised_stock_options', 'poi_email_ratio']

    #doPlot(myDataSet,DtFeaturesList)

    minMaxfeaturesList = doScaleData()


    features_train, minMaxfeaturesList, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    clf.fit(features_train, labels_train)

    pred = clf.predict(features_test)

    test_classifier(clf, myDataSet, DtFeaturesList, folds=1000)



    print(accuracy_score(labels_test, pred))
    print(precision_score(labels_test, pred))
    print(recall_score(labels_test, pred))
    print(f1_score(labels_test, pred))
    print(classification_report(labels_test, pred))
    print(confusion_matrix(labels_test, pred))

    return clf

'''
    input: clf object
    returns: adaboost classifier object
    Non PCA
'''
def doAdaBoost ():
    clf = AdaBoostClassifier(n_estimators=100)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    clf.fit(features_train, labels_train)

    pred = clf.predict(features_test)

    test_classifier(clf, myDataSet, featuresList, folds=1000)

    print(accuracy_score(labels_test, pred))
    print(precision_score(labels_test, pred))
    print(recall_score(labels_test, pred))
    print(f1_score(labels_test, pred))
    print(classification_report(labels_test, pred))
    print(confusion_matrix(labels_test, pred))


    return clf

'''
    input: clf object
    returns: K-Nearest Neighbors object
    Non PCA
'''
def doKneighbors():
    clf = KNeighborsClassifier(n_neighbors=7)

    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    clf.fit(features_train, labels_train)

    pred = clf.predict(features_test)

    test_classifier(clf, myDataSet, featuresList, folds=1000)

    print(accuracy_score(labels_test, pred))
    print(precision_score(labels_test, pred))
    print(recall_score(labels_test, pred))
    print(f1_score(labels_test, pred))
    print(classification_report(labels_test, pred))
    print(confusion_matrix(labels_test, pred))

    return clf


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 


from sklearn.cross_validation import train_test_split

clf.fit(features_train, labels_train)

'''
    plots data
    input data set
'''
def doPlot(dataSet,features):
    import pandas as pd
    import matplotlib.pyplot as plt1

    #build df from dataset
    t_dataDF = pd.DataFrame.from_dict(dataSet)


    columSet = {}

    for k in iter(t_dataDF.keys()):
        if t_dataDF[k]['poi']:
            exercised_stock_options = t_dataDF[k]['exercised_stock_options']
            poi_email_ratio = t_dataDF[k]['poi_email_ratio']
            try:
                if exercised_stock_options > 0 and poi_email_ratio > 0:
                    columSet.__setitem__(exercised_stock_options,poi_email_ratio)
            except:
                    print('could not process: ' , t_dataDF[k])

    xaxis = list(columSet.keys())
    yaxis = list(columSet.values())


    #plt1.xlim(min(xaxis), max(xaxis))
    #plt1.ylim(min(yaxis), max(yaxis))

    plt1.scatter(xaxis, yaxis)
    plt1.xlabel('exercised_stock_options')
    plt1.ylabel('poi_email_ratio')
    plt1.show()


def doPlotSalaryAndBonus():
    import matplotlib.pyplot as plt2
    import pandas as pd

    sb_dataDF = pd.DataFrame.from_dict(myDataSet)

    print(sb_dataDF)

    salBonDic = {}
    for k in iter(sb_dataDF.keys()):
        try:
            if sb_dataDF[k]['salary'] > 0 and sb_dataDF[k]['salary'] != 'NaN':
                if sb_dataDF[k]['bonus'] != 'NaN':
                    salBonDic.__setitem__(sb_dataDF[k]['salary'], sb_dataDF[k]['bonus'])

        except:
            print ('in function doPlotSalaryAndBonus, something happend. Probably uncomparable data: ', sb_dataDF[k]['salary'] , ' and ' , '0 could not be compared. Ignoring issue and proceeding')
            continue


    salaryCol = []
    for k in iter(salBonDic.keys()):
        salaryCol.append(k)

    bonusCol = []
    for v in iter(salBonDic.values()):
        bonusCol.append(v)

    
    plt2.ylim(min(bonusCol),max(bonusCol))
    plt2.xlim(min(salaryCol),max(salaryCol))

    plt2.scatter(salaryCol , bonusCol)
    plt2.xlabel('salary')
    plt2.ylabel('bonus')
    plt2.show()


def doExercisedStockOptionsLongTermIncentive():
    import matplotlib.pyplot as plt3
    import pandas as pd

    es_dataDF = pd.DataFrame.from_dict(myDataSet)

    print(es_dataDF)

    exersiseLongtermIncetiveDic = {}
    for k in iter(es_dataDF.keys()):
        try:
            if es_dataDF[k]['long_term_incentive'] > 0 and es_dataDF[k]['long_term_incentive'] != 'NaN':
                if es_dataDF[k]['exercised_stock_options'] != 'NaN':
                    exersiseLongtermIncetiveDic.__setitem__(es_dataDF[k]['long_term_incentive'], es_dataDF[k]['exercised_stock_options'])

        except:
            print('in function doExercisedStockOptionsLongTermIncentive, something happend. Probably uncomparable data: ',
                  es_dataDF[k]['long_term_incentive'], ' and ', '0 could not be compared. Ignoring issue and proceeding')
            continue

    long_term_incentiveCol = []
    for k in iter(exersiseLongtermIncetiveDic.keys()):
        long_term_incentiveCol.append(k)

    exercised_stock_optionsCol = []
    for v in iter(exersiseLongtermIncetiveDic.values()):
        exercised_stock_optionsCol.append(v)

    plt3.xlim(min(long_term_incentiveCol), max(long_term_incentiveCol))
    plt3.ylim(min(exercised_stock_optionsCol), max(exercised_stock_optionsCol))

    plt3.scatter(long_term_incentiveCol, exercised_stock_optionsCol)
    plt3.xlabel('Long Term Incetive')
    plt3.ylabel('Exercised Stock Options')
    plt3.show()

#show plotting
#doPlot()
#doPlotSalaryAndBonus()
#doExercisedStockOptionsLongTermIncentive()

################################## DRIVER FUNCTIONS ##################################

#run kbest default is 10
#clf = kbest.SelectKbest(data_dict,featuresList,14)


clf = doDecisionTree()

#clf = doAdaBoost()

#clf = doKneighbors()


### Task 6: Dump your classifier, dataset, and features_list so anyone can

dump_classifier_and_data(clf, myDataSet, featuresList)