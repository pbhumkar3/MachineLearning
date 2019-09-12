
import random
import numpy as np
import pandas as pd
import io
import shutil
import argparse
import xml.etree.ElementTree as ET
import os.path
from os import listdir , makedirs
from os.path import isfile,join, exists
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc

def get_text_and_userid(files_list, file_path):
    text_buffer = []
    for file in files_list:
        user_id = file.replace(".txt","")

        with io.open(file_path + "/" + file, 'r' , encoding ='latin-1') as file_obj:
          text = file_obj.read()
          text_buffer.append({'userid': user_id, 'transcript': text})
          file_obj.close()
    return text_buffer


def copy_training_testing_data(profile_train_data, profile_test_data, test_text_dir, train_text_dir):
    textfilestrain = [f for f in listdir(train_text_dir) if isfile(join(train_text_dir, f))]
    textfilestest = [f for f in listdir(test_text_dir) if isfile(join(test_text_dir, f))]

    # getting text from text file and building a dataFrame
    TextDFtrain = pd.DataFrame(get_text_and_userid(textfilestrain, train_text_dir))
    TextDFtest = pd.DataFrame(get_text_and_userid(textfilestest, test_text_dir))

    # Creating complete data for training model
    CompleteDataTrain = pd.merge(TextDFtrain, profile_train_data, on='userid')
    CompleteDataTest = pd.merge(TextDFtest, profile_test_data, on='userid')
    return CompleteDataTrain, CompleteDataTest

def training_testing_ML_gender(CompleteDataTrain,CompleteDataTest):
    # Preparing the train and test data for Gender Prediction
    data_gender = CompleteDataTrain.loc[:, ['transcript', 'gender']]

    n = 1500
    all_Ids = np.arange(len(data_gender))
    random.shuffle(all_Ids)
    test_Ids = all_Ids[0:n]
    train_Ids = all_Ids[n:]
    data_test = data_gender.loc[test_Ids, :]
    data_train = data_gender.loc[train_Ids, :]

    # Training a Naive Bayes model
    count_vect = CountVectorizer()
    X_train = count_vect.fit_transform(data_train['transcript'])
    y_train = data_train['gender']
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # Testing the Naive Bayes model
    X_test = count_vect.transform(data_test['transcript'])
    y_test = data_test['gender']
    y_predicted = clf.predict(X_test)
    print("Accuracy Gender: %.2f" % accuracy_score(y_test, y_predicted))

    CompleteDataTest['gender'] = clf.predict(count_vect.transform(CompleteDataTest['transcript']))
    classes = [0.0 , 1.0]
    cnf_matrix = confusion_matrix(y_test, y_predicted, labels=classes)
    print("Confusion matrix:")
    print(cnf_matrix)
    return CompleteDataTest


def get_openness_prediction(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.ope
    test = liwc_test_data[LIWC_features]
    opeAvg = 0
    # linreg = LinearRegression()
    linreg = LinearRegression()
    openness_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in openness_kf.split(X, y):
        X_train, X_test = X.loc[train_index,], X.loc[test_index,]
        y_train, y_test = y.loc[train_index,], y.loc[test_index,]
        linreg.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, linreg.predict(X_test)))
        opeAvg = opeAvg + error
    y_predict = linreg.predict(test)
    print("Predicted RMSE ope:", round((opeAvg / 10), 2))
    return dict(zip(profile_test_data.userid, y_predict))



def get_extroversion_prediction(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.ext
    test = liwc_test_data[LIWC_features]
    opeAvg = 0
    # linreg = LinearRegression()
    linreg = LinearRegression()
    openness_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in openness_kf.split(X, y):
        X_train, X_test = X.loc[train_index,], X.loc[test_index,]
        y_train, y_test = y.loc[train_index,], y.loc[test_index,]
        linreg.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, linreg.predict(X_test)))
        opeAvg = opeAvg + error
    y_predict = linreg.predict(test)
    print("Predicted RMSE ext:", round((opeAvg / 10), 2))
    return dict(zip(profile_test_data.userid, y_predict))


def get_neurotic_prediction(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.neu
    test = liwc_test_data[LIWC_features]
    opeAvg = 0
    # linreg = LinearRegression()
    linreg = LinearRegression()
    openness_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in openness_kf.split(X, y):
        X_train, X_test = X.loc[train_index,], X.loc[test_index,]
        y_train, y_test = y.loc[train_index,], y.loc[test_index,]
        linreg.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, linreg.predict(X_test)))
        opeAvg = opeAvg + error
    y_predict = linreg.predict(test)
    print("Predicted RMSE neu:", round((opeAvg / 10), 2))
    return dict(zip(profile_test_data.userid, y_predict))


def get_agreeable_prediction(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    # training_data.to_csv('/Users/abhi/c:temp/tcss555/training/testfile.csv', sep=',', encoding='utf-8')
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.agr
    test = liwc_test_data[LIWC_features]
    opeAvg = 0
    # linreg = LinearRegression()
    linreg = LinearRegression()
    openness_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in openness_kf.split(X, y):
        X_train, X_test = X.loc[train_index,], X.loc[test_index,]
        y_train, y_test = y.loc[train_index,], y.loc[test_index,]
        linreg.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, linreg.predict(X_test)))
        opeAvg = opeAvg + error
    y_predict = linreg.predict(test)
    print("Predicted RMSE agr:", round((opeAvg / 10), 2))
    return dict(zip(profile_test_data.userid, y_predict))

def get_conscientious_prediction(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.con
    test = liwc_test_data[LIWC_features]
    opeAvg = 0
    # linreg = LinearRegression()
    linreg = LinearRegression()
    openness_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in openness_kf.split(X, y):
        X_train, X_test = X.loc[train_index,], X.loc[test_index,]
        y_train, y_test = y.loc[train_index,], y.loc[test_index,]
        linreg.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, linreg.predict(X_test)))
        opeAvg = opeAvg + error
    y_predict = linreg.predict(test)
    print("Predicted RMSE con:", round((opeAvg / 10), 2))
    return dict(zip(profile_test_data.userid, y_predict))



def get_openness_trait_prediction_gradient(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.ope
    opeAvg = 0
    test = liwc_test_data[LIWC_features]
    gradient_model = GradientBoostingRegressor(n_estimators=250, max_depth=1, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
    ope_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in ope_kf.split(X, y):
        X_train, X_test = X.ix[train_index,], X.ix[test_index,]
        y_train, y_test = y.ix[train_index,], y.ix[test_index,]
        gradient_model.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, gradient_model.predict(X_test)))
        opeAvg = opeAvg + error

    print("Predicted RMSE ope:", round((opeAvg / 10), 2))
    y_predict = gradient_model.predict(test)
    profile_test_data.ope = y_predict
    return dict(zip(profile_test_data.userid, y_predict))


def get_extroversion_trait_prediction_gradient(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.ext
    opeAvg = 0
    test = liwc_test_data[LIWC_features]
    gradient_model = GradientBoostingRegressor(n_estimators=250, max_depth=1, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
    ope_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in ope_kf.split(X, y):
        X_train, X_test = X.ix[train_index,], X.ix[test_index,]
        y_train, y_test = y.ix[train_index,], y.ix[test_index,]
        gradient_model.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, gradient_model.predict(X_test)))
        opeAvg = opeAvg + error

    print("Predicted RMSE ext:", round((opeAvg / 10), 2))
    #gradient_model.fit(X, y)
    y_predict = gradient_model.predict(test)
    profile_test_data.ext = y_predict
    return dict(zip(profile_test_data.userid, y_predict))


def get_neurotic_trait_prediction_gradient(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.neu
    opeAvg = 0
    test = liwc_test_data[LIWC_features]
    gradient_model = GradientBoostingRegressor(n_estimators=250, max_depth=1, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
    ope_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in ope_kf.split(X, y):
        X_train, X_test = X.ix[train_index,], X.ix[test_index,]
        y_train, y_test = y.ix[train_index,], y.ix[test_index,]
        gradient_model.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, gradient_model.predict(X_test)))
        opeAvg = opeAvg + error

    print("Predicted RMSE neu:", round((opeAvg / 10), 2))
    #gradient_model.fit(X, y)
    y_predict = gradient_model.predict(test)
    return dict(zip(profile_test_data.userid, y_predict))


def get_agreeable_trait_prediction_gradient(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.agr
    opeAvg = 0
    test = liwc_test_data[LIWC_features]
    gradient_model = GradientBoostingRegressor(n_estimators=250, max_depth=1, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
    ope_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in ope_kf.split(X, y):
        X_train, X_test = X.ix[train_index,], X.ix[test_index,]
        y_train, y_test = y.ix[train_index,], y.ix[test_index,]
        gradient_model.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, gradient_model.predict(X_test)))
        opeAvg = opeAvg + error

    print("Predicted RMSE agr:", round((opeAvg / 10), 2))
    #gradient_model.fit(X, y)
    y_predict = gradient_model.predict(test)
    return dict(zip(profile_test_data.userid, y_predict))


def get_conscientious_trait_prediction_gradient(profile_train_data, liwc_train_data, profile_test_data, liwc_test_data):
    training_data = pd.merge(left=profile_train_data,right=liwc_train_data, how='left',left_on='userid',right_on= 'userId')
    training_data.drop('userId', axis=1)
    LIWC_features =[x for x in liwc_train_data.columns.tolist()]
    LIWC_features.remove('userId')
    X = training_data[LIWC_features]
    y = training_data.con
    opeAvg = 0
    test = liwc_test_data[LIWC_features]
    gradient_model = GradientBoostingRegressor(n_estimators=250, max_depth=1, learning_rate=.1, min_samples_leaf=9, min_samples_split=9)
    ope_kf = KFold(n_splits=10, shuffle=True)
    for train_index, test_index in ope_kf.split(X, y):
        X_train, X_test = X.ix[train_index,], X.ix[test_index,]
        y_train, y_test = y.ix[train_index,], y.ix[test_index,]
        gradient_model.fit(X_train, y_train)
        error = np.sqrt(metrics.mean_squared_error(y_test, gradient_model.predict(X_test)))
        opeAvg = opeAvg + error

    print("Predicted RMSE con:", round((opeAvg / 10), 2))
    #gradient_model.fit(X, y)
    y_predict = gradient_model.predict(test)
    return dict(zip(profile_test_data.userid, y_predict))



def get_average_data(CompleteDataTrain):
    age_range_xx_24 = 0
    age_range_25_34 = 0
    age_range_35_49 = 0
    age_range_50_xx = 0
    most_freq_gender = 1
    ope_sum = 0.0
    con_sum = 0.0
    ext_sum = 0.0
    agr_sum = 0.0
    neu_sum = 0.0

    for row in range( 0, len(CompleteDataTrain)):
        current_age = CompleteDataTrain.iloc[row]['age']
        #current_age = row['age']
        if current_age < 25:
            age_range_xx_24 += 1
        elif 25 <= current_age < 35:
            age_range_25_34 += 1
        elif 35 >= current_age < 50:
            age_range_35_49 += 1
        else:
            age_range_50_xx += 1

        current_ope = CompleteDataTrain.iloc[row]['ope']
        current_con = CompleteDataTrain.iloc[row]['con']
        current_ext = CompleteDataTrain.iloc[row]['ext']
        current_agr = CompleteDataTrain.iloc[row]['agr']
        current_neu = CompleteDataTrain.iloc[row]['neu']
        ope_sum += current_ope
        con_sum += current_con
        ext_sum += current_ext
        agr_sum += current_agr
        neu_sum += current_neu

    ope_avg = ope_sum / len(CompleteDataTrain)
    con_avg = con_sum / len(CompleteDataTrain)
    ext_avg = ext_sum / len(CompleteDataTrain)
    agr_avg = agr_sum / len(CompleteDataTrain)
    neu_avg = neu_sum / len(CompleteDataTrain)

    max_age_count = max(age_range_xx_24, age_range_25_34, age_range_35_49, age_range_50_xx)
    most_frequent_age_range = ""
    if age_range_xx_24 == max_age_count:
        most_frequent_age_range = "xx-24"
    elif age_range_25_34 == max_age_count:
        most_frequent_age_range = "25-34"
    elif age_range_35_49 == max_age_count:
        most_frequent_age_range = "35-49"
    else:
        most_frequent_age_range = "50-xx"

    return {"age_group": most_frequent_age_range,"gender": most_freq_gender, "open": ope_avg, "conscientious": con_avg,
            "extrovert": ext_avg, "agreeable": agr_avg, "neurotic": neu_avg}



#
# def write_data(output_dir,CompleteDataTest, averages):
#     print "--------------------------------------------------"
#
#     print "writing data into xml files into output dir " + output_dir
#
#     for x in range( 0, len(CompleteDataTest)):
#         id = CompleteDataTest.iloc[x]['userid']
#         age_group = str(averages.get('age_group'))
#         gender = str(CompleteDataTest.iloc[x]['gender'])
#         extrovert = str(averages.get('extrovert'))
#         neurotic = str(averages.get('neurotic'))
#         agreeble = str(averages.get('agreeable'))
#         conscientious = str(averages.get('conscientious'))
#         open_nature = str(averages.get('open'))
#
#         # create the file structure
#         data = ET.Element('user')
#         data.set('id', id)
#         data.set('age_group', age_group)
#         data.set('gender', gender)
#         data.set('extrovert', extrovert)
#         data.set('neurotic', neurotic)
#         data.set('agreeable', agreeble)
#         data.set('conscientious', conscientious)
#         data.set('open', open_nature)
#
#         record_data = ET.tostring(data)
#         record = open(output_dir + '/' + id, "w")
#         record.write(record_data)
#
#     print "writing data into xml files completed"
#     print "--------------------------------------------------"

# def fillup_test_data(averages , CompleteDataTest):
#     for x in range(0, len(CompleteDataTest)):
#         CompleteDataTest.iloc[x, CompleteDataTest.column['age']] = averages.get('age_group')
#         # CompleteDataTest.iloc[x]['age'] = averages.get('age_group')
#         # CompleteDataTest.iloc[x]['ope'] = averages.get('open')
#         # CompleteDataTest.iloc[x]['con'] = averages.get('conscientious')
#         # CompleteDataTest.iloc[x]['ext'] = averages.get('extrovert')
#         # CompleteDataTest.iloc[x]['agr'] = averages.get('agreeable')
#         # CompleteDataTest.iloc[x]['neu'] = averages.get('neurotic')
#     print CompleteDataTest.loc[:, ['age_group', 'gender']]
#     return CompleteDataTest


def write_data(output_dir, CompleteDataTest):
    print("--------------------------------------------------")

    print("writing data into xml files into output dir " + output_dir)
    # age_group = str('xx-24')
    # extrovert = str(3.486857894736829)
    # neurotic = str(2.7324242105263203)
    # agreeble = str(3.5839042105263155)
    # conscientious = str(3.445616842105264)
    # open_nature = str(3.9086905263157825)

    for x in range( 0, len(CompleteDataTest)):
        id = CompleteDataTest.iloc[x]['userid']
        gender = (CompleteDataTest.iloc[x]['gender'])
        age_group = 'xx-24'
        extrovert = str(CompleteDataTest.iloc[x]['extrovert'])
        neurotic = str(CompleteDataTest.iloc[x]['neurotic'])
        agreeble = str(CompleteDataTest.iloc[x]['agreeable'])
        conscientious = str(CompleteDataTest.iloc[x]['conscientious'])
        open_nature = str(CompleteDataTest.iloc[x]['open'])


        file_dir_name = os.path.join(output_dir, str(id) + '.xml')
        current_file = open(file_dir_name, "w+")
        current_file.write("<user\n")
        current_file.write("id=\"" + id + "\"\n")
        current_file.write("age_group=\"" + age_group + "\"\n")
        current_file.write("gender=\"")
        if gender == 0:
            current_file.write("male")
        else:
            current_file.write("female")
        current_file.write("\"\n")
        current_file.write("extrovert=\"" + extrovert + "\"\n")
        current_file.write("neurotic=\"" + neurotic + "\"\n")
        current_file.write("agreeable=\"" + agreeble + "\"\n")
        current_file.write("conscientious=\"" + conscientious + "\"\n")
        current_file.write("open=\"" + open_nature + "\"\n")
        current_file.write("/>")
        current_file.close()

    print("writing data into xml files completed")
    print("--------------------------------------------------")

def create_output_dir(output_dir):
  if exists(output_dir):
      shutil.rmtree(output_dir)
  makedirs(output_dir)

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input directory with trailing /', )
parser.add_argument('-o', type=str, help='output directory with trailing /')
args = parser.parse_args()

def main():
    output_dir = "/Users/abhi/c:temp/tcss555/public-test-data/output_folder"
    train_text_dir = '/Users/abhi/c:temp/tcss555/training/text/'
    test_text_dir = '/Users/abhi/c:temp/tcss555/public-test-data/text/'

    train_liwc_file = '/Users/abhi/c:temp/tcss555/training/LIWC/LIWC.csv'
    train_profile_file = '/Users/abhi/c:temp/tcss555/training/profile/profile.csv'
    test_profile_file = '/Users/abhi/c:temp/tcss555/public-test-data/profile/profile.csv'
    test_liwc_file = '/Users/abhi/c:temp/tcss555/public-test-data/LIWC/LIWC.csv'

    profile_train_data = pd.read_csv(train_profile_file)
    profile_test_data = pd.read_csv(test_profile_file)
    train_LIWC_data = pd.read_csv(train_liwc_file)
    test_LIWC_data = pd.read_csv(test_liwc_file)


    print ("-----------------using gradiant----------------")

    predicted_openness = get_openness_trait_prediction_gradient(profile_train_data, train_LIWC_data, profile_test_data,test_LIWC_data)
    print(predicted_openness)

    predicted_extoversion = get_extroversion_trait_prediction_gradient(profile_train_data, train_LIWC_data, profile_test_data,test_LIWC_data)
    print(predicted_extoversion)

    predicted_neurotic = get_neurotic_trait_prediction_gradient(profile_train_data, train_LIWC_data,profile_test_data, test_LIWC_data)
    print(predicted_neurotic)

    predicted_aggreable = get_agreeable_trait_prediction_gradient(profile_train_data, train_LIWC_data,profile_test_data, test_LIWC_data)
    print(predicted_aggreable)

    predicted_Conscientious = get_conscientious_trait_prediction_gradient(profile_train_data, train_LIWC_data,profile_test_data, test_LIWC_data)
    print(predicted_Conscientious)

    users = profile_test_data['userid']
    users_data_list = pd.DataFrame()
    for user in users:
        new_user = {}
        new_user['userid'] = user
        new_user['extrovert'] = round(predicted_extoversion[user], 2)
        new_user['neurotic'] = round(predicted_neurotic[user], 2)
        new_user['agreeable'] = round(predicted_aggreable[user], 2)
        new_user['conscientious'] = round(predicted_Conscientious[user], 2)
        new_user['open'] = round(predicted_openness[user], 2)
        users_data_list = users_data_list.append(new_user, ignore_index=True)


    print("-----------------using linear----------------")
    Openness_predict = get_openness_prediction(profile_train_data, train_LIWC_data, profile_test_data, test_LIWC_data)
    print(Openness_predict)

    extrovert_predict = get_extroversion_prediction(profile_train_data, train_LIWC_data, profile_test_data, test_LIWC_data)
    print(extrovert_predict)

    neurocratic_predict = get_neurotic_prediction(profile_train_data, train_LIWC_data, profile_test_data,test_LIWC_data)
    print(neurocratic_predict)

    agreeable_predict = get_agreeable_prediction(profile_train_data, train_LIWC_data, profile_test_data,test_LIWC_data)
    print(agreeable_predict)

    conscientious_predict = get_conscientious_prediction(profile_train_data, train_LIWC_data, profile_test_data,test_LIWC_data)
    print(conscientious_predict)

    users = profile_test_data['userid']
    users_data_list = pd.DataFrame()
    for user in users:
        new_user = {}
        new_user['userid'] = user
        new_user['extrovert'] = round(extrovert_predict[user], 2)
        new_user['neurotic'] = round(neurocratic_predict[user], 2)
        new_user['agreeable'] = round(agreeable_predict[user], 2)
        new_user['conscientious'] = round(conscientious_predict[user], 2)
        new_user['open'] = round(Openness_predict[user], 2)
        users_data_list = users_data_list.append(new_user, ignore_index=True)

    # test_relation_file = '/Users/abhi/c:temp/tcss555/public-test-data/relation/relation.csv'

    CompleteDataTrain, CompleteDataTest = copy_training_testing_data(profile_train_data, profile_test_data, test_text_dir, train_text_dir)
    # averages = get_average_data(CompleteDataTrain)
    CompleteDataTest = training_testing_ML_gender(CompleteDataTrain,CompleteDataTest)
    #CompleteDataTest = fillup_test_data(averages, CompleteDataTest)    #need to work on IMP

    completeTestData = pd.merge(users_data_list, CompleteDataTest, on='userid')

    create_output_dir(output_dir)
    write_data(output_dir, completeTestData)
   # write_data(output_dir, CompleteDataTest, averages)
    #convert_to_xml(users_data_list, args.output)

if __name__ == "__main__":
    #args = parse_arguments()
    main()
