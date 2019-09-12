import random
import numpy as np
import pandas as pd
import io
import argparse
import xml.etree.ElementTree as ET
from os import listdir
import os.path
from os.path import join as join
from os.path import isfile as isfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


def get_text_and_userid(files_list, file_path):
    text_buffer = []
    for file in files_list:
        user_id = file.replace(".txt","")

        with io.open(file_path + "/" + file, 'r' , encoding ='latin-1') as file_obj:
          text = file_obj.read()
          text_buffer.append({'userid': user_id, 'transcript': text})
          file_obj.close()
    return text_buffer


def copy_training_testing_data(profile_train_data, profile_test_data, train_text_dir, test_text_dir):
    textfilestrain = [f for f in listdir(train_text_dir) if isfile(join(train_text_dir, f))]
    textfilestest = [f for f in listdir(test_text_dir) if isfile(join(test_text_dir, f))]

    # getting text from text file and building a dataFrame
    TextDFtrain = pd.DataFrame(get_text_and_userid(textfilestrain, train_text_dir))
    TextDFtest = pd.DataFrame(get_text_and_userid(textfilestest, test_text_dir))

    # Creating complete data for training model
    CompleteDataTrain = pd.merge(TextDFtrain, profile_train_data, on='userid')
    CompleteDataTest = pd.merge(TextDFtest, profile_test_data, on='userid')
    return CompleteDataTrain , CompleteDataTest


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

def training_testing_gender_SGD(CompleteDataTrain,CompleteDataTest):

    gender_X = CompleteDataTrain['transcript']
    gender_Y = CompleteDataTrain['gender']
    sgdModel_gender = SGDClassifier(shuffle=True)
    count_vectorizer = CountVectorizer()
    # print(count_vectorizer.get_feature_names())
    tf_idf_transformer = TfidfTransformer()
    accuracy = 0
    age_kfold = KFold(n_splits=10, shuffle=True)
    for training_index, test_index in age_kfold.split(gender_X, gender_Y):
        gender_X_training, gender_X_test = gender_X.loc[training_index,], gender_X.loc[test_index,]
        gender_Y_training, gender_Y_test = gender_Y.loc[training_index,], gender_Y.loc[test_index,]
        gender_training_count = count_vectorizer.fit_transform(gender_X_training)
        gender_training_tf_idf = tf_idf_transformer.fit_transform(gender_training_count)
        sgdModel_gender.fit(gender_training_tf_idf, gender_Y_training)
        gender_test_count = count_vectorizer.transform(gender_X_test)
        gender_test_tfidf = tf_idf_transformer.transform(gender_test_count)
        gender_predicted = sgdModel_gender.predict(gender_test_tfidf)
        accuracy += accuracy_score(gender_Y_test, gender_predicted)
    # print("Accuracy Gender: %.2f" % accuracy_score(gender_Y_test,gender_predicted))
    print("Accuracy Gender SGD=", accuracy / 10)
    gender_test_count = count_vectorizer.transform(CompleteDataTest['transcript'])
    gender_test_tf_idf = tf_idf_transformer.transform(gender_test_count)
    predicted_gender_values = sgdModel_gender.predict(gender_test_tf_idf)
    # print("test_data=",test_data)
    CompleteDataTest['gender'] = (predicted_gender_values)
    return CompleteDataTest

def copy_test_train_data_Age(CompleteDataTrain , CompleteDataTest):
    # training_files_list = [f for f in listdir(train_text_dir) if isfile(join(train_text_dir, f))]
    # training_data_frame = pd.DataFrame(get_text_and_userid(training_files_list, train_text_dir))
    # training_data_age_original = pd.merge(training_data_frame, profile_train_data, on='userid')
    training_age_userid = CompleteDataTrain.loc[:, ['userid', 'age']]
    training_age_userid['age'] = np.where(((training_age_userid['age'] > 0) & (training_age_userid['age'] < 25)), 1,
                                          training_age_userid['age'])
    training_age_userid['age'] = np.where(((training_age_userid['age'] > 24) & (training_age_userid['age'] < 35)), 2,
                                          training_age_userid['age'])
    training_age_userid['age'] = np.where(((training_age_userid['age'] > 34) & (training_age_userid['age'] < 50)), 3,
                                          training_age_userid['age'])
    training_age_userid['age'] = np.where(((training_age_userid['age'] > 49)), 4, training_age_userid['age'])
    training_data = pd.merge(left=training_age_userid, right=CompleteDataTrain, how='left', left_on='userid',
                             right_on='userid')
    training_data.rename(columns={'age_x': 'age', 'age_y': 'age_original'}, inplace=True)

    # preparing test data
    # test_files_list = [f for f in listdir(test_text_dir) if isfile(join(test_text_dir, f))]
    # test_data_frame = pd.DataFrame(get_text_and_userid(test_files_list, test_text_dir))
    # test_data = pd.merge(test_data_frame, profile_test_data, on='userid')
    return training_data, CompleteDataTest

def training_testing_age_NaiveBayes(CompleteDataTrain,CompleteDataTest):
        feature_name = ['transcript']
        ageX = CompleteDataTrain[feature_name]
        ageY = CompleteDataTrain.age
        multiNB = MultinomialNB()
        count_vectorizer = CountVectorizer()
        tf_idf_transformer = TfidfTransformer()
        accuracy = 0
        age_kfold = KFold(n_splits=10, shuffle=True)
        for training_index, test_index in age_kfold.split(ageX, ageY):
            ageX_training, ageX_test = ageX.loc[training_index,], ageX.loc[test_index,]
            ageY_training, ageY_test = ageY.loc[training_index,], ageY.loc[test_index,]
            age_training_count = count_vectorizer.fit_transform(ageX_training.transcript)
            age_training_tf_idf = tf_idf_transformer.fit_transform(age_training_count)
            multiNB.fit(age_training_tf_idf, ageY_training)
            age_test_count = count_vectorizer.transform(ageX_test.transcript)
            age_test_tfidf = tf_idf_transformer.transform(age_test_count)
            age_predicted = multiNB.predict(age_test_tfidf)
            accuracy += accuracy_score(ageY_test, age_predicted)

        print("Accuracy of Age Naive Bayes=", accuracy / 10)
        age_test_count = count_vectorizer.transform(CompleteDataTest.transcript)
        age_test_tf_idf = tf_idf_transformer.transform(age_test_count)
        predicted_age_values = multiNB.predict(age_test_tf_idf)
        # print("test_data=",test_data)
        CompleteDataTest['age'] = np.int_(predicted_age_values)
        # test_age_prediction_list = test_data['userid']
        predicted_age = predicted_age_values.astype(np.str)
        predicted_age[predicted_age == "1.0"] = "xx-24"
        predicted_age[predicted_age == "2.0"] = "25-34"
        predicted_age[predicted_age == "3.0"] = "35-49"
        predicted_age[predicted_age == "4.0"] = "50-xx"
        CompleteDataTest['age'] = predicted_age
        predicted_age_test = CompleteDataTest.loc[:, ['userid', 'age']]
        return CompleteDataTest


def training_testing_ML_Age(CompleteDataTrain,CompleteDataTest):
    feature_name = ['transcript']
    ageX = CompleteDataTrain[feature_name]
    ageY = CompleteDataTrain.age
    sgdModel = SGDClassifier()
    count_vectorizer = CountVectorizer()
    tf_idf_transformer = TfidfTransformer()
    accuracy = 0
    age_kfold = KFold(n_splits= 10, shuffle=True)
    for training_index, test_index in age_kfold.split(ageX, ageY):
        ageX_training, ageX_test = ageX.loc[training_index,], ageX.loc[test_index,]
        ageY_training, ageY_test = ageY.loc[training_index,], ageY.loc[test_index,]
        age_training_count = count_vectorizer.fit_transform(ageX_training.transcript)
        age_training_tf_idf = tf_idf_transformer.fit_transform(age_training_count)
        sgdModel.fit(age_training_tf_idf, ageY_training)
        age_test_count = count_vectorizer.transform(ageX_test.transcript)
        age_test_tfidf = tf_idf_transformer.transform(age_test_count)
        age_predicted = sgdModel.predict(age_test_tfidf)
        accuracy += accuracy_score(ageY_test, age_predicted)

    print("Accuracy of Age SGD=", accuracy / 10)
    age_test_count = count_vectorizer.transform(CompleteDataTest.transcript)
    age_test_tf_idf = tf_idf_transformer.transform(age_test_count)
    predicted_age_values = sgdModel.predict(age_test_tf_idf)
    # print("test_data=",test_data)
    CompleteDataTest['age'] = np.int_(predicted_age_values)
    # test_age_prediction_list = test_data['userid']
    predicted_age = predicted_age_values.astype(np.str)
    predicted_age[predicted_age == "1.0"] = "xx-24"
    predicted_age[predicted_age == "2.0"] = "25-34"
    predicted_age[predicted_age == "3.0"] = "35-49"
    predicted_age[predicted_age == "4.0"] = "50-xx"
    CompleteDataTest['age'] = predicted_age
    predicted_age_test = CompleteDataTest.loc[:, ['userid', 'age']]
    return CompleteDataTest




def write_data(output_dir, CompleteDataTest):
    print("--------------------------------------------------")

    print("writing data into xml files into output dir " + output_dir)

    extrovert = str(3.486857894736829)
    neurotic = str(2.7324242105263203)
    agreeble = str(3.5839042105263155)
    conscientious = str(3.445616842105264)
    open_nature = str(3.9086905263157825)

    for x in range( 0, len(CompleteDataTest)):
        id = CompleteDataTest.iloc[x]['userid']
        gender = (CompleteDataTest.iloc[x]['gender'])
        age_group = str(CompleteDataTest.iloc[x]['age'])

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




# def write_data(output_dir, CompleteDataTest):
#     print("--------------------------------------------------")
#
#     print("writing data into xml files into output dir " + output_dir)
#
#     for x in range( 0, len(CompleteDataTest)):
#         id = CompleteDataTest.iloc[x]['userid']
#         age_group = CompleteDataTest.iloc[x]('age_group')
#         gender = str(CompleteDataTest.iloc[x]['gender'])
#
#         # extrovert = str(averages.get('extrovert'))
#         # neurotic = str(averages.get('neurotic'))
#         # agreeble = str(averages.get('agreeable'))
#         # conscientious = str(averages.get('conscientious'))
#         # open_nature = str(averages.get('open'))
#
#         extrovert = str(3.486857894736829)
#         neurotic = str(2.7324242105263203)
#         agreeble = str(3.5839042105263155)
#         conscientious = str(3.445616842105264)
#         open_nature = str(3.9086905263157825)
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
#     print("writing data into xml files completed")
#     print("--------------------------------------------------")

parser = argparse.ArgumentParser()
parser.add_argument('-i', type=str, help='input directory with trailing /', )
parser.add_argument('-o', type=str, help='output directory with trailing /')

# args = parser.parse_args()
# input_directory = os.path.expanduser(args.i)
# output_directory = os.path.expanduser(args.o)

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

    print("----------------------- Gender Prediction -------------------------")
    CompleteDataTrain, CompleteDataTest = copy_training_testing_data(profile_train_data, profile_test_data, train_text_dir, test_text_dir)
    CompleteDataTest = training_testing_ML_gender(CompleteDataTrain, CompleteDataTest)
    CompleteDataTest = training_testing_gender_SGD(CompleteDataTrain, CompleteDataTest)


    # # write_data(output_directory, CompleteDataTest)

    print("----------------------- Age Prediction -------------------------")

    CompleteDataTrain, CompleteDataTest = copy_test_train_data_Age(CompleteDataTrain, CompleteDataTest)
    CompleteDataTest = training_testing_age_NaiveBayes(CompleteDataTrain, CompleteDataTest)
    CompleteDataTest = training_testing_ML_Age(CompleteDataTrain, CompleteDataTest)

    # completeDataresult = pd.merge(CompleteDataTestAge, CompleteDataTest, on='userid')

    # print(CompleteDataTest.head())

    write_data(output_dir, CompleteDataTest)


if __name__ == "__main__":
    #args = parse_arguments()
    main()