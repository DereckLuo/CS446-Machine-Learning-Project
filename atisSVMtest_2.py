from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer


def wordToNum(word):
    word = word.upper()
    L = [ord(c) for c in word]
    ret = L[0]
    for i in range(1,len(L)):
        ret *= 100
        ret += L[i]
    #print(ret)
    return ret

def sentenceToArray(line):
    value = []
    sentence = []
    word = "".join((char if char.isalpha() else " ") for char in line).split()
    # word.sort()
    label = word[len(word)-1]
    idx = 1
    while(word[idx] != "EOS"):
        sentence.append(word[idx])
        idx += 1
    # print(sentence)
    # sentence.sort()
    for i in range(0,15):
        if(i < len(sentence)):
            value.append(wordToNum(sentence[i]))
        else:
            value.append(0)
    return value, label

def extractSentence(line):
    word = "".join((char if char.isalpha() else " ") for char in line).split()
    label = word[len(word)-1]
    sentence = []
    for w in word:
        if w == "BOS":
            continue
        elif w =="EOS":
            break
        else:
            sentence.append(w)
    return sentence, label


def generateDictionary(fname):
    word_dic = {}
    label_dic = {}
    word_id = 1
    with open(fname) as f:
        
        for line in f:
            sentence, label = extractSentence(line)
            if label not in label_dic:
                label_dic[label] = 1
            else:
                label_dic[label] += 1
            # print(sentence)
            # single word
            for w in sentence:
                if w not in word_dic:
                    word_dic[w] = word_id
                    word_id += 1
            

            # # 2-gram feature
            # for i in range(len(sentence)-1):
            #     twoPair = sentence[i] + sentence[i+1]
            #     if twoPair not in dic:
            #         dic[twoPair] = word_id
            #         word_id += 1
    
    return word_dic, label_dic

def featureExtract(line, dic):
    word = "".join((char if char.isalpha() else " ") for char in line).split()
    label = word[len(word)-1]
    sentence = []
    for w in word:
        if w == "BOS":
            continue
        elif w == "EOS":
            break
        else:
            sentence.append(w)
    line_feature = [0]*len(dic)
    for w in sentence:
        if w in dic:
            line_feature[dic[w]] = 1
    # for i in range(len(sentence)-1):
    #     twoPair = sentence[i] + sentence[i+1]
    #     if twoPair in dic:
    #         line_feature[dic[twoPair]] = 1

    return line_feature, label

def computeError(pred, y_test):
    err = 0.0
    predicts = {}
    for i in range(len(y_test)):
        if pred[i] != y_test[i]:
            err += 1

        if pred[i] not in predicts:
            predicts[pred[i]] = 1
        else:
            predicts[pred[i]] += 1

    print "prediction labels are: " + str(predicts)
    return 1 - err / len(y_test)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def process_labels(labels):
    converted_labels = []
    for label in labels:
        if label != 'flight':
            converted_labels.append('others')
        else:
            converted_labels.append('flight')

    return converted_labels

def get_pure_sentences(filename):
    sentence_array = []
    with open(filename) as f:
        for line in f:
            sentence = []
            word_array = "".join((char if char.isalpha() else " ") for char in line).split()
            #label = word[len(word)-1]
            for word in word_array:
                if word == "BOS":
                    continue
                elif word =="EOS":
                    break
                else:
                    sentence.append(word)
            sentence_array.append(' '.join(sentence))

    return sentence_array


def generate_feature_vector(filename):
    sentence_array = get_pure_sentences(filename)
    vectorizer = CountVectorizer(min_df=1)
    #bigram_vectorizer = CountVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
    features = vectorizer.fit_transform(sentence_array)

    fea_vec = features.toarray() 
    #fea_vec = bigram_vectorizer.fit_transform(sentence_array).toarray()
    return fea_vec

def get_labels(filename):
    labels = []
    with open(filename) as f:
        for line in f:
            sentence = []
            word_array = "".join((char if char.isalpha() else " ") for char in line).split()
            labels.append(word_array[len(word_array)-1])
            
    return labels

def generate_second_input(fea_vec, labels, reference=None):
    n = len(labels)
    new_dataset = []
    new_labels = []
    for i in xrange(n):
        if labels[i] != 'flight':
            new_dataset.append(fea_vec[i])
            new_labels.append(reference[i])

    return new_dataset, new_labels


def generate_second_test(fea_vec, labels, reference=None):
    n = len(labels)
    new_dataset = []
    new_labels = []
    for i in xrange(n):
        if labels[i] != 'flight' and reference[i] != 'flight':
            new_dataset.append(fea_vec[i])
            new_labels.append(reference[i])

    return new_dataset, new_labels



def test():
    fname = 'atis_intent_train.txt'
    word_dic, label_dic = generateDictionary(fname)
    training_num = 4000
    testing_num = 500

    print(len(word_dic))
    print(len(label_dic))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    labels = {}
    predict = {}

    with open(fname) as f:
        for i in range(training_num):
            line = f.readline()
            line_feature, label = sentenceToArray(line)
            x_train.append(line_feature)
            y_train.append(label)
        for i in range(testing_num):
            line = f.readline()
            line_feature, label = sentenceToArray(line)
            x_test.append(line_feature)
            y_test.append(label)
            if label not in labels:
                labels[label] = 1
            else:
                labels[label] += 1
    y_train = process_labels(y_train)
    y_test = process_labels(y_test)

    print("Testing labels are : ~~~~~ \n")
    print(labels)

    clf = svm.SVC(kernel='rbf', decision_function_shape='ovr')
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)

    err = computeError(pred, y_test)

    print(err)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    #plot_learning_curve(clf, "Learning Curve", x_train, y_train, cv= cv)
    #plt.show()



'''
    y_train_dict = {}
    for i in y_train:
        if i not in y_train_dict:
            y_train_dict[i] = 1
        else:
            y_train_dict[i] += 1

    print "The dictionary of train labels: \n" 
    print y_train_dict

    y_test_dict = {}
    for i in y_test:
        if i not in y_test_dict:
            y_test_dict[i] = 1
        else:
            y_test_dict[i] += 1

    print "The dictionary of test labels: \n" 
    print y_test_dict
'''


if __name__ == "__main__":

    filename = 'atis_intent_train.txt'
    fea_vec = generate_feature_vector(filename)
    original_labels = get_labels(filename)
    labels = process_labels(original_labels)
    print fea_vec[0]
    # split the fea_vec into train set and test set
    x_train = fea_vec[:4000]
    y_train = labels[:4000]

    x_test = fea_vec[4000:]
    y_test = labels[4000:]

    clf = svm.SVC(kernel='rbf')
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)
    # 1st classification: 1->flight 0->others
    acc1 = computeError(pred, y_test)

    # extract others for train
    x2_train, y2_train = generate_second_input(x_train, y_train, original_labels)
    # generate second classifier on the remaining 'others', used for test
    x2_test, y2_test = generate_second_test(x_test, pred, original_labels)

    # 2nd labyer classifier
    clf2 = svm.SVC(kernel='rbf', decision_function_shape = 'ovr')
    clf2.fit(x2_train, y2_train)

    pred2 = clf2.predict(x2_test)

    # 2nd classification: 1->flight 0->others
    acc2 = computeError(pred2, y2_test)
    #acc = clf.score(x_test, y_test)
    print "The accuracy of 1st layer: " + str(acc1) + "\n"
    print "The accuracy of 2nd layer: " + str(acc2) + "\n"
  
    


    

    #test()
    '''
    fname = 'atis_intent_train.txt'
    word_dic, label_dic = generateDictionary(fname)
    training_num = 4000
    testing_num = 500

    print(len(word_dic))
    print(len(label_dic))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    labels = {}
    predict = {}

   
    with open(fname) as f:
        for i in range(training_num):
            line = f.readline()
            line_feature, label = featureExtract(line, word_dic)
            x_train.append(line_feature)
            y_train.append(label)
        for i in range(testing_num):
            line = f.readline()
            line_feature, label = featureExtract(line, word_dic)
            x_test.append(line_feature)
            y_test.append(label)
            if label not in labels:
                labels[label] = 1
            else:
                labels[label] += 1

    print("Testing labels are : ~~~~~ \n")
    print(labels)

    clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)

    err = computeError(pred, y_test)

    print(err)
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    plot_learning_curve(clf, "Learning Curve", x_train, y_train, ylim = (0.6, 0.9), cv= cv, n_jobs = 4)
    plt.show()
    '''


