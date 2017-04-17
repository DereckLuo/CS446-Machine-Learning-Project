from sklearn import svm


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
                print(label)
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

    print("prediction labels are : ~~~~~~~~\n")
    print(predicts)
    return 1 - err / len(y_test)

def test():

    fname = 'atis_intent_data/atis_intent_train.txt'
    # word_dic, label_dic = generateDictionary(fname)
    training_num = 4000
    testing_num = 500

    # print(len(word_dic))
    # print(len(label_dic))
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

    print("Testing labels are : ~~~~~ \n")
    print(labels)

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)

    err = computeError(pred, y_test)

    print(err)


if __name__ == "__main__":

    # test()
    fname = 'atis_intent_data/atis_intent_train.txt'
    word_dic, label_dic = generateDictionary(fname)
    training_num = 4000
    testing_num = 500

    print(label_dic)
    # print(len(word_dic))
    # print(len(label_dic))
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

    clf = svm.SVC(decision_function_shape='ovo')
    clf.fit(x_train,y_train)
    pred = clf.predict(x_test)

    err = computeError(pred, y_test)

    print(err)

