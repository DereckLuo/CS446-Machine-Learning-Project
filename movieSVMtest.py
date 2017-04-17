from sklearn import svm
import pytreebank


# sentence = open('stanfordSentimentTreebank/datasetSentences.txt')
# label = open('stanfordSentimentTreebank/sentiment_labels.txt')

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
    word = "".join((char if char.isalpha() else " ") for char in line).split()
    # print(word)
    value = []

    for i in range(0,15):

        if(i < len(word)):
            value.append(wordToNum(word[i]))
        else:
            value.append(0)
    #print(value)
    return value

dataset = pytreebank.import_tree_corpus("stanfordSentimentTreebank/STree.txt")

print(dataset)
#
# x = []
# test = []
# with open('stanfordSentimentTreebank/datasetSentences.txt') as f:
#     # for line in f:
#     line = f.readline()
#     for i in range(50):
#         line = f.readline()
#         # print(line)
#         linearr = sentenceToArray(line)
#         # print(linearr)
#         x.append(linearr)
#     for i in range(1,11):
#         line = f.readline()
#         linearr = sentenceToArray(line)
#         test.append(linearr)
#
# print((x))
# print(len(test))
#
# y = []
# with open('stanfordSentimentTreebank/label.txt') as f:
#     # for line in f:
#     #   y.append(line)
#     for i in range(50):
#         line = f.readline()
#         y.append(int(line))
#
# print((y))
#
# test_label = []
# with open('stanfordSentimentTreebank/label_test.txt') as f:
#     for i in range(10):
#         line = f.readline()
#         test_label.append(int(line))
#
#
# clf = svm.SVC(decision_function_shape= 'ovo')
# clf.fit(x,y)
# pred = clf.predict(test)
# print(pred)
#
# err = 0.0
# for i in range(10):
#     if pred[i] != test_label[i]:
#         err += 1
#
# print(err)