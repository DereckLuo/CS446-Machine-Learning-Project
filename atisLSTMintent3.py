# from sklearn import svm
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import Word2Vec
import tensorflow as tf
import numpy as np
import tflearn
import matplotlib.pyplot as plt

#import tensorflow.contrib.rnn.BasicRNNCell as BasicRNNCell
def wordToNum(word):
    word = word.upper()
    L = [ord(c) for c in word]
    ret = L[0]
    for i in range(1,len(L)):
        ret *= 100
        ret += L[i]
    #print(ret)
    return ret

# def sentenceToArray(line):
#     word = "".join((char if char.isalpha() else " ") for char in line).split()
#     # print(word)
#     value = []
#     label = word[len(word)-1]
#     for i in range(0,15):
#         if(i < len(word)):
#             value.append(wordToNum(word[i]))
#         else:
#             value.append(0)
#
#     return value, label
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

# def featureExtract(line, dic):
#     word = "".join((char if char.isalpha() else " ") for char in line).split()
#     label = word[len(word)-1]
#     sentence = []
#     for w in word:
#         if w == "BOS":
#             continue
#         elif w == "EOS":
#             break
#         else:
#             sentence.append(w)
#     # line_feature = [0]*len(dic)
#     line_feature = [0]* 50
#     i=0
#     for w in sentence:
#         if w in dic:
#     #        line_feature[dic[w]] = 1
#             line_feature[i]=dic[w]
#             i+=1
#             if(i>=50):
#                 break
#     # for i in range(len(sentence)-1):
#     #     twoPair = sentence[i] + sentence[i+1]
#     #     if twoPair in dic:
#     #         line_feature[dic[twoPair]] = 1

    # return line_feature, label

def read_lines(line, m):
    word = "".join((char if char.isalpha() else " ") for char in line).split()
    label = word[len(word) - 1]
    sentence = [np.zeros(20)]*50
    i=0
    for w in word:
        if w == "BOS":
            continue
        elif w == "EOS":
            break
        elif i>=50:
            break
        else:
            w1= m.wv[w]
            sentence[i]=w1
    s= np.stack(sentence,axis=0)
    # print(s.shape)
    return s, label

# def generateDictionary(fname):
#     word_dic = {}
#     label_dic = {}
#     word_id = 1
#     with open(fname) as f:
#         for line in f:
#             sentence, label = extractSentence(line)
#             if label not in label_dic:
#                 label_dic[label] = 1
#             # print(sentence)
#             # single word
#             for w in sentence:
#                 if w not in word_dic:
#                     word_dic[w] = word_id
#                     word_id += 1
#             # # 2-gram feature
#             # for i in range(len(sentence)-1):
#             #     twoPair = sentence[i] + sentence[i+1]
#             #     if twoPair not in dic:
#             #         dic[twoPair] = word_id
#             #         word_id += 1
#
#     return word_dic, label_dic

def generateLabelDict(fname):
    label_dict= {}
    labelid= 0
    with open(fname) as f:
        for line in f:
            sentence, label = extractSentence(line)
            if label not in label_dict:
                label_dict[label]= labelid
                labelid +=1
    return label_dict

def generateLabelArray(label_dict, label):
    labelid= label_dict[label]
    l1= len(label_dict)
    label_array= np.zeros(l1)
    label_array[labelid]=1
    return label_array


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


fname = 'atis_intent_data/atis_intent_train.txt'
# word_dic, label_dic = generateDictionary(fname)

training_num = 3000
testing_num = 1000
word_feature_size=100
x_train = []
y_train = []
x_test = []
y_test = []
sentences = []
label_dict= generateLabelDict(fname)
# print(len(label_dict))


with open(fname) as f:
    for line in f:
        sentence, label = extractSentence(line)
        sentences.append(sentence)
print(len(sentences),"data length")
m = Word2Vec(sentences, size=word_feature_size, window=5, min_count=1, workers=4)

with open(fname) as f:
    for i in range(testing_num):
        line = f.readline()
        # line_feature, label = sentenceToArray(line)
        line_feature, label = read_lines(line,m)
        label_array= generateLabelArray(label_dict,label)
        x_test.append(line_feature)
        y_test.append(label_array)

    for i in range(training_num):
        line = f.readline()
        # line_feature, label = sentenceToArray(line)
        line_feature, label = read_lines(line,m)
        label_array= generateLabelArray(label_dict,label)
        x_train.append(line_feature)
        y_train.append(label_array)


#print("test_label")
#print(x_test)
#print("y dict")
#print(label_dict)

word1= m.wv["at"]
print(word1,"word_1_at ")
print(len(word1))
print(len(x_train[0]))
print(len(x_test[0]))
x_train= np.array(x_train)
x_test = np.array(x_test)
y_train= np.array(y_train)
y_test= np.array(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
sess= tf.InteractiveSession()
n_length=50
n_step= 20
n_input = 50
n_hidden= 100
n_batch=1
net= tflearn.input_data(shape=[None,n_input,word_feature_size])
# net= net[0:19]
net= tf.reshape(net,[-1,n_input,word_feature_size])
# net= tf.unstack(net,[-1,n_step,n_input])

net= tflearn.layers.recurrent.lstm(net, n_hidden)
# lastout= net_out[n_step-1]
net= tflearn.layers.core.fully_connected(net,17)
#net= tflearn.activations.softmax (net)
momentum= tflearn.optimizers.Momentum(learning_rate=0.1, momentum=0.9)
net= tflearn.layers.estimator.regression(net,optimizer=momentum)
model=tflearn.DNN(net)
model.fit(x_train,y_train,n_epoch=10,show_metric=True,batch_size=1,snapshot_epoch=True,run_id="atis_lstm")
#
#
# a=model.evaluate(x_test,y_test,batch_size=1)
a= model.predict(x_test)
b= model.predict(x_train)
# print(len(a))
# print(len(a[0]))
s1=[]
acc_num=0.0
train_acc=0.0
# print("y dict")
# print(label_dict)

test_label_dict= {}
for l in label_dict:
    test_label_dict[l]=0

test_label_id=0
for j in range(testing_num):
    if(np.argmax(y_test[j],-1)==np.argmax(a[j],-1) ):
        acc_num+=1

    ti= np.argmax(a[j],-1)
    #print(ti)
    for w in label_dict:
        if(ti== label_dict[w]):
            test_label_dict[w]+=1
            break


for i in range(training_num):
    if (np.argmax(y_train[i], -1) == np.argmax(b[i], -1)):
        train_acc+=1
accuracy= 1.0 * acc_num / testing_num
train_acc= 1.0* train_acc / training_num
print(label_dict,"y_dict")
print(test_label_dict,"test_dict")
print(acc_num)
print(train_acc,"train_acc")
print(accuracy,"accuracy")
