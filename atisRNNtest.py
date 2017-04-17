from sklearn import svm
import tensorflow as tf
import numpy as np
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

def generateDictionary(fname):
    word_dic = {}
    label_dic = {}
    word_id = 1
    with open(fname) as f:
        for line in f:
            sentence, label = extractSentence(line)
            if label not in label_dic:
                label_dic[label] = 1
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
training_num = 3000
testing_num = 1000
x_train = []
y_train = []
x_test = []
y_test = []
label_dict= generateLabelDict(fname)
print(len(label_dict))
with open(fname) as f:
    for i in range(training_num):
        line = f.readline()
        line_feature, label = sentenceToArray(line)
        label_array= generateLabelArray(label_dict,label)
        x_train.append(line_feature)
        y_train.append(label_array)
    for i in range(testing_num):
        line = f.readline()
        line_feature, label = sentenceToArray(line)
        label_array= generateLabelArray(label_dict,label)
        x_test.append(line_feature)
        y_test.append(label_array)

# print(y_train)

# with open('atis_intent_data/atis_intent_train.txt') as f:
#     # for line in f:
#     for i in range(1000):
#         line = f.readline()
#         linearr, label = sentenceToArray(line)
#         x_train.append(linearr)
#         y_train.append(label)
#     for i in range(100):
#         line = f.readline()
#         linearr, label = sentenceToArray(line)
#         x_test.append(linearr)
#         y_test.append(label)

#clf = svm.SVC(decision_function_shape= 'ovo')
#clf.fit(x_train,y_train)
#pred = clf.predict(x_test)

#err = 0.0
#for i in range(len(y_test)):
#    if pred[i] != y_test[i]:
#        err += 1;

#print(err)
#print(1- err/len(y_test))



sess= tf.InteractiveSession()
n_step= 15
n_input = 1
n_hidden= 40

x1 = tf.placeholder(tf.float32, [15])
y1=  tf.placeholder(tf.float32,[17])
x_img= tf.reshape(x1, [-1,15,1])
# print(x_img.get_shape().as_list())
x= tf.unstack(x_img,n_step,1)
# print(len(x))
cell1= tf.contrib.rnn.BasicRNNCell(n_hidden)
outputs, state = tf.contrib.rnn.static_rnn(cell1, x, dtype=tf.float32)
lastout= outputs[n_step-1]


w1= weight_variable([n_hidden,17])
b1= bias_variable([17])
y= tf.nn.softmax(tf.matmul(lastout,w1)+ b1)
print(y.get_shape().as_list())
z= tf.argmax(y,-1)
print(z.get_shape().as_list())
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
correct_prediction = tf.equal(tf.argmax(y,-1), tf.argmax(y1,-1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()
acc1=[]
acc2=[]
words=[]
for i in range(3000):
    batch_x= x_train[i][0:15]
    batch_y = y_train[i]
    # print(batch_x)
    # print(batch_y)
    # sess.run(lastout,feed_dict={x1:batch_x,y1:batch_y})
    # print(lastout.get_shape().as_list())
    sess.run(train_step,feed_dict={x1:batch_x,y1:batch_y})
    output1= accuracy.eval({x1:batch_x,y1:batch_y})
    output2= z.eval({x1:batch_x,y1:batch_y})
    acc1.append(output1)
    words.append(output2[0])
    if i%100==0 :
        acc2.append(np.mean(acc1))
        acc1=[]



  #   out=lastout.eval({x1:batch_x})
  #   print(out)
s1=[]
s2=[]
for j in range(1000):
    batch_x= x_train[j][0:15]
    batch_y= y_train[j]
    test_x= x_test[j][0:15]
    test_y= y_test[j]
    t1= accuracy.eval({x1:batch_x, y1:batch_y})
    s1.append(t1)
    t2= accuracy.eval({x1:test_x,y1:test_y})
    s2.append(t2)

print("train_acc",np.mean(s1))
print("test_acc",np.mean(s2))
predicts={}
# print(words[0:10])
for i in range(3000):
    if words[i] not in predicts:
        predicts[words[i]]=1
    else:
        predicts[words[i]]+=1
print(label_dict)
print(predicts)
fig1= plt.figure('atisRNN')
ax1 = fig1.add_subplot(111)
xaxis= np.linspace(0,3000,num=30)
plt.plot(xaxis, acc2, 'r', label="train")
# plt.plot(xaxis, testerr, 'b', label="test")
plt.legend(bbox_to_anchor=(0.8, 0.8), loc=2, borderaxespad=0.)
plt.show()

