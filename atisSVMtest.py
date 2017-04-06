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
    word = "".join((char if char.isalpha() else " ") for char in line).split()
    # print(word)
    value = []
    label = word[len(word)-1]
    for i in range(0,15):
        if(i < len(word)):
            value.append(wordToNum(word[i]))
        else:
            value.append(0)

    return value, label

x_train = []
y_train = []
x_test = []
y_test = []
with open('atis_intent_data/atis_intent_train.txt') as f:
    # for line in f:
    for i in range(1000):
        line = f.readline()
        linearr, label = sentenceToArray(line)
        x_train.append(linearr)
        y_train.append(label)
    for i in range(100):
        line = f.readline()
        linearr, label = sentenceToArray(line)
        x_test.append(linearr)
        y_test.append(label)

clf = svm.SVC(decision_function_shape= 'ovo')
clf.fit(x_train,y_train)
pred = clf.predict(x_test)

err = 0.0
for i in range(len(y_test)):
    if pred[i] != y_test[i]:
        err += 1;

print(err)
print(1- err/len(y_test))

