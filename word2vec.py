import gensim
import numpy as np 


def get_pure_sentences(filename):
    sentences = []
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
            sentences.append(sentence)

    return sentences


def convert2vec(filename):
    sentences = get_pure_sentences(filename)
    model = gensim.models.Word2Vec(sentences, size=100, min_count=3, hs=1, negative=0)
    #fname = 'vector_model'
    #model.save(fname)
    # for each word, use model.wv[word] to get the vector for this word
    print model.wv['flight']


def main():
    filename = 'atis_intent_train.txt'
    convert2vec(filename)

if __name__ == '__main__':
    main()
