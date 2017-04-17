



if __name__ == "__main__":
    read_file = 'atis_intent_data/atis_intent_test.txt'
    write_file = 'atis_intent_data/atis_intent_test2.txt'

    f1 = open(read_file, 'r')
    f2 = open(write_file, 'w')

    sentence = ""
    for line in f1:
        words = "".join((char if char.isalpha() else " ") for char in line).split()
        if len(words) != 0:
            sentence += (words[0] + " ")
            if words[0] == "EOS":
                sentence += (words[len(words)-1] + "\n")
                f2.write(sentence)
                sentence = ""
    f1.close()
    f2.close()