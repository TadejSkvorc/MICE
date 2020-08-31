import tensorflow as tf
import numpy as np
import treetaggerwrapper
from tensorflow.keras.layers import Concatenate, concatenate, Bidirectional, LSTM, Dense, Activation, TimeDistributed, Masking, GRU, Input, Embedding
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
import tensorflow_datasets as tfds
from ast import literal_eval
from sklearn.model_selection import train_test_split, ShuffleSplit
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC


VALID_CLASSES = ['NE', 'DA', '*', 'NEJASEN_ZGLED']
CLS_TO_INT_DICT = {'NE': 3, 'DA': 2, '*':1, 'NEJASEN_ZGLED':4, 'INVALID_CLASS': 5}
IGNORE_POS = ['$.', '$,', '$:', '$;']
INPUT_FILENAMES = ''


def read_data(input_filenames):
    i = 0
    word_classes = []            
    for filename in input_filenames
        with open(filename, 'r', encoding='utf-8') as classf:
            for line in classf:
                parts = line.split(' ')
                if len(parts) != 2:
                    """
                    # Skipping sentences should be unnecessary when no using the vector file
                    if i not in skipped_sents:
                        word_classes.append(curr_sentence)
                    else:
                        print('skipped', i)
                    """
                    i += 1
                    if i > 200:
                        return word_classes
                    continue
                word = parts[0]
                if word[-1] in ['.', ',', ':', ';', '?', '!'] and len(word) > 1:
                    word = word[:-1]
                if word in ['.', ',', ':', ';', '?', '!']:
                    continue
                cls = parts[1]
                if cls[:-1] in VALID_CLASSES:
                    word_classes.append([word, cls])
                else:
                    word_classes.append([word, 'INVALID_CLASS\n']) 
    return word_classes
            
            
def words_to_dataset(word_classes, window_size):
    # transform single words into groups of +/- window_size words, where the class corresponds to the class of the central word
    X = []
    Y = []
    for i in range(len(word_classes)):
        if i < window_size:
            continue
        else:
            X.append(" ".join([x[0] for x in word_classes[i-window_size:i+window_size+1]]))
            Y.append(word_classes[i][1])
    return X, Y

    
def pos_tag_data(sentences):
    tagger = treetaggerwrapper.TreeTagger(TAGDIR='E:/kauc-stran-berljivost-host-latest/tree-tagger/', 
                                          TAGPARFILE='E:/kauc-stran-berljivost-host-latest/tree-tagger/slovenian.par')
    pos_tagged_sentences = []
    curr_sentence = []
    for i, sent in enumerate(sentences):
        words = [w[0] for w in sent]
        classes = [w[1] for w in sent]
        tagged_sentence = tagger.tag_text(words, tagonly=True)
        tagged_sentence_tags = treetaggerwrapper.make_tags(tagged_sentence)
        #print(tagged_sentence_tags)
        tags_only = []
        lemmas_only = []
        for t in tagged_sentence_tags:
            #print(t, type(t) is treetaggerwrapper.NotTag, type(t))
            if type(t) is treetaggerwrapper.NotTag:
                tags_only.append('x')
                lemmas_only.append('x')
            else:
                #print(t, type(t) is treetaggerwrapper.NotTag, type(t))
                if t.pos in IGNORE_POS:
                    continue
                else:
                    tags_only.append(t.pos)
                    lemmas_only.append(t.lemma)
        
        #print(i, 'lens', len(words), len(tags_only), len(lemmas_only), len(classes))
        if i % 500 == 0:
            print(i)
        if not (len(words) == len(tags_only) == len(lemmas_only) == len(classes)):
            print('Error: lemmas, tags, words, and classes lengths do not match')
            print('words', words)
            print('tags', tags_only)
            print('lemmas', lemmas_only)
            print('classes', classes)
            for i in range(len(words)):
                print(words[i], tags_only[i], lemmas_only[i], classes[i])
            raise TypeError
        for w, t, l, c in zip(words, tags_only, lemmas_only, classes):
            curr_sentence.append([w, t, l, c])
        pos_tagged_sentences.append(curr_sentence)
        curr_sentence = []
    return pos_tagged_sentences
                


word_classes = read_data('./classes_elmo.txt')
word_classes_test = read_data('./classes_elmo_2_dataset.txt')
X_train, Y_train = words_to_dataset(word_classes, 2)

for i in range(500):
    print(X_train[i], Y_train[i])
X_test, Y_test = words_to_dataset(word_classes_test, 2)
print(X_test[0], Y_test[0])
print(X_test[1], Y_test[2])
print(X_test[2], Y_test[1])

# Create the vocab for words
text = []
for sent in X_train:
    words = [w[0] for w in sent.split(' ')]
    text += words
for sent in X_test:
    words = [w[0] for w in sent.split(' ')]
    text += words

text = list(set(text))

# Vectorize dataset
print('vectorizing')

vectorizer = CountVectorizer()
vectorizer.fit(X_train + X_test)
X_train = vectorizer.transform(X_train).toarray()
X_test = vectorizer.transform(X_test).toarray()
#print('vectorized X_train[0]', X_train.toarray())
#exit()

nb_model = GaussianNB()
nb_model.fit(X_train, Y_train)
preds = nb_model.predict(X_test)
print(Counter(preds))
print('nb acc', sum(Y_test == preds) / len(Y_test))
svm_model = LinearSVC()
svm_model.fit(X_train, Y_train)
preds = svm_model.predict(X_test)
print(Counter(preds))
print('svm acc', sum(Y_test == preds) / len(Y_test))


"""
# Perform splitting. This is only relevant if there isn't a second test file provided
splitter = ShuffleSplit(n_splits=1, train_size=0.7)
train_i, test_i = list(splitter.split(X_words))[0]
print(train_i)
print(test_i)

X_train_words_single_file = X_words[train_i]
X_train_tags_single_file = X_tags[train_i]
X_train_lemmas_single_file = X_lemmas[train_i]
Y_train_single_file = padded_Y[train_i]


X_test_words_single_file = X_words[test_i]
X_test_tags_single_file = X_tags[test_i]
X_test_lemmas_single_file = X_lemmas[test_i]
Y_test_single_file = padded_Y[test_i]


#print(len(X_train_words), len(X_train_tags), len(X_train_lemmas), len(Y_train)) 
#print(len(X_test_words), len(X_test_tags), len(X_test_lemmas), len(Y_test))

#print(Y_train[-1]) 
#print('Y_train[-1].shape', Y_train[-1].shape)
#print('Y_train.shape', Y_train.shape)


model = build_model(vs_words, vs_lemmas, vs_tags)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#One file version              
#history = model.fit([X_train_words_single_file, X_train_lemmas_single_file, X_train_tags_single_file],
#                    Y_train_single_file,
#                    batch_size=32,
#                    epochs=5,
#                    validation_split=0.2)
#print(model.summary())

#Two file version
history = model.fit([X_words, X_lemmas, X_tags],
                    padded_Y,
                    batch_size=32,
                    epochs=5,
                    validation_split=0.2)

# One file version
#test_scores = model.evaluate([X_test_words_single_file, X_test_lemmas_single_file, X_test_tags_single_file],
#                             Y_test_single_file,
#                             verbose=2)

print('testing')
print('X_words', X_words.shape, X_words_test.shape)
print('X_lemmas', X_lemmas.shape, X_lemmas_test.shape)
print('X_tags', X_tags.shape, X_lemmas_test.shape)
print('padded_Y', padded_Y.shape, padded_Y_test.shape)
# Two file version
test_scores = model.evaluate([X_words_test, X_lemmas_test, X_tags_test],
                             padded_Y_test,
                             verbose=2)

print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

#One file version
#preds = model.predict([X_test_words_single_file, X_test_lemmas_single_file, X_test_tags_single_file], verbose=1)

# Two file version
preds = model.predict([X_words_test, X_lemmas_test, X_tags_test], verbose=1)

print(preds)
print(preds[0])
num_correct = 0
num_incorrect = 0
with open('preds_out_temp_mumuls.txt', 'w') as tempoutf:
    # Two file version
    for sent_preds, sent_Y in zip(preds, padded_Y_test):
    # Single file version
    #for sent_preds, sent_Y in zip(preds, Y_test_single_file):
        for word_preds, word_Y in zip(sent_preds, sent_Y):
            print(np.argmax(word_preds), np.argmax(word_Y), file=tempoutf)
            if np.argmax(word_Y) == 2 or np.argmax(word_Y) == 3:
                print('found', np.argmax(word_Y), np.argmax(word_preds))
                if np.argmax(word_Y) == np.argmax(word_preds):
                    num_correct += 1
                else:
                    num_incorrect += 1
print('custom accuracy is', num_correct/(num_correct+num_incorrect))
print(len(preds), len(Y_test_single_file))



"""