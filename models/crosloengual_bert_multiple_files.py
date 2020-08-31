import tensorflow as tf
import tensorflow as tf
import numpy as np
import sys
import torch
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, TimeDistributed, Masking, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from ast import literal_eval
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
from random import shuffle
from transformers import AutoTokenizer, AutoModel



# Modified tensorflow bilstm for sentence-level prediction

#VECTOR_DIM = 1024
VECTOR_DIM = 768 # bert
MAX_SEQUENCE_LEN = 50
IN_FILENAMES_TRAIN = ['../test_datasets/classes_elmo_1.txt',
                      '../test_datasets/classes_elmo_2.txt']
IN_FILENAMES_TEST = ['../test_datasets/classes_elmo_3.txt']
#IN_FILENAME = '../vector_datasets/vectors_with_classes_elmo_slo_top_full.txt'
#IN_FILENAME = '../vector_datasets/vectors_with_classes_elmo_slo_200k.txt'
#IN_FILENAME_TEST = '../vector_datasets/vectors_with_classes_elmo_slo_avg_2_50k.txt'
#IN_FILENAME_TEST = './vectors_with_classes_elmo_parseme_slo_top_full.txt'
IN_FILENAME_TEST = None
NUM_CLASSES = 5

    
def get_xy_per_expression(filenames, tokenizer, model):
    data_by_expressions = {}
    sent_wide_Y = []
    sents_X = []
    sents_Y = []
    curr_sent_X = []
    curr_sent_Y = []
    expressions = []
    curr_sent_words = []
    print('starting')
    CLS_TO_INT_DICT = {'NE': 3, 'DA': 2, '*':1, 'NEJASEN_ZGLED':4}
    classes = []
    words = []
    X = []
    Y = []
    for filename in filenames:
        print('reading file', filename)
        with open(filename, 'r', encoding='utf-8') as f:
            debug_sent = []
            for i, line in enumerate(f):
                if i % 500 == 0:
                    print(i)
                #if i >= 1500:
                #    break
                parts = line.split('\t')
                word = parts[0]
                #print(len(parts))
                if len(word) == 0:
                    continue
                if len(parts) != 3:
                    continue

                #print('len of parts', len(parts))
                word = parts[0]
                cls = parts[1]
                expression = parts[2]
                debug_sent.append((word, cls, expression))
                #print(word, cls)
                classes.append(cls)
                words.append(word)
                #print(exp, vector[:10], cls, expression)
                if not (cls == 'DA' or cls == 'NE' or cls == '*'):
                    continue
                curr_sent_words.append(word)
                #print(len(literal_eval(vector)))
                curr_sent_Y.append(CLS_TO_INT_DICT[cls]) 
                if word[-1] == '.':
                    #print('curr sent words', curr_sent_words)
                    str_sentence = ' '.join([x for x in curr_sent_words])
                    basic_tokens = [x for x in curr_sent_words]
                    tokenized_text = tokenizer.tokenize(str_sentence)
                    tokenized_text = [x for x in tokenized_text if x not in [',',':',';','!','?', '.', '"', "'", '/', '\\']]
                    if len(basic_tokens) == len(tokenized_text):
                        print('tokenizer didn\'t do anything')
                        print(basic_tokens)
                        print(tokenized_text)
                    if len(tokenized_text) > 510:
                        curr_sent = []
                        curr_sent_words = []
                        curr_sent_X = []
                        curr_sent_Y = []
                        debug_sent = []
                        continue
                    #print(expression)
                    expanded_classes = []
                    current_class_index = -1
                    #print(basic_tokens)
                    for w in tokenized_text:
                        #print(w[0:2])
                        if w[0:2] == '##':
                            expanded_classes.append(curr_sent_Y[current_class_index])
                            #print(w, curr_sent_Y[current_class_index], current_class_index, len(curr_sent_Y))
                        else:
                            current_class_index += 1
                            expanded_classes.append(curr_sent_Y[current_class_index])
                            #print(w, curr_sent_Y[current_class_index], current_class_index, len(curr_sent_Y))
                            
                    #for x, y in zip(tokenized_text, expanded_classes):
                    #    print(x,y)
                    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
                    tokens_tensor = torch.tensor([indexed_tokens])
                    with torch.no_grad():
                        outputs = model(tokens_tensor)
                        predictions = outputs[0]
                    vectors = predictions.numpy()[0]
                    if expression not in data_by_expressions.keys():
                        #print(expression)
                        data_by_expressions[expression] = [(vectors, np.array(expanded_classes))]
                    else:
                        #print(expression)
                        data_by_expressions[expression].append((vectors, np.array(expanded_classes)))
                    sent_wide_cls = None
                    if CLS_TO_INT_DICT['DA'] in curr_sent_Y:
                        sent_wide_cls = CLS_TO_INT_DICT['DA']
                    elif CLS_TO_INT_DICT['NE'] in curr_sent_Y:
                        sent_wide_cls = CLS_TO_INT_DICT['NE']
                    else:
                        sent_wide_cls = CLS_TO_INT_DICT['NEJASEN_ZGLED']
                        print('debug sent', debug_sent)
                    sent_wide_Y.append(sent_wide_cls)
                    debug_sent = []
                    curr_sent_X = []
                    curr_sent_words = []
                    curr_sent_Y = []
                #if cls == '5':
                #    print(line)
            X = np.array(X)
            Y = np.array(Y)
            sents_X = np.array(sents_X)
            sents_Y = np.array(sents_Y)
            
        print(Counter(classes))
        #print(Counter(words))
        #print(X.shape)
        #print(Y.shape)
        #print(sents_X.shape)
        #print(sents_Y.shape)
    #return sents_X, sents_Y
    return data_by_expressions

    
def bert_tensorflow_test(X_train, X_test, Y_train, Y_test):
    # Model
    model = Sequential()
    
    #model.add(Masking(mask_value=0.0, input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    model.add(Masking(mask_value=0.0, dtype='float64'))
    #forward_layer = LSTM(200, return_sequences=True)
    forward_layer = GRU(10, return_sequences=True, dropout=0.5)
    #backward_layer = LSTM(200, activation='relu', return_sequences=True,
    backward_layer = GRU(10, return_sequences=True, dropout=0.5,
                       go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer))#,
                         #input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    #model.add(TimeDistributed(Dense(NUM_CLASSES)))
    # Remove TimeDistributed() so that predictions are now made for the entire sentence
    model.add(TimeDistributed(Dense(NUM_CLASSES)))
    model.add(Activation('softmax'))
    #print('preds shape', model.predict(X_train[:3]).shape)
    #print('Y_train shape', Y_train[:3].shape)
    #print(list(Y_train[:3]))
    classes = []
    for y in Y_train:
        cls = np.argmax(y)
        classes.append(cls)
    print(Counter(classes))

    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    print('compiled model')
    model.fit(X_train, Y_train, batch_size=4, epochs=2, validation_split=0.0)
    print('fit model')
    eval = model.evaluate(X_test, Y_test, batch_size=4)
    #print('X_test[0]')
    #print(X_test[0])
    #print(X_train[0])
    preds = model.predict_proba(X_test, verbose=1, batch_size=4)
    #print(preds)
    num_correct = 0
    num_incorrect = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # idiomatic = 2, non-idiomatic = 3
    with open('preds_out_temp.txt', 'w') as tempoutf:
        for pred, y in zip(preds, Y_test):
            for token_pred, token_y in zip(pred, y):
                #print(np.argmax(token_pred), np.argmax(token_y))
                if np.argmax(token_y) == 2 or np.argmax(token_y) == 3:
                    if np.argmax(token_y) == np.argmax(token_pred):
                        num_correct += 1
                    else:
                        num_incorrect += 1
                if np.argmax(token_pred) == 2 and np.argmax(token_y) == 2:
                    TP += 1
                if np.argmax(token_pred) != 2 and np.argmax(token_y) != 2:
                    TN += 1
                if np.argmax(token_pred) == 2 and np.argmax(token_y) != 2:
                    FP += 1
                if np.argmax(token_pred) != 2 and np.argmax(token_y) == 2:
                    FN += 1
    if num_correct + num_incorrect == 0:
        custom_accuracy = 0
    else:
        custom_accuracy = num_correct/(num_correct+num_incorrect)
    print('custom accuracy is', custom_accuracy)
    for y in Y_test:
        cls = np.argmax(y)
        classes.append(cls)
    class_nums = Counter(classes)
    print(class_nums)
    default_acc = class_nums[2] / (class_nums[2] + class_nums[3])
    print('default accuracy is', default_acc, 'or', 1 - default_acc)
    return eval, custom_accuracy, default_acc, [TP, TN, FP, FN]


def get_already_processed(filename):
    if filename == None:
        return set([])
    already_processed = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            words = line.split(' ')
            if words[0] == 'EXP':
                already_processed.append(' '.join(words[1:]))
    return set(already_processed)



"""
    if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
"""

#X_train, X_test, Y_train, Y_test = get_bert_data(IN_FILENAME, None)
#bert_tensorflow_test(X_train, X_test, Y_train, Y_test)
#tf.disable_v2_behavior()
model = AutoModel.from_pretrained("EMBEDDIA/crosloengual-bert")
tokenizer = AutoTokenizer.from_pretrained("EMBEDDIA/crosloengual-bert")
print('model loaded ok')


dbe_train = get_xy_per_expression(IN_FILENAMES_TRAIN, tokenizer, model)
train_data = []
for k, i in dbe_train.items():
    train_data += i

dbe_test = get_xy_per_expression(IN_FILENAMES_TEST, tokenizer, model)
test_data = []
for k, i in dbe_test.items():
    test_data += i

with open('./test_results_12vs3_tokens.txt', 'w', encoding='utf-8') as outf:
    f1s = []
    accs = []
    shuffle(train_data)
    shuffle(test_data)
    train_data = train_data[:int(len(train_data)*0.8)]
    #print(k, len(train_data), len(test_data))
    
    #train_X = np.array([x[0] for x in train_data])
    train_X = [x[0] for x in train_data]
    train_Y = [x[1] for x in train_data]
    #test_X = np.array([x[0] for x in test_data])
    test_X = [x[0] for x in test_data]
    test_Y = [x[1] for x in test_data]
    

    #print(sent_train_Y[:10])
    #print(sent_test_Y[:10])
    
    #print(train_Y)
    for y in train_Y:
        y2 = to_categorical(y, num_classes=NUM_CLASSES)   
        #print(y2) 
    #train_Y = to_categorical(train_Y, num_classes=NUM_CLASSES)
    #test_Y = to_categorical(stest_Y, num_classes=NUM_CLASSES)
    train_Y = [to_categorical(y, num_classes=NUM_CLASSES) for y in train_Y]
    test_Y = [to_categorical(y, num_classes=NUM_CLASSES) for y in test_Y]
    train_Y = np.array(train_Y)
    test_Y = np.array(test_Y)
    print('training length', len(train_X))
    print('test lentgh', len(test_X))
    #for x in train_X:
        #print('x', x)
        #print('type', type(x))
        #print('len', len(x))
        #print('type x[0][0]', type(x[0][0]))
    padded_train_X = pad_sequences(train_X, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float', value=0.0)
    padded_test_X = pad_sequences(test_X, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float', value=0.0)
    padded_train_Y = pad_sequences(train_Y, padding='post', maxlen=MAX_SEQUENCE_LEN)
    padded_test_Y = pad_sequences(test_Y, padding='post', maxlen=MAX_SEQUENCE_LEN)
    #test_masking_layer = Masking(mask_value=0.0)
    #masked_embedding = test_masking_layer(padded_train_X)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(padded_train_X.shape, file=outf)
    #print('masked layer', masked_embedding._keras_mask, file=outf)
    #print('wrote masking layer')
    #raise TypeError
    results = bert_tensorflow_test(padded_train_X, padded_test_X, padded_train_Y, padded_test_Y)
    TP, TN, FP, FN = results[3]
    print('EXP', k[:-1], file=outf)
    print('eval is', results[0], file=outf)
    print('eval is', results[0])
    accs.append(results[0][1])
    print('custom accuracy is', results[1], file=outf)
    print('custom accuracy is', results[1])
    print('default accuracy is', results[2], 'or', 1-results[2], file=outf)
    print('default accuracy is', results[2], 'or', 1-results[2])
    print('num train is', len(padded_train_X), file=outf) 
    print('num train is', len(padded_train_X))
    print('num test is', len(padded_test_X), file=outf)
    print('num test is', len(padded_test_X))
    print('TP', TP, 'TN', TN, 'FP', FP, 'FN', FN, file=outf)
    print('TP', TP, 'TN', TN, 'FP', FP, 'FN', FN)

    if TP == 0:
        precision = 0
        recall = 0
        print('precision', 0, file=outf)
        print('recall', 0, file=outf)
        print('F1 score', 0)
        f1s.append(0)
    else:
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        print('precision', TP/(TP+FP), file=outf)
        print('recall', TP/(TP+FN), file=outf)
        print('F1 score', (2*precision*recall)/(precision+recall))
        f1s.append((2*precision*recall)/(precision+recall))
        
print('acc average', sum(accs)/len(accs))
print('F1 average', sum(f1s)/len(f1s))

    
