import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, TimeDistributed, Masking, GRU
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from ast import literal_eval
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter


# Modified tensorflow bilstm for sentence-level prediction

#VECTOR_DIM = 1024
VECTOR_DIM = 768 # bert
MAX_SEQUENCE_LEN = 30
IN_FILENAMES = ['./vectors_with_classes_bert_with_expressions_1_1.txt',
                './vectors_with_classes_bert_with_expressions_2_1.txt',
                './vectors_with_classes_bert_with_expressions_3_1.txt']
#IN_FILENAME = '../vector_datasets/vectors_with_classes_elmo_slo_top_full.txt'
#IN_FILENAME = '../vector_datasets/vectors_with_classes_elmo_slo_200k.txt'
#IN_FILENAME_TEST = '../vector_datasets/vectors_with_classes_elmo_slo_avg_2_50k.txt'
#IN_FILENAME_TEST = './vectors_with_classes_elmo_parseme_slo_top_full.txt'
IN_FILENAME_TEST = None
NUM_CLASSES = 5

def get_xy(filename):
    sent_wide_Y = []
    sents_X = []
    sents_Y = []
    curr_sent_X = []
    curr_sent_Y = []
    print('starting')
    CLS_TO_INT_DICT = {'NE': 3, 'DA': 2, '*':1, 'NEJASEN_ZGLED':4}
    classes = []
    words = []
    X = []
    Y = []
    print('reading file')
    with open(filename, 'r', encoding='utf-8') as f:
        debug_sent = []
        for i, line in enumerate(f):
            if i % 500 == 0:
                print(i)
            #if i >= 1500:
            #    break
            parts = line.split('\t')
            word = parts[0]
            if len(word) == 0:
                continue
           
            vector = parts[1]
            cls = parts[2][:-1]
            debug_sent.append((word, cls))
            #print(word, cls)
            classes.append(cls)
            words.append(word)
            if not (cls == 'DA' or cls == 'NE' or cls == '*'):
                continue
            curr_sent_X.append(literal_eval(vector))
            #print(len(literal_eval(vector)))
            curr_sent_Y.append(CLS_TO_INT_DICT[cls])
            
            if word[-1] == '.':
                sents_X.append(np.array(curr_sent_X))
                sents_Y.append(np.array(curr_sent_Y))
                sent_wide_cls = None
                if CLS_TO_INT_DICT['DA'] in curr_sent_Y:
                    sent_wide_cls = CLS_TO_INT_DICT['DA']
                elif CLS_TO_INT_DICT['NE'] in curr_sent_Y:
                    sent_wide_cls = CLS_TO_INT_DICT['NE']
                else:
                    sent_wide_cls = CLS_TO_INT_DICT['NEJASEN_ZGLED']
                    #print('debug sent', debug_sent)
                sent_wide_Y.append(sent_wide_cls)
                debug_sent = []
                curr_sent_X = []
                curr_sent_Y = []
            #if cls == '5':
            #    print(line)
        X = np.array(X)
        Y = np.array(Y)
        sents_X = np.array(sents_X)
        sents_Y = np.array(sents_Y)
        
        print(Counter(classes))
        print(Counter(words))
        print(X.shape)
        print(Y.shape)
        print(sents_X.shape)
        print(sents_Y.shape)
    #return sents_X, sents_Y
    return sents_X, sent_wide_Y

    
def get_xy_per_expression(filenames):
    data_by_expressions = {}
    sent_wide_Y = []
    sents_X = []
    sents_Y = []
    curr_sent_X = []
    curr_sent_Y = []
    expressions = []
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
                #if i >= 30000:
                #    break
                parts = line.split('\t')
                word = parts[0]
                #print(len(parts))
                if len(word) == 0:
                    continue
                if len(parts) != 4:
                    continue
                #print('len of parts', len(parts))
                exp = parts[0]
                vector = parts[1]
                cls = parts[2]
                expression = parts[3]
                #print(word, cls, expression)
                debug_sent.append((word, cls, expression))
                #print(word, cls)
                classes.append(cls)
                words.append(word)
                #print(exp, vector[:10], cls, expression)
                if not (cls == 'DA' or cls == 'NE' or cls == '*'):
                    continue
                curr_sent_X.append(literal_eval(vector))
                #print(len(literal_eval(vector)))
                curr_sent_Y.append(CLS_TO_INT_DICT[cls]) 
                if word[-1] == '.':
                    if expression not in data_by_expressions.keys():
                        #print(expression)
                        data_by_expressions[expression] = [(np.array(curr_sent_X), np.array(curr_sent_Y))]
                    else:
                        #print(expression)
                        data_by_expressions[expression].append((np.array(curr_sent_X), np.array(curr_sent_Y)))
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
                    curr_sent_Y = []
                #if cls == '5':
                #    print(line)
            X = np.array(X)
            Y = np.array(Y)
            sents_X = np.array(sents_X)
            sents_Y = np.array(sents_Y)
            
        print(Counter(classes))
        print(Counter(words))
        print(X.shape)
        print(Y.shape)
        print(sents_X.shape)
        print(sents_Y.shape)
    #return sents_X, sents_Y
    return data_by_expressions

def get_bert_data_per_expression(IN_FILENAME):
    X, Y = get_xy(IN_FILENAME)

    
    
def get_bert_data(IN_FILENAME, IN_FILENAME_TEST):    
    if IN_FILENAME_TEST != None:
        X_train, Y_train = get_xy(IN_FILENAME)
        print(Y_train[:5])
        Y_train = to_categorical(Y_train, num_classes=NUM_CLASSES)
        Y_train = np.array(Y_train)
        X_test, Y_test = get_xy(IN_FILENAME_TEST)
        Y_test = to_categorical(Y_test, num_classes=NUM_CLASSES)
        Y_test = np.array(Y_test)
        # Pad everything
        X_train = pad_sequences(X_train, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float')
        X_test = pad_sequences(X_test, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float')
        #Y_train = pad_sequences(Y_train, padding='post', maxlen=MAX_SEQUENCE_LEN)
        #Y_test = pad_sequences(Y_test, padding='post', maxlen=MAX_SEQUENCE_LEN)
    else:
        X, Y = get_xy(IN_FILENAME)
        print('first X')
        print(X[0])
        Y = to_categorical(Y, num_classes=NUM_CLASSES)
        Y = np.array(Y)
        print('y shape', Y.shape)
        print('y[0]', Y[0])
        print('y[1]', Y[1])
        print('y[2]', Y[2])
        sent_lens = [len(x) for x in X]
        print(sent_lens)
        padded_X = pad_sequences(X, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float')
        print('padded_X')
        print(padded_X[0])
        #padded_Y = pad_sequences(Y, padding='post', maxlen=MAX_SEQUENCE_LEN)
        sent_lens = [len(x) for x in padded_X]
        print(sent_lens)

        #exit()
        X_train, X_test, Y_train, Y_test = train_test_split(padded_X, Y, test_size=0.33, shuffle=True)

    print(X_train.shape, X_train[0].shape)
    print('first X_test')
    print(X_test[0])
    return X_train, X_test, Y_train, Y_test

    
def bert_tensorflow_test(X_train, X_test, Y_train, Y_test):
    # Model
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    #forward_layer = LSTM(200, return_sequences=True)
    forward_layer = GRU(10, return_sequences=False, dropout=0.5)
    #backward_layer = LSTM(200, activation='relu', return_sequences=True,
    backward_layer = GRU(10, return_sequences=False, dropout=0.5,
                       go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                         input_shape=(MAX_SEQUENCE_LEN,VECTOR_DIM)))
    #model.add(TimeDistributed(Dense(NUM_CLASSES)))
    # Remove TimeDistributed() so that predictions are now made for the entire sentence
    model.add(Dense(NUM_CLASSES))
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
    model.fit(X_train, Y_train, batch_size=8, epochs=10)#, validation_split=0.1)
    print('fit model')
    eval = model.evaluate(X_test, Y_test, batch_size=8)
    #print('X_test[0]')
    #print(X_test[0])
    #print(X_train[0])
    preds = model.predict_proba(X_test, verbose=1, batch_size=8)
    print(preds)
    num_correct = 0
    num_incorrect = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # idiomatic = 2, non-idiomatic = 3
    with open('preds_out_temp.txt', 'w') as tempoutf:
        for pred, y in zip(preds, Y_test):
            if np.argmax(y) == 2 or np.argmax(y) == 3:
                if np.argmax(y) == np.argmax(pred):
                    num_correct += 1
                else:
                    num_incorrect += 1
            if np.argmax(pred) == 2 and np.argmax(y) == 2:
                TP += 1
            if np.argmax(pred) == 3 and np.argmax(y) == 3:
                TN += 1
            if np.argmax(pred) == 2 and np.argmax(y) == 3:
                FP += 1
            if np.argmax(pred) == 3 and np.argmax(y) == 2:
                FN += 1
    custom_accuracy = num_correct/(num_correct+num_incorrect)
    print('custom accuracy is', num_correct/(num_correct+num_incorrect))
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



gpus = tf.config.experimental.list_physical_devices('GPU')
#already_processed = get_already_processed('./test_results_named_32.txt')
already_processed = set()
print(already_processed)
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


#X_train, X_test, Y_train, Y_test = get_bert_data(IN_FILENAME, None)
#bert_tensorflow_test(X_train, X_test, Y_train, Y_test)
dbe = get_xy_per_expression(IN_FILENAMES)
print(dbe.keys())

with open('./test_results_named.txt', 'w', encoding='utf-8') as outf:
    for k, i in dbe.items():
        print(k, len(i))
    
    for k, i in dbe.items():
        if k in already_processed:
            print('already processed', k)
            continue
        train_data = []
        test_data = []
        for k2, i2 in dbe.items():
            if k == k2:
                test_data += i2
            else:
                train_data += i2
        
        #print(k, len(train_data), len(test_data))
        
        train_X = np.array([x[0] for x in train_data])
        train_Y = [x[1] for x in train_data]
        test_X = np.array([x[0] for x in test_data])
        test_Y = [x[1] for x in test_data]
        
        sent_train_Y = []
        sent_test_Y = []

        for y in train_Y:
            #print(y)
            if 2 in y:
                sent_train_Y.append(2)
            elif 3 in y:
                sent_train_Y.append(3)
            elif 4 in y:
                sent_train_Y.append(4)
            else:
                sent_train_Y.append(1)

        for y in test_Y:
            #print(y)
            if 2 in y:
                sent_test_Y.append(2)
            elif 3 in y:
                sent_test_Y.append(3)
            elif 4 in y:
                sent_test_Y.append(4)
            else:
                sent_test_Y.append(1)

        #print(sent_train_Y[:10])
        #print(sent_test_Y[:10])
        
        train_Y = to_categorical(sent_train_Y, num_classes=NUM_CLASSES)
        test_Y = to_categorical(sent_test_Y, num_classes=NUM_CLASSES)
        train_Y = np.array(train_Y)
        test_Y = np.array(test_Y)
        print('training shape', train_X.shape)
        print('test shape', test_X.shape)
        padded_train_X = pad_sequences(train_X, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float')
        padded_test_X = pad_sequences(test_X, padding='post', maxlen=MAX_SEQUENCE_LEN, dtype='float')
        
        results = bert_tensorflow_test(padded_train_X, padded_test_X, train_Y, test_Y)
        TP, TN, FP, FN = results[3]
        print('EXP', k[:-1], file=outf)
        print('eval is', results[0], file=outf)
        print('eval is', results[0])
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
            print('precision', 0, file=outf)
            print('recall', 0, file=outf)
        else:
            print('precision', TP/(TP+FP), file=outf)
            print('recall', TP/(TP+FN), file=outf)
        
