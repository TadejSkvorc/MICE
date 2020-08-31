# MICE: Mining idioms with contextual embeddings

This repository contains the code for the paper MICE: Mining idioms with contextual embeddings

The code for running the classification models is located in /tools. The directory contains four approaches:
* A baseline BOW classifier
* An approach using multilingual crosloengual bert embeddings (https://huggingface.co/EMBEDDIA/crosloengual-bert)
* An approach using multilingual from Google Research (multi_cased_L-12_H-768_A-12, https://github.com/google-research/bert/blob/master/multilingual.md)
* An approach using slovenian elmo embeddings (https://www.clarin.si/repository/xmlui/handle/11356/1257)

The last two approaches require the models to be downloaded manually before they can be used. Crosloengual embeddings are downloaded by the pyton script using the huggingface transformers library

As input, the first two models take a tab seperated file using the following format:
* The first column contains the word
* The second column contains the class ('DA' for tokens with idiomatic meanings, 'NE' for tokens with non-idiomatic meanings, and '*' for tokens that do not appear in the potentially-idiomatic phrase). 
* The third column contains the potentially-idiomatic phrase
    
The second two models take a similar files, but require the word embeddings to be pre-computed, using the following format:
* The first column contains the word
* The second column contains the pre-computed embeddings, 
* The third column contains the class ('DA' for tokens with idiomatic meanings, 'NE' for tokens with non-idiomatic meanings, and '*' for tokens that do not appear in the potentially-idiomatic phrase)
* The fourth column contains the potentially-idiomatic phrase

The /test_datasets folder contains a slovene dataset in the first format (without the pre-computed embeddings)

