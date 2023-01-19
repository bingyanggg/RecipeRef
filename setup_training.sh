#!/bin/bash

#python minimize.py
#python get_char_vocab.py



#python filter_embeddings.py data/glove.840B.300d.txt ./data/fulldata/with_state/train.english.jsonlines ./data/fulldata/with_state/dev.english.jsonlines
python filter_embeddings.py data/glove.840B.300d.txt /home/bingyang/projects/coref-under-transformation/data/r2vq_cutler_120/reciperef/flatten/fine/train.jsonl /home/bingyang/projects/coref-under-transformation/data/r2vq_cutler_120/reciperef/flatten/fine/dev.jsonl


#python cache_elmo.py ./data/fulldata/with_state/train.english.jsonlines ./data/fulldata/with_state/dev.english.jsonlines ./data/fulldata/with_state/test.english.jsonlines
#python cache_elmo.py train.english.jsonlines
python cache_elmo.py /home/bingyang/projects/coref-under-transformation/data/r2vq_cutler_120/reciperef/flatten/fine/train.jsonl /home/bingyang/projects/coref-under-transformation/data/r2vq_cutler_120/reciperef/flatten/fine/dev.jsonl

