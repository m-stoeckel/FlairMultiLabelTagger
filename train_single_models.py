from typing import List, Dict

import flair
import torch
from flair.datasets import WIKINER_GERMAN
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, \
    BytePairEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger

from multi_model_trainer import MultiModelTrainer
from single_label_corpora import DynamicSingleLabelCorpus

flair.device = torch.device("cuda:0")

corpus = WIKINER_GERMAN(in_memory=False)
tag_type = 'ner'
corpus = DynamicSingleLabelCorpus(corpus, tag_type=tag_type)
tag_dictionaries = corpus.make_tag_dictionary(tag_type=tag_type)
for tag, tag_dictionary in tag_dictionaries.items():
    print(tag, str(tag_dictionary))

embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de-wiki'),
    BytePairEmbeddings('de', 100, 5000),
    CharacterEmbeddings(),
    FlairEmbeddings('de-forward'),
    FlairEmbeddings('de-backward')
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

taggers: Dict[str, SequenceTagger] = {}
tag_dictionary_items = list(tag_dictionaries.items())
for tag, tag_dictionary in tag_dictionary_items:
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=256,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True
    )
    taggers.update({tag: tagger})

    # share parameters
    if tag != tag_dictionary_items[0][0]:
        first_tagger = taggers.get(tag_dictionary_items[0][0])
        tagger.embedding2nn = first_tagger.embedding2nn
        tagger.rnn = first_tagger.rnn
        if tagger.train_initial_hidden_state:
            tagger.hs_initializer = first_tagger.hs_initializer
            tagger.lstm_init_h = first_tagger.lstm_init_h
            tagger.lstm_init_c = first_tagger.lstm_init_c

trainer: MultiModelTrainer = MultiModelTrainer(taggers, corpus)
trainer.train(
    'resources/taggers/multi-wikiner',
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=25,
    monitor_train=True,
    monitor_test=True,
    embeddings_storage_mode="gpu"
)
