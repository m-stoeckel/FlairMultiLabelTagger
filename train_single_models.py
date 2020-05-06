from typing import List, Dict

import flair
import torch
from flair.datasets import ColumnCorpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, BytePairEmbeddings
from flair.models import SequenceTagger

from multi_model_trainer import MultiModelTrainer
from single_label_corpora import DynamicSingleLabelCorpus

flair.device = torch.device("cuda:0")

corpus = ColumnCorpus("resources/data/", {0: "text", 1: "pos", 2: "ner"}, train_file="test_corpus.conll",
                      in_memory=True, tag_to_bioes='ner')
tag_type = 'ner'
corpus = DynamicSingleLabelCorpus(corpus, tag_type=tag_type)
tag_dictionaries = corpus.make_tag_dictionary(tag_type=tag_type)
for tag, tag_dictionary in tag_dictionaries.items():
    print(tag, str(tag_dictionary))

# 4. initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de'),
    BytePairEmbeddings('de', 100, 5000),
    CharacterEmbeddings(),
    # FlairEmbeddings('de-forward'),
    # FlairEmbeddings('de-backward')
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

taggers: Dict[str, SequenceTagger] = {}
tag, tag_dictionary = list(tag_dictionaries.items())[0]
first_tagger: SequenceTagger = SequenceTagger(
    hidden_size=64,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    use_crf=True
)
taggers.update({tag: first_tagger})

for tag, tag_dictionary in list(tag_dictionaries.items())[1:]:
    tagger: SequenceTagger = SequenceTagger(
        hidden_size=64,
        embeddings=embeddings,
        tag_dictionary=tag_dictionary,
        tag_type=tag_type,
        use_crf=True
    )
    tagger.embedding2nn = first_tagger.embedding2nn
    tagger.rnn = first_tagger.rnn
    if tagger.train_initial_hidden_state:
        tagger.hs_initializer = first_tagger.hs_initializer
        tagger.lstm_init_h = first_tagger.lstm_init_h
        tagger.lstm_init_c = first_tagger.lstm_init_c

    tagger.rnn = first_tagger.rnn
    taggers.update({tag: tagger})

trainer: MultiModelTrainer = MultiModelTrainer(taggers, corpus)
trainer.train(
    'resources/taggers/example-ner',
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=3,
    monitor_train=True,
    monitor_test=True,
    embeddings_storage_mode="gpu"
)
