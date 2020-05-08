from typing import List

import flair
import torch
from flair.datasets import WIKINER_GERMAN
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, \
    BytePairEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger
from flair.trainers.trainer import ModelTrainer

flair.device = torch.device("cuda:0")

corpus = WIKINER_GERMAN(in_memory=True)
tag_type = 'ner'
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)

embedding_types: List[TokenEmbeddings] = [
    WordEmbeddings('de-wiki'),
    BytePairEmbeddings('de', 100, 5000),
    CharacterEmbeddings(),
    FlairEmbeddings('de-forward'),
    FlairEmbeddings('de-backward')
]
embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    use_crf=True
)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)
trainer.train(
    'resources/taggers/ordinary-ner',
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=25,
    monitor_train=True,
    monitor_test=True,
    embeddings_storage_mode="gpu"
)
