from typing import List, Dict

import flair
import torch
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, \
    BytePairEmbeddings, FlairEmbeddings
from flair.models import SequenceTagger

from multi_model_trainer import MultiModelTrainer
from single_label_corpora import MultiLabelColumnCorpus

flair.device = torch.device("cuda:0")
column_format = {
    0: 'text', 1: 'pos', 2: 'lemma', 3: 'Act_Action_Activity', 4: 'Animal_Fauna', 5: 'Archaea', 6: 'Artifact',
    7: 'Attribute_Property', 8: 'Bacteria', 9: 'Body_Corpus', 10: 'Chromista', 11: 'Cognition_Ideation',
    12: 'Communication', 13: 'Event_Happening', 14: 'Feeling_Emotion', 15: 'Food', 16: 'Fungi', 17: 'Group_Collection',
    18: 'Habitat', 19: 'Lichen', 20: 'Location_Place', 21: 'Morphology', 22: 'Motive', 23: 'NaturalObject',
    24: 'NaturalPhenomenon', 25: 'Person_HumanBeing', 26: 'Plant_Flora', 27: 'Possession_Property', 28: 'Process',
    29: 'Protozoa', 30: 'Quantity_Amount', 31: 'Relation', 32: 'Reproduction', 33: 'Shape', 34: 'Society',
    35: 'State_Condition', 36: 'Substance', 37: 'Taxon', 38: 'Time', 39: 'Viruses'
}

corpus = MultiLabelColumnCorpus(
    "resources/data/", column_format,
    train_file="train_biofid.conll",
    in_memory=True,
    tag_to_bioes=list(column_format.values())[3:],
    comment_symbol="#",
    tags_to_keep=['Animal_Fauna', 'Plant_Flora', 'Taxon']
)
tag_dictionaries = corpus.make_tag_dictionary()
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
        tag_type=tag,
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
    'resources/taggers/multi-biofid',
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=25,
    monitor_train=True,
    monitor_test=True,
    embeddings_storage_mode="gpu"
)
