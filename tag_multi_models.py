from glob import glob

import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger

flair.device = torch.device("cuda:0")
taggers = []
for tagger_name in glob("resources/taggers/multi-biofid/*-best-model.pt"):
    tagger: SequenceTagger = SequenceTagger.load(tagger_name)
    taggers.append(tagger)

    # share parameters
    if len(taggers) > 1:
        first_tagger = taggers[0]
        tagger.embedding2nn = first_tagger.embedding2nn
        tagger.rnn = first_tagger.rnn
        if tagger.train_initial_hidden_state:
            tagger.hs_initializer = first_tagger.hs_initializer
            tagger.lstm_init_h = first_tagger.lstm_init_h
            tagger.lstm_init_c = first_tagger.lstm_init_c

sentences = [
    Sentence(
        "Hibiskus (Hibiscus) – auf Deutsch Eibisch – ist eine Pflanzengattung aus der Familie der Malvengewächse (Malvaceae) mit etwa 200 bis 675 Arten."
    ),
    Sentence(
        "Im deutschen Sprachgebrauch ist damit meistens der Rotfuchs gemeint, allgemeiner die Gattungsgruppe der Echten Füchse."
    ),
    Sentence(
        "In einer aktuellen Systematik der Hunde, die auf molekulargenetischen Untersuchungen gründete, wurde die Gattung Vulpes als Schwestertaxon dem Marderhund (Nyctereutes procyonoides) gegenübergestellt."
    ),
    Sentence(
        "Die Zannichellia fand ich in der Nähe von Czattkau unterhalb Dirschau in einem Innendeichkolk, und zwar in der seltenen Form pedicellata; es scheint derselbe Fundort zu sein, den Caspary bei Czattkau angibt."
    ),
    Sentence(
        "Beim Verpuppen sind die Puppenwiegen mehr den Aussenwänden des Zapfens genähert und sind hauptsächlich im stärkeren Teile des Zapfens angelegt, Anfangs September findet man die elfenbeinweisse Puppa libera."
    )
]

for tagger in taggers:
    for sent in sentences:
        tagger.predict(sent)

with open("tag_multi_results.txt", 'w', encoding='utf8') as fout:
    for sent in sentences:
        fout.write(sent.to_tagged_string())
        fout.write("\n")

