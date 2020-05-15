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
otaggers = []
for tagger_name in glob("resources/taggers/multi-wikiner/*-best-model.pt"):
    tagger: SequenceTagger = SequenceTagger.load(tagger_name)
    otaggers.append(tagger)
    tag = tagger_name.split("/")[-1].split("-")[0]
    tagger.tag_type = tag

    # share parameters
    if len(otaggers) > 1:
        first_tagger = otaggers[0]
        tagger.embedding2nn = first_tagger.embedding2nn
        tagger.rnn = first_tagger.rnn
        if tagger.train_initial_hidden_state:
            tagger.hs_initializer = first_tagger.hs_initializer
            tagger.lstm_init_h = first_tagger.lstm_init_h
            tagger.lstm_init_c = first_tagger.lstm_init_c

taggers += otaggers

sentences = [
    Sentence(
        "Zellecken der Lamina deutlich verdickt, die Zellwände getüpfelt."
    ),
    Sentence(
        "Die Spitze der Parichätialblätter entwickelt aus zahlreichen Zellen braune Rhizoiden."
    ),
    Sentence(
        "Blattränder überall bis zum nächsten Blatte herablaufend."
    ),
    Sentence(
        "Gefunden am Rande des Hammersees "
    ),
    Sentence(
        "Die Blumen sind von vollendeter Form mit elegant gewellten und nach Art der Petunien und chinesischen Primeln gefranzten Petalen, oft zur Füllung neigend und von edler, meist aufrechter Faltung ."
    ),
    Sentence(
        "Obwohl fast 70 Jahre alt, als er sich der mühevollen Aufgabe unterzog, diese schwierige Pilzgruppe systematisch zu beschreiben, widmete er dem Werke mit dem Eifer und der Schaffenskraft eines Jugendlichen seine letzten Lebensjahre fast ausschliefslich."
    ),
    Sentence(
        "Dies gilt allerdings im allgemeinen nur von den Nordküsten der Inseln, welche von März bis Oktober der abkühlenden Wirkung des Nordostpassates ausgesetzt sind."
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
