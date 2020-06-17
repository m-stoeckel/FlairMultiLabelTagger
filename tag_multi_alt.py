from glob import glob

import flair
import torch
from flair.data import Sentence
from flair.models import SequenceTagger

flair.device = torch.device("cuda:0")
taggers = []
for tagger_name in glob("path/to/models/*-best-model.pt"):
    tagger: SequenceTagger = SequenceTagger.load(tagger_name)
    taggers.append(tagger)

    # share parameters
    if len(taggers) > 1:
        first_tagger = taggers[0]
        tagger.embeddings[0] = first_tagger.embeddings[0]
        tagger.embeddings[1] = first_tagger.embeddings[1]
        tagger.embeddings[3] = first_tagger.embeddings[3]
        tagger.embeddings[4] = first_tagger.embeddings[4]

# sentences = load_plaintext()
sentences = [
    Sentence(
        "Zellecken der Lamina deutlich verdickt, die Zellwände getüpfelt."
    ),
    Sentence(
        "Die Spitze der Parichätialblätter entwickelt aus zahlreichen Zellen braune Rhizoiden."
    )
]

for tagger in taggers:
    for sent in sentences:
        tagger.predict(sent)

# Write output as XMI?
