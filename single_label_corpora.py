from collections import defaultdict
from typing import Union, List, Dict, Sequence, Set

from flair.data import Label, Corpus, Dictionary, Sentence, Span
from torch.utils.data import Subset, Dataset


class FilteredColumnCorpus(Corpus):
    def __init__(self, corpus: Corpus, tags_to_keep: Union[str, List[str]], tag_type='ner'):
        super(FilteredColumnCorpus, self).__init__(corpus._train, corpus._dev, corpus._test, name=corpus.name)

        self.tags_to_keep = set("O")
        if isinstance(tags_to_keep, str):
            tags_to_keep = [tags_to_keep]
        for tag in tags_to_keep:
            if not tag.startswith(("B", "I", "S", "E")):
                self.tags_to_keep.add("B-" + tag)
                self.tags_to_keep.add("I-" + tag)
                self.tags_to_keep.add("E-" + tag)
                self.tags_to_keep.add("S-" + tag)

        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                if token.tags[tag_type].value not in self.tags_to_keep:
                    token.tags[tag_type] = Label("O")
        self.filter_empty_sentences()


class FilteringSentence(Sentence):
    def __init__(self, sentence: Sentence, tag_set: Set[str]):
        super().__init__()
        self.__dict__.update(sentence.__dict__)
        self.tag_set = tag_set

    def get_spans(self, tag_type: str, min_score=-1) -> List[Span]:

        spans: List[Span] = []

        current_span = []

        tags = defaultdict(lambda: 0.0)

        previous_tag_value: str = "O"
        for token in self:

            tag: Label = token.get_tag(tag_type)
            tag_value = tag.value

            # non-set tags are OUT tags
            if tag_value == "" or tag_value == "O":
                tag_value = "O-"

            # anything that is not a BIOES tag is a SINGLE tag
            if tag_value[0:2] not in ["B-", "I-", "O-", "E-", "S-"]:
                tag_value = "S-" + tag_value

            # anything that is not in the given tag_set is OUT
            if tag_value not in self.tag_set:
                tag_value = "O-"

            # anything that is not OUT is IN
            in_span = False
            if tag_value[0:2] not in ["O-"]:
                in_span = True

            # single and begin tags start a new span
            starts_new_span = False
            if tag_value[0:2] in ["B-", "S-"]:
                starts_new_span = True

            if (
                    previous_tag_value[0:2] in ["S-"]
                    and previous_tag_value[2:] != tag_value[2:]
                    and in_span
            ):
                starts_new_span = True

            if (starts_new_span or not in_span) and len(current_span) > 0:
                scores = [t.get_tag(tag_type).score for t in current_span]
                span_score = sum(scores) / len(scores)
                if span_score > min_score:
                    spans.append(
                        Span(
                            current_span,
                            tag=sorted(
                                tags.items(), key=lambda k_v: k_v[1], reverse=True
                            )[0][0],
                            score=span_score,
                        )
                    )
                current_span = []
                tags = defaultdict(lambda: 0.0)

            if in_span:
                current_span.append(token)
                weight = 1.1 if starts_new_span else 1.0
                tags[tag_value[2:]] += weight

            # remember previous tag
            previous_tag_value = tag_value

        if len(current_span) > 0:
            scores = [t.get_tag(tag_type).score for t in current_span]
            span_score = sum(scores) / len(scores)
            if span_score > min_score:
                spans.append(
                    Span(
                        current_span,
                        tag=sorted(tags.items(), key=lambda k_v: k_v[1], reverse=True)[
                            0
                        ][0],
                        score=span_score,
                    )
                )

        return spans


class FilteringSubset(Subset):
    def __init__(self, dataset: Dataset, indices: Sequence[int], tag_set: Set[str]):
        super().__init__(dataset, indices)
        self.tag_set = tag_set

    def __getitem__(self, idx):
        return FilteringSentence(self.dataset[self.indices[idx]], self.tag_set)


class DynamicSingleLabelCorpus(Corpus):
    def __init__(self, corpus: Corpus, tags_to_keep: Union[str, List[str]] = None, tag_type='ner'):
        super(DynamicSingleLabelCorpus, self).__init__(corpus._train, corpus._dev, corpus._test, name=corpus.name)

        if tags_to_keep is None:
            self.base_tags_to_keep = set()
            for sentence in self.get_all_sentences():
                for token in sentence.tokens:
                    tag: str = token.get_tag(tag_type).value
                    if not tag.startswith(tuple("BIES")):
                        continue
                    tag = strip_tag(tag)
                    self.base_tags_to_keep.add(tag)
        else:
            if isinstance(tags_to_keep, str):
                tags_to_keep = [tags_to_keep]
            self.base_tags_to_keep = set(map(strip_tag, tags_to_keep))

        self.tag_sets = {}
        for tag in self.base_tags_to_keep:
            tag_set = set()
            tag_set.add("B-" + tag)
            tag_set.add("I-" + tag)
            tag_set.add("E-" + tag)
            tag_set.add("S-" + tag)
            self.tag_sets.update({tag: tag_set})

        self._train_subsets = {}
        self._dev_subsets = {}
        self._test_subsets = {}
        for base_tag in self.base_tags_to_keep:
            self._train_subsets.update(
                {base_tag: Subset(self._train, self.get_indices(self._train, self.tag_sets[base_tag], tag_type))})
            self._dev_subsets.update(
                {base_tag: Subset(self._dev, self.get_indices(self._dev, self.tag_sets[base_tag], tag_type))})
            self._test_subsets.update(
                {base_tag: FilteringSubset(self._test,
                                           self.get_indices(self._test, self.tag_sets[base_tag], tag_type),
                                           self.tag_sets[base_tag])})

    def get_indices(self, dataset, tag_set, tag_type) -> List[int]:
        indices = []
        for index, sentence in enumerate(dataset):
            any_has_tag = False
            for token in sentence.tokens:
                value = token.get_tag(tag_type).value
                if value in tag_set:
                    any_has_tag = True
                    break
            if any_has_tag:
                indices.append(index)
        return indices

    def get_train(self, id: str):
        return self._train_subsets[id]

    def get_dev(self, id: str):
        return self._dev_subsets[id]

    def get_test(self, id: str):
        return self._test_subsets[id]

    def make_tag_dictionary(self, tag_type: str) -> Dict[str, Dictionary]:
        dicts: Dict[str: Dictionary] = {}

        # Make the tag dictionaries
        for tag, _ in self.tag_sets.items():
            tag_dictionary: Dictionary = Dictionary(add_unk=False)
            tag_dictionary.add_item("O")
            dicts.update({tag: tag_dictionary})

        for sentence in self.get_all_sentences():
            for token in sentence.tokens:
                value = token.get_tag(tag_type).value
                for tag, tag_set in self.tag_sets.items():
                    if value in tag_set:
                        dicts[tag].add_item(value)

        for tag, _ in self.tag_sets.items():
            dicts[tag].add_item("<START>")
            dicts[tag].add_item("<STOP>")

        return dicts


def strip_tag(tag):
    return tag.lstrip("BIES").lstrip("-")
