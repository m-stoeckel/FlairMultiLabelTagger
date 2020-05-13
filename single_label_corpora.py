import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Union, List, Dict, Sequence, Set

from flair.data import Label, Corpus, Dictionary, Sentence, Span, Token, FlairDataset
from torch.utils.data import Subset, Dataset, random_split

log = logging.getLogger("flair")


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
            base_tagset = self.tag_sets[base_tag]
            self._train_subsets.update(
                {base_tag: Subset(self._train, self.get_indices(self._train, base_tagset, tag_type))})
            self._dev_subsets.update(
                {base_tag: FilteringSubset(self._dev,
                                           self.get_indices(self._dev, base_tagset, tag_type),
                                           base_tagset)})
            self._test_subsets.update(
                {base_tag: FilteringSubset(self._test,
                                           self.get_indices(self._test, base_tagset, tag_type),
                                           base_tagset)})

    @staticmethod
    def get_indices(dataset, tag_set, tag_type) -> List[int]:
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
    return re.sub("[BIES]-", "", tag)


class MultiLabelColumnCorpus(Corpus):
    _default_tags_to_keep = {'Act_Action_Activity', 'Animal_Fauna', 'Archaea', 'Artifact', 'Attribute_Property',
                             'Bacteria', 'Body_Corpus', 'Chromista', 'Cognition_Ideation', 'Communication',
                             'Event_Happening', 'Feeling_Emotion', 'Food', 'Fungi', 'Group_Collection', 'Habitat',
                             'Lichen', 'Location_Place', 'Morphology', 'Motive', 'NaturalObject', 'NaturalPhenomenon',
                             'Person_HumanBeing', 'Plant_Flora', 'Possession_Property', 'Process', 'Protozoa',
                             'Quantity_Amount', 'Relation', 'Reproduction', 'Shape', 'Society', 'State_Condition',
                             'Substance', 'Taxon', 'Time', 'Viruses'}

    def __init__(
            self,
            data_folder: Union[str, Path],
            column_format: Dict[int, str],
            train_file=None,
            test_file=None,
            dev_file=None,
            tag_to_bioes=None,
            comment_symbol: str = None,
            in_memory: bool = True,
            encoding: str = "utf-8",
            document_separator_token: str = None,
            tags_to_keep: List[str] = None
    ):
        """
        Instantiates a Corpus from CoNLL column-formatted task data such as CoNLL03 or CoNLL2000.

        :param data_folder: base folder with the task data
        :param column_format: a map specifying the column format
        :param train_file: the name of the train file
        :param test_file: the name of the test file
        :param dev_file: the name of the dev file, if None, dev data is sampled from train
        :param tag_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        :return: a Corpus with annotated train, dev and test data
        """

        if type(data_folder) == str:
            data_folder: Path = Path(data_folder)

        if train_file is not None:
            train_file = data_folder / train_file
        if test_file is not None:
            test_file = data_folder / test_file
        if dev_file is not None:
            dev_file = data_folder / dev_file

        # automatically identify train / test / dev files
        if train_file is None:
            for file in data_folder.iterdir():
                file_name = file.name
                if file_name.endswith(".gz"):
                    continue
                if "train" in file_name and not "54019" in file_name:
                    train_file = file
                if "dev" in file_name:
                    dev_file = file
                if "testa" in file_name:
                    dev_file = file
                if "testb" in file_name:
                    test_file = file

            # if no test file is found, take any file with 'test' in name
            if test_file is None:
                for file in data_folder.iterdir():
                    file_name = file.name
                    if file_name.endswith(".gz"):
                        continue
                    if "test" in file_name:
                        test_file = file

        log.info("Reading data from {}".format(data_folder))
        log.info("Train: {}".format(train_file))
        log.info("Dev: {}".format(dev_file))
        log.info("Test: {}".format(test_file))

        # get train data
        train = SingleLabelColumnDataset(
            train_file,
            column_format,
            tag_to_bioes,
            encoding=encoding,
            comment_symbol=comment_symbol,
            in_memory=in_memory,
            document_separator_token=document_separator_token,
        )

        # read in test file if exists, otherwise sample 10% of train data as test dataset
        if test_file is not None:
            test = SingleLabelColumnDataset(
                test_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
            )
        else:
            train_length = len(train)
            test_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - test_size, test_size])
            train = splits[0]
            test = splits[1]

        # read in dev file if exists, otherwise sample 10% of train data as dev dataset
        if dev_file is not None:
            dev = SingleLabelColumnDataset(
                dev_file,
                column_format,
                tag_to_bioes,
                encoding=encoding,
                comment_symbol=comment_symbol,
                in_memory=in_memory,
                document_separator_token=document_separator_token,
            )
        else:
            train_length = len(train)
            dev_size: int = round(train_length / 10)
            splits = random_split(train, [train_length - dev_size, dev_size])
            train = splits[0]
            dev = splits[1]

        super(MultiLabelColumnCorpus, self).__init__(train, dev, test, name=data_folder.name)

        ###############
        # Multi Label #
        ###############

        if tags_to_keep is None:
            self.base_tags_to_keep = self._default_tags_to_keep
            # self.base_tags_to_keep = set()
            # for sentence in self.get_all_sentences():
            #     for token in sentence.tokens:
            #         tag: str = token.get_tag(tag_type).value
            #         if not tag.startswith(tuple("BIES")):
            #             continue
            #         tag = strip_tag(tag)
            #         self.base_tags_to_keep.add(tag)
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
            base_tagset = self.tag_sets[base_tag]
            self._train_subsets.update(
                {base_tag: Subset(self._train, self.get_indices(self._train, base_tagset, base_tag))})
            self._dev_subsets.update(
                {base_tag: Subset(self._dev, self.get_indices(self._dev, base_tagset, base_tag))})
            self._test_subsets.update(
                {base_tag: Subset(self._test, self.get_indices(self._test, base_tagset, base_tag))})

    @staticmethod
    def get_indices(dataset, tag_set, tag_type) -> List[int]:
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

    def make_tag_dictionary(self, _=None, add_unk=True) -> Dict[str, Dictionary]:
        dicts: Dict[str: Dictionary] = {}
        # Make the tag dictionaries
        for base_tag, tag_set in self.tag_sets.items():
            if min(len(self.get_train(base_tag)), len(self.get_dev(base_tag)), len(self.get_test(base_tag))) == 0:
                continue
            tag_dictionary: Dictionary = Dictionary(add_unk=add_unk)
            tag_dictionary.add_item("O")
            for tag in sorted(tag_set):
                tag_dictionary.add_item(tag)
            tag_dictionary.add_item("<START>")
            tag_dictionary.add_item("<STOP>")
            dicts.update({base_tag: tag_dictionary})

        return dicts


class SingleLabelColumnDataset(FlairDataset):
    def __init__(
            self,
            path_to_column_file: Path,
            column_name_map: Dict[int, str],
            tags_to_bioes: List[str] = None,
            comment_symbol: str = '#',
            in_memory: bool = True,
            document_separator_token: str = None,
            encoding: str = "utf-8",
    ):
        """
        Instantiates a column dataset (typically used for sequence labeling or word-level prediction).

        :param path_to_column_file: path to the file with the column-formatted data
        :param column_name_map: a map specifying the column format
        :param tags_to_bioes: whether to convert to BIOES tagging scheme
        :param comment_symbol: if set, lines that begin with this symbol are treated as comments
        :param in_memory: If set to True, the dataset is kept in memory as Sentence objects, otherwise does disk reads
        :param document_separator_token: If provided, multiple sentences are read into one object. Provide the string token
        that indicates that a new document begins
        """
        assert path_to_column_file.exists()
        self.path_to_column_file = path_to_column_file
        self.tags_to_bioes = tags_to_bioes
        self.column_name_map = column_name_map
        self.comment_symbol = comment_symbol
        self.document_separator_token = document_separator_token

        # store either Sentence objects in memory, or only file offsets
        self.in_memory = in_memory
        if self.in_memory:
            self.sentences: List[Sentence] = []
        else:
            self.indices: List[int] = []

        self.total_sentence_count: int = 0

        # most data sets have the token text in the first column, if not, pass 'text' as column
        self.text_column: int = 0
        for column in self.column_name_map:
            if column_name_map[column] == "text":
                self.text_column = column

        # determine encoding of text file
        self.encoding = encoding

        sentence: Sentence = Sentence()
        with open(str(self.path_to_column_file), encoding=self.encoding) as f:

            line = f.readline()
            position = 0

            while line:

                if self.comment_symbol is not None and line.startswith(comment_symbol):
                    line = f.readline()
                    continue

                if self.__line_completes_sentence(line):

                    if len(sentence) > 0:

                        sentence.infer_space_after()
                        if self.in_memory:
                            if self.tags_to_bioes is not None:
                                for tag in self.tags_to_bioes:
                                    sentence.convert_tag_scheme(
                                        tag_type=tag, target_scheme="iobes"
                                    )
                            self.sentences.append(sentence)
                        else:
                            self.indices.append(position)
                            position = f.tell()
                        self.total_sentence_count += 1
                    sentence: Sentence = Sentence()

                else:
                    fields: List[str] = re.split("\s+", line)
                    token = Token(fields[self.text_column])
                    for column in column_name_map:
                        if len(fields) > column:
                            if column != self.text_column:
                                token.add_tag(
                                    self.column_name_map[column], fields[column]
                                )

                    if not line.isspace():
                        sentence.add_token(token)

                line = f.readline()

        if len(sentence.tokens) > 0:
            sentence.infer_space_after()
            if self.in_memory:
                self.sentences.append(sentence)
            else:
                self.indices.append(position)
            self.total_sentence_count += 1

    def __getitem__(self, index: int = 0) -> Sentence:

        if self.in_memory:
            sentence = self.sentences[index]

        else:
            with open(str(self.path_to_column_file), encoding=self.encoding) as file:
                file.seek(self.indices[index])
                line = file.readline()
                sentence: Sentence = Sentence()
                while line:
                    if self.comment_symbol is not None and line.startswith(
                            self.comment_symbol
                    ):
                        line = file.readline()
                        continue

                    if self.__line_completes_sentence(line):
                        if len(sentence) > 0:
                            sentence.infer_space_after()
                            if self.tags_to_bioes is not None:
                                for tag in self.tags_to_bioes:
                                    sentence.convert_tag_scheme(
                                        tag_type=tag, target_scheme="iobes"
                                    )
                            return sentence

                    else:
                        fields: List[str] = re.split("\s+", line)
                        token = Token(fields[self.text_column])
                        for column in self.column_name_map:
                            if len(fields) > column:
                                if column != self.text_column:
                                    token.add_tag(
                                        self.column_name_map[column], fields[column]
                                    )

                        if not line.isspace():
                            sentence.add_token(token)

                    line = file.readline()
        return sentence

    def __line_completes_sentence(self, line: str) -> bool:
        sentence_completed = line.isspace()
        if self.document_separator_token:
            sentence_completed = False
            fields: List[str] = re.split("\s+", line)
            if len(fields) >= self.text_column:
                if fields[self.text_column] == self.document_separator_token:
                    sentence_completed = True
        return sentence_completed

    def is_in_memory(self) -> bool:
        return self.in_memory

    def __len__(self):
        return self.total_sentence_count
