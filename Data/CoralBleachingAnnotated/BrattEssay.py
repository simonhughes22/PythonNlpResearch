
import os

from FindFiles import find_files
import Settings

from collections import defaultdict
from nltk.tokenize import sent_tokenize

class AnnotationBase(object):
    def __init__(self, line):
        self.split = line.strip().split("\t")
        self.id = self.split[0]
        self.type = self.id[0]

        self.contents = self.split[1].replace("  ", " ").split(" ")
        self.code = self.contents[0]

    def __repr__(self):
        return "[" + str(type(self)) + "]" + self.code + ":" + self.type

def is_compound(line):
    if ";" in line:
        l = line.replace(" ","")
        index = l.index(";")
        return l[index-1].isdigit() or l[index+1].isdigit()

class CompoundTextAnnotation(AnnotationBase):
    def __init__(self, line, full_text):
        """ Basic essay element, has
                txt
                id
                start
                end
        """
        AnnotationBase.__init__(self, line)
        self.txt = self.split[2]

        first  = self.contents[1]
        second, third = self.contents[2].split(";")
        fourth = self.contents[3]

        first, second, third, fourth = int(first) + 1, int(second) + 1, int(third) + 1, int(fourth) + 1

        txt_split = self.txt.split(" ")

        txt_first  = txt_split[0]
        txt_second = txt_split[-1]

        if len(txt_split) != 2:
            length = len(txt_split)
            for i in range(1, length):
                a = " ".join(txt_split[:i])
                b = " ".join(txt_split[i:])
                if a in full_text[first-10:second+10] and b in full_text[third-10:third+10]:
                    txt_first = a
                    txt_second = b
                    break

        self.first_part  = TextAnnotation(self.id + "\t" + self.code + " " + str(first) + " " + str(second) + "\t" + txt_first, full_text)
        self.second_part = TextAnnotation(self.id + "\t" + self.code + " " + str(third) + " " + str(fourth) + "\t" + txt_second, full_text)
        pass

class TextAnnotation(AnnotationBase):
    def __init__(self, line, full_text):
        """ Basic essay element, has
                txt
                id
                start
                end
        """
        AnnotationBase.__init__(self, line)
        self.start = int(self.contents[1])
        self.end = int(self.contents[2])
        self.txt = ""

        if len(self.split) > 2:
            self.txt = self.split[2].strip()
            match = full_text[self.start:self.end].strip()
            if match != self.txt:
                #Correct start and end
                offset = min(self.start, max(5, len(self.txt)))
                substr = full_text[self.start-offset:]
                ix = substr.index(self.txt)
                self.start = self.start - offset + ix
                self.end = self.start + len(self.txt)
                assert full_text[self.start:self.end].strip() == self.txt
        assert self.start <= self.end, "Start index should be before the end"

    def clone(self):
        annotation = AnnotationBase("a\tb c")
        annotation.split = self.split[::]
        annotation.id = self.id[:]
        annotation.type = self.type[:]
        annotation.contents = self.contents[:]
        annotation.code = self.code[::]
        annotation.start = self.start
        annotation.end = self.end
        annotation.txt = self.txt[::]
        return annotation

class AttributeAnnotation(AnnotationBase):

    def __init__(self, line):
        AnnotationBase.__init__(self, line)
        self.child_annotations = []
        self.child_annotation_ids = self.contents[1:]

class RelationshipAnnotation(AnnotationBase):
    def __init__(self, line):
        AnnotationBase.__init__(self, line)
        pass

class EventAnnotation(AnnotationBase):
    def __init__(self, line, id2annotation):
        AnnotationBase.__init__(self, line)

        self.__dependencies__ = None
        self.id2annotation = id2annotation

    def dependencies(self):
        if self.__dependencies__ is None:
            self.__dependencies__ = []
            for annotation in self.contents:
                typ, id = annotation.split(":")
                if typ.strip().startswith("explicit"):
                    continue
                if typ[-1].isdigit():
                    typ = typ[:-1]

                dep = self.id2annotation[id]
                if type(dep) == CompoundTextAnnotation:
                    clonea = dep.first_part.clone()
                    clonea.code = typ

                    cloneb = dep.second_part.clone()
                    cloneb.code = typ
                    self.__dependencies__.append(clonea)
                    self.__dependencies__.append(cloneb)
                else:
                    clone = dep.clone()
                    clone.code = typ
                    self.__dependencies__.append(clone)
        return self.__dependencies__


class NoteAnnotation(AnnotationBase):
    def __init__(self, line):
        AnnotationBase.__init__(self, line)
        self.child_annotations = []
        self.child_annotation_ids = self.contents[1:]
        pass

class Essay(object):

    def __init__(self, full_path):

        self.full_path = full_path

        txt_file = full_path[:-4] + ".txt"

        assert full_path.endswith(".ann")
        assert os.path.exists(txt_file), "Missing associated text file for %s" % self.full_path

        with open(txt_file, "r+") as f:
            self.txt = f.read()

        self.tagged_words = []
        self.tagged_sentences = []
        self.id2annotation = {}
        self.split_sents = []

        with open(full_path, "r+") as f:
            lines = f.readlines()

        codes_start = defaultdict(set)
        codes_end = defaultdict(set)

        def process_text_annotation(annotation):
            if annotation.start == annotation.end:
                return False
            codes_start[annotation.start].add(annotation.code)
            codes_end[annotation.end].add(annotation.code)
            return True

        annotations_with_dependencies = []
        for line in lines:
            if len(line.strip()) == 0:
                continue
            first_char = line[0]
            if first_char == "T":
                if is_compound(line):
                    annotation = CompoundTextAnnotation(line, self.txt)
                    process_text_annotation(annotation.first_part)
                    process_text_annotation(annotation.second_part)

                    """
                    print ""
                    print line.strip()
                    print annotation.txt
                    print "First:  ", self.txt[annotation.first_part.start:annotation.first_part.end]
                    print "Second: ", self.txt[annotation.second_part.start:annotation.second_part.end]
                    print annotation.first_part.start, annotation.first_part.end, " ",
                    print annotation.second_part.start, annotation.second_part.end
                    """
                else:
                    annotation = TextAnnotation(line, self.txt)
                    #Bad annotation, ignore
                    if not process_text_annotation(annotation):
                        continue

            elif first_char == "A":
                annotation = AttributeAnnotation(line)
                for id in annotation.child_annotation_ids:
                    annotation.child_annotations.append(self.id2annotation[id])
            elif first_char == "R":
                annotation = RelationshipAnnotation(line)
            elif first_char == "E":
                annotation = EventAnnotation(line, self.id2annotation)
                annotations_with_dependencies.append(annotation)
            elif first_char == "#":
                annotation = NoteAnnotation(line)
                for id in annotation.child_annotation_ids:
                    annotation.child_annotations.append(self.id2annotation[id])
            else:
                raise Exception("Unknown annotation type")

            self.id2annotation[annotation.id] = annotation

        for annotation in annotations_with_dependencies:
            for dependency in annotation.dependencies():
                process_text_annotation(dependency)

        codes = set()
        current_word = ""
        current_sentence = []

        def add_pair(current_word, current_sentence, codes, ch, ix):
            if current_word.strip() != "":
                pair = (current_word, codes)
                current_sentence.append(pair)
                self.tagged_words.append(pair)
            if ch.strip() != "" and ch != "/":
                if ix in codes_start:
                    pair2 = (ch, codes_start[ix])
                else:
                    pair2 = (ch, set())
                current_sentence.append(pair2)
                self.tagged_words.append(pair2)

        def onlyascii(s):
            out = ""
            for char in s:
                if ord(char) > 127:
                    out += ""
                else:
                    out += char
            return out

        def add_sentence(sentence, str_sent):

            sents = filter(lambda s: len(s) > 1, sent_tokenize(onlyascii(str_sent.strip())))
            if len(sents) > 1:
                # filter to # of full sentences, and we should get at least this many out
                expected_min_sents = len([s for s in sents if s.strip().split(" ") > 1])

                unique_wds = set(map(lambda s: s.lower(), zip(*sentence)[0]))

                processed = []
                partitions = []
                for i, sent in enumerate(sents):
                    # last but one only
                    if i < (len(sents) - 1):
                        last = sent.split(" ")[-1]
                        if last.lower() == "temp.":
                            expected_min_sents -= 1
                            continue
                        if last[-1] in {".", "?", "?", "\n"}:
                            last = last[-1]
                        elif last.lower() not in unique_wds:
                            last = last[-1]

                        assert last.lower() in unique_wds

                        first = sents[i+1].split()[0]
                        if first.lower() not in unique_wds:
                            if not first[-1].isalnum():
                                first = first[:-1]
                            elif not first[0].isalnum():
                                first = first[0]

                        assert first.lower() in unique_wds

                        partitions.append((last, first))

                if len(partitions) == 0:
                    # handle the temp. error from above (one sentence where there is a temp. Increase)
                    self.tagged_sentences.append(sentence)
                    return
                current = []
                for j in range(0, len(sentence)-1):
                    wd,tag = sentence[j]
                    current.append((wd, tag))
                    if len(partitions) > 0:
                        last, first = partitions[0]
                        if last == wd:
                            nextWd, nextTg = sentence[j + 1]
                            if first.startswith(nextWd):
                                self.tagged_sentences.append(current)
                                processed.append(zip(*current)[0])
                                current = []
                                partitions = partitions[1:]
                current.append(sentence[-1])
                self.tagged_sentences.append(current)
                processed.append(zip(*current)[0])
                assert len(processed) >= max(2,expected_min_sents)
                self.split_sents.append(processed)
            else:
                self.tagged_sentences.append(sentence)

        str_sent = ""
        for ix, ch in enumerate(self.txt):

            if ch.isalnum() or ch == "'":
                current_word += ch
            else:
                add_pair(current_word, current_sentence, codes.copy(), ch, ix)
                str_sent += current_word + ch
                # don't always split on periods, as not all periods terminate sentences (e.g. acronyms)
                if len(current_sentence) > 0 and \
                        ((ch in {"\n", "!", "?"}) or
                         (ch == "/" and ix > 0 and self.txt[ix-1] in {"\n", ".", "!", "?"})
                        ):
                    add_sentence(current_sentence, str_sent)
                    current_sentence = []
                    str_sent = ""
                current_word = ""

            if ix in codes_start:
                codes.update(codes_start[ix])
            if ix in codes_end:
                codes.difference_update(codes_end[ix])

        # add any remaining
        add_pair(current_word, current_sentence, codes.copy(), "", ix)
        if len(current_sentence) > 0:
            self.tagged_sentences.append(current_sentence)

def load_bratt_essays(directory = None):
    bratt_root_folder = directory
    if not bratt_root_folder:
        settings = Settings.Settings()
        bratt_root_folder = settings.data_directory + "CoralBleaching/BrattData/Merged/"

    files = find_files(bratt_root_folder, "\.ann$", remove_empty=True)
    print len(files), "files found"

    essays = []
    for f in files:
        #try:
        essay = Essay(f)
        essays.append(essay)
        #except Exception, e:
        #    print "Error processing file: ", e.message, f

    print "%s essays processed" % str(len(essays))
    return essays


if __name__ == "__main__":

    essays = load_bratt_essays()

    for essay in essays:
        if essay.split_sents:
            print essay.full_path
            for splt_sent in essay.split_sents:
                for splt in splt_sent:
                    print " ".join(splt)
                print ""

    print "Done"