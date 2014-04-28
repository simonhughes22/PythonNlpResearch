
import os
from FindFiles import find_files
from collections import defaultdict

class AnnotationBase(object):
    def __init__(self, line):
        self.split = line.strip().split("\t")
        self.id = self.split[0]
        self.type = self.id[0]

        self.contents = self.split[1].replace("  ", " ").split(" ")
        self.code = self.contents[0]

class TextAnnotation(AnnotationBase):
    def __init__(self, line):
        """ Basic essay element, has
                txt
                id
                start
                end
        """
        AnnotationBase.__init__(self, line)
        self.start = int(self.contents[1])
        self.end = int(self.contents[2])
        assert self.start <= self.end, "Start index should be before the end"
        if len(self.split) > 2:
            self.txt = self.split[2]
        else:
            self.txt = ""
    pass

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
    def __init__(self, line):
        AnnotationBase.__init__(self, line)
        pass

class NoteAnnotation(AnnotationBase):
    def __init__(self, line):
        AnnotationBase.__init__(self, line)
        self.child_annotations = []
        self.child_annotation_ids = self.contents[1:]
        pass

class Essay(object):

    SENTENCE_TERM = set([".", "!", "?"])

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

        with open(full_path, "r+") as f:
            lines = f.readlines()

        codes_start = defaultdict(set)
        codes_end = defaultdict(set)
        for line in lines:
            if len(line.strip()) == 0:
                continue
            first_char = line[0]
            if first_char == "T":
                annotation = TextAnnotation(line)
                #Bad annotation, ignore
                if annotation.start == annotation.end:
                    continue
                codes_start[annotation.start].add(annotation.code)
                codes_end[annotation.end].add(annotation.code)
            elif first_char == "A":
                annotation = AttributeAnnotation(line)
                for id in annotation.child_annotation_ids:
                    annotation.child_annotations.append(self.id2annotation[id])
            elif first_char == "R":
                annotation = RelationshipAnnotation(line)
            elif first_char == "E":
                annotation = EventAnnotation(line)
            elif first_char == "#":
                annotation = NoteAnnotation(line)
                for id in annotation.child_annotation_ids:
                    annotation.child_annotations.append(self.id2annotation[id])
            else:
                raise Exception("Unknown annotation type")

            self.id2annotation[annotation.id] = annotation

        codes = set()
        current_word = ""
        current_sentence = []

        def add_pair(current_word, current_sentence, codes, ch):
            if current_word.strip() != "":
                pair = (current_word.lower(), codes)
                current_sentence.append(pair)
                self.tagged_words.append(pair)
            if ch.strip() != "":
                pair2 = (ch, set())
                current_sentence.append(pair2)
                self.tagged_words.append(pair2)

        for ix, ch in enumerate(self.txt):

            if ch.isalnum() or ch == "'":
                current_word += ch
            else:
                add_pair(current_word, current_sentence, codes.copy(), ch)
                if ch in self.SENTENCE_TERM:
                    self.tagged_sentences.append(current_sentence)
                    current_sentence = []
                current_word = ""

            if ix in codes_start:
                codes.update(codes_start[ix])
            if ix in codes_end:
                codes.difference_update(codes_end[ix])

        # add any remaining
        add_pair(current_word, current_sentence, codes.copy(), "")

def load_bratt_essays():
    import Settings

    settings = Settings.Settings()
    bratt_root_folder = settings.data_directory + "CoralBleaching/BrattData/Merged/"

    files = find_files(bratt_root_folder, "\.ann$", remove_empty=True)
    print len(files), "files found"

    essays = map(Essay, files)

    pass

if __name__ == "__main__":

    load_bratt_essays()





