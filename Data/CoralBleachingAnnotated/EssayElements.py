from IterableFP import compact

def removeNonAscii(s):
    return str("".join(filter(lambda x: ord(x)<128, s)))

def get_val(element, key, fn = lambda v: removeNonAscii(v)):
    return fn(element.attributes[key].nodeValue)

#closure fn over get_val for int. vals
get_int_val = lambda v, key: get_val(v, key, fn= int)

class EssayElement(object):
    def __init__(self, xmlDomElement):
        """ Basic essay element, has
                txt
                type
                id
                start
                end
        """
        if "type" in xmlDomElement.attributes.keys():
            self.type = get_val(xmlDomElement, "type").lower()
        else:
            # Compiled Score and KeyWord Text do not have a type
            self.type = None

        self.element = xmlDomElement
        self.txt    = get_val(xmlDomElement, "text")
        self.id     = get_val(xmlDomElement, "id").lower() #enforce lc in case of noise

        """ Subtract one so we get indexes not character positions """
        self.start  = max(0, get_int_val(xmlDomElement, "start") -2)
        self.end    = max(0, get_int_val(xmlDomElement, "end") -2)

        assert self.start < self.end, "Start index should be before the end"

        self.parent = None

    def __repr__(self):
        return "Text: '%s'\nStart: %d End: %d" % (self.txt, self.start, self.end)

class CommentElement(EssayElement):
    def __init__(self, xmlDomElement):
        """ Like Essay Element, but with a comment tag
        """
        """ Super call """
        EssayElement.__init__(self, xmlDomElement)
        self.comment = get_val(xmlDomElement, "comment")
        self.has_inferences = "-" in self.comment
    pass

class Sentence(CommentElement):
    def __init__(self, xmlDomElement):
        CommentElement.__init__(self, xmlDomElement)
        self.types = ["codes", "bck", "nd"]

        # if type is codes
        if self.type == self.types[0] and self.comment.strip() != "":
            self.codes = self.extract_codes()
        else:
            self.codes = []

        self.prev = None
        self.next = None

    def extract_codes(self):
        code_str = self.comment.strip().lower()
        to_replace = ["-", ",", ";", "  "]
        for replace_str in to_replace:
            code_str = code_str.replace(replace_str, " ")
        code_str = code_str.replace("  ", " ")
        split_stripped = map(lambda s: s.strip(), code_str.split(" "))
        return compact(split_stripped)
    pass

class KeyWord(Sentence):
    pass

class CompiledScore(CommentElement):
    pass

class Causal(EssayElement):

    def __init__(self, xmlDomElement):
        EssayElement.__init__(self, xmlDomElement)
        self.types = ["single", "linked"]
        self.is_single = self.type == self.types[0]
        self.is_linked = self.type == self.types[1]

class Link(object):
    def __init__(self, xmlDomElement):

        self.element = xmlDomElement
        self.type = get_val(xmlDomElement, "type").lower()

        self.id = get_val(xmlDomElement, "id").lower() #enforce lc in case of noise
        self.to_id = get_val(xmlDomElement, "toID").lower() #enforce lc in case of noise
        self.from_id = get_val(xmlDomElement, "fromID").lower() #enforce lc in case of noise

        self.to_txt = get_val(xmlDomElement, "toText")
        self.from_txt = get_val(xmlDomElement, "fromText")

        self.types = ["causer", "result"]
        self.is_causer = self.type == self.types[0]
        self.is_result = self.type == self.types[1]

        self.from_element = None
        self.to_element = None

        self.prev = None
        self.next = None
        self.parent = None
    pass

class CausalLink(Link):
    pass

class Structure(EssayElement):
    pass

class StructureLINK(Link):
    pass

class Concept(EssayElement):
    def __init__(self, xmlDomElement):
        EssayElement.__init__(self, xmlDomElement)
        self.code = self.type
        if len(self.code) == 0:
            raise Exception("0 len code for Concept Node: %s" % self.id)

    def __repr__(self):
        return self.code

class VagueConcept(Concept):
    def __init__(self, xmlDomElement):
        Concept.__init__(self, xmlDomElement)
        if self.code[0] == "v":
            self.non_vague_code = self.code[1:]
        else:
            self.non_vague_code = self.code
    pass