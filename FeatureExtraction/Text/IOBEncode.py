__author__ = 'simon.hughes'

def IOBE_encode(tags,
                o_fn=lambda t: "O",
                b_fn=lambda t: "B" + str(t),
                i_fn=lambda t: "I" + str(t),
                e_fn=lambda t: "E" + str(t)):
    """
    tags:   :   list of str
                    tags to convert
    o_fn    :   fn(str) : str
                    other encoding fn
    b_fn    :   fn(str) : str
                    begin encoding fn
    i_fn    :   fn(str) : str
                    inside encoding fn
    e_fn    :   fn(str) : str
                    other encoding fn
    returns :   list of any

    Convert a set of tags to IOB encoded tags
    """

    last = None
    iobe_encoded = []
    for i, tag in enumerate(tags):
        if i > 0:
            last = tags[i-1]
            if last == "":
                last = None
        if i < len(tags) -1:
            next = tags[i+1]
        else:
            next = None
        if next == "":
            next = None

        if tag is None or tag == "":
            iobe_encoded.append(o_fn(tag))
        elif last is None:
            iobe_encoded.append(b_fn(tag))
        elif next is None:
            iobe_encoded.append(e_fn(tag))
        else:
            iobe_encoded.append(i_fn(tag))
        last = tag
    return iobe_encoded

if __name__ == "__main__":

    tags = [None, None, "tag", "tag", "tag", "tag", "", "tag",
            None, "tag", "tag", None, "tag", "tag", "tag",
            None, None, "tag"]
    iob_encoded = IOBE_encode(tags)

    for a,b in zip(tags, iob_encoded):
        print (str(a) + "|").ljust(10), str(b).rjust(10)
