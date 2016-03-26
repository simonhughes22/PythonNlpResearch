# coding=utf-8
# Collapse tag variants
def tag_transformer(tag):
    # Collapse PXX to XX
    #if tag[0] == "P":
    #    return tag[1:]
    #if ":P" in tag:
    #    tag = tag.replace(":P", ":")
    # Collapse XX.Y to XX
    if "." in tag:
        tag = tag[:tag.index(".")]
    # Collapse XX_Y to XX
    if "_" in tag:
        tag = tag[:tag.index("_")]
    return tag
