def get_utf8(text):
    # text = str(text)
    encode = text.encode()
    decode = str(encode, 'unicode_escape')
    # text = str(decode)
    return decode

def truncateUTF8length(unicodeStr, maxsize):
        # This method can be used to truncate the length of a given unicode
        # string such that the corresponding utf-8 string won't exceed
        # maxsize bytes. It will take care of multi-byte utf-8 chars intersecting
        # with the maxsize limit: either the whole char fits or it will be
        # truncated completely. Make sure that unicodeStr is in Unicode
        # Normalization Form C (NFC), else strange things can happen as
        # mentioned in the examples below.
        # Returns a unicode string, so if you need it encoded as utf-8, call
        # .decode("utf-8") after calling this method.
        # # >>> truncateUTF8lengthIfNecessary(u"ö", 2) == (u"ö", False)
        # # True
        # # >>> truncateUTF8length(u"ö", 1) == u""
        # # True
        # # >>> u'u1ebf'.encode('utf-8') == 'xe1xbaxbf'
        # # True
        # # >>> truncateUTF8length(u'hiu1ebf', 2) == u"hi"
        # # True
        # # >>> truncateUTF8lengthIfNecessary(u'hiu1ebf', 3) == (u"hi", True)
        # # True
        # # >>> truncateUTF8length(u'hiu1ebf', 4) == u"hi"
        # # True
        # # >>> truncateUTF8length(u'hiu1ebf', 5) == u"hiu1ebf"
        # True
        #
        # Make sure the unicodeStr is in NFC (see unicodedata.normalize("NFC", ...) ).
        # The following would not be true, as e and u'u0301' would be seperate
        # unicode chars. This could be handled with unicodedata.combining
        # and a loop deleting chars from the end until after the first non
        # combining char, but this is _not_ done here!
        # #>>> u'eu0301'.encode('utf-8') == 'exccx81'
        # #True
        # #>>> truncateUTF8length(u'eu0301', 0) == u"" # not in NFC (u'xe9'), but in NFD
        # #True
        # #>>> truncateUTF8length(u'eu0301', 1) == u"" #decodes to utf-8:
        # #True
        # #>>> truncateUTF8length(u'eu0301', 2) == u""
        # #True
        # #>>> truncateUTF8length(u'eu0301', 3) == u"eu0301"
        # #True
        return str(unicodeStr.encode("utf-8")[:maxsize], "utf-8", errors="ignore")