def get_utf8(text):
    # text = str(text)
    encode = text.encode()
    decode = str(encode, 'unicode_escape')
    # text = str(decode)
    return decode