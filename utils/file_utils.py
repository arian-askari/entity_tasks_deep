def create_file(path, data):
    f = open(path, 'w')
    f.write(data)
    f.close()


def append_file(path, data):
    f = open(path, 'a')
    f.write(data)
    f.close()