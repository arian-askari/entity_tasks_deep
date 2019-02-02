def create_file(path, data):
    f = open(path, 'w')
    f.write(data)
    f.close()