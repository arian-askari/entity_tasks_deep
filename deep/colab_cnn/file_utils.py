import json
import csv

def create_file(path, data):
    f = open(path, 'w')
    f.write(data)
    f.close()


def append_file(path, data):
    f = open(path, 'a')
    f.write(data)
    f.close()

def write_file(path, data, force=False):
    f= None

    if force == False:
        f = open(path, 'a')
    else:
        f = open(path, 'w')

    f.write(data)
    f.close()

def wirte_json_file(path, data, force=False):
    if force == False:
        json.dump(data, fp=open(path, 'a'))
    else:
        json.dump(data, fp=open(path, 'w'))