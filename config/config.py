from configparser import ConfigParser

import os


class Config:
    def __init__(self):
        self.cf = ConfigParser()
        self.cf.read(os.path.dirname(os.path.realpath(__file__))+'/Config.ini')