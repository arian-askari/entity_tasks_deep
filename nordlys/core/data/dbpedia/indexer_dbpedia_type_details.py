"""
DBpedia Types Indexer
=====================

Build an index of DBpedia types from entity abstracts.

:Authors: Krisztian Balog, Dario Garigliotti
"""

import os, json
import argparse
from random import sample
from math import floor

from rdflib.plugins.parsers.ntriples import NTriplesParser
from rdflib.plugins.parsers.ntriples import ParseError
from rdflib.term import URIRef
from nordlys.core.storage.parser.nt_parser import Triple
from nordlys.core.storage.parser.uri_prefix import URIPrefix
from nordlys.core.utils.file_utils import FileUtils
from nordlys.core.retrieval.elastic import Elastic
from nordlys.config import DATA_DIR
from nordlys.config import PLOGGER
import utils.file_utils as file_utils

# -------
# Constants

# About index
ID_KEY = "id"  # not used
CONTENT_KEY = "content"

# Distinguished strings
ABSTRACTS_SEPARATOR = "\n"
DBO_PREFIX = "<dbo:"

# Sizes
BULK_LEN = 1
MAX_BULKING_DOC_SIZE = 20000000  # max doc len when bulking, in chars (i.e., 20MB)
AVG_SHORT_ABSTRACT_LEN = 216  # according to all entities_name appearing in DBpedia-2015-10 entity-to-type mapping


# -------
# Indexer class

class DictTypesDetail(object):

    def __init__(self, config):
        self.__config = config
        self.__type2entity_file = config["type2entity_file"]
        self.__types_details_dict = config["types_details_dict"]
        self.__types_details_csv = config["types_details_csv"]
        self.__types_details_light_csv = config["types_details_light_csv"]

    def build_dict(self, force=False):
        """Builds the dict for types details.

        :param force: True iff it is required to overwrite the index (i.e. by creating it by force); False by default.
        :type force: bool
        :return:
        """
        types_dict = dict()
        prefix = URIPrefix()

        # process type2entity file
        last_type = None
        entities_name = []
        delimeter = "\t"
        csv_str = "type\tcnt_entities\tentities\n"
        csv_str_light = "type\tcnt_entities\n"
        types_cnt = 1
        with FileUtils.open_file_by_type(self.__type2entity_file) as f:
            for line in f:
                line = line.decode()  # o.w. line is made of bytes
                if not line.startswith("<"):  # bad-formed lines in dataset
                    continue
                subj, obj = line.rstrip().split()

                type = prefix.get_prefixed(subj)  # subject prefixed
                entity = prefix.get_prefixed(obj)

                # use only DBpedia Ontology native types (no bibo, foaf, schema, etc.)
                if not type.startswith(DBO_PREFIX):
                    continue

                if last_type is not None and type != last_type:
                    print(types_cnt, "-", last_type)
                    entities_cnt = len(entities_name)
                    types_dict[last_type] = (entities_cnt, entities_name)
                    csv_str += last_type + delimeter + str(entities_cnt) + delimeter + str(entities_name) + "\n"
                    csv_str_light += last_type + delimeter + str(entities_cnt) + "\n"
                    types_cnt+=1
                    entities_name = [] #important to reset it:)
                    # break # for test everything is ok:)

                last_type = type
                entities_name.append(entity)

        #save types_dict to file as json :)
        file_utils.wirte_json_file(self.__types_details_dict, types_dict, force=force)
        file_utils.write_file(self.__types_details_csv, csv_str, force=force)
        file_utils.write_file(self.__types_details_light_csv, csv_str_light, force=force)


# -------
# Main script

def main(args):
    config = FileUtils.load_config(args.config)

    type2entity_file = os.path.expanduser(os.path.join(config.get("type2entity_file", "")))
    if (not os.path.isfile(type2entity_file)):
        exit(1)

    indexer = DictTypesDetail(config)
    indexer.build_dict(force=True)
    PLOGGER.info("Types Detail Dict build, have a nice day:) ")


def arg_parser(description=None):
    """Returns a 2-uple with the parsed paths to the type-to-entity and entity abstracts source files."""
    default_description = "It indexes DBpedia types storing the abstracts of their respective assigning entities_name."
    description_str = description if description else default_description
    parser = argparse.ArgumentParser(description=description_str)

    parser.add_argument("config", help="config file", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(arg_parser())
