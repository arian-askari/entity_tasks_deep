"""
DBpedia Types Indexer
=====================

Build an index of DBpedia Hierarchical types for each entity.

:Authors: Arian Askari
"""

import os
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
from nordlys.core.storage.mongo import Mongo
import utils.utf8_helper as utf8_helper

# -------
# Constants

# About index
ID_KEY = "id"  # not used
TYPE_KEY = "type_keys"

# Distinguished strings
ABSTRACTS_SEPARATOR = "\n"
DBO_PREFIX = "<dbo:"

# Sizes
BULK_LEN = 4000


# -------
# Indexer class

'''
echo "############ Building DBpedia Hierarchical types for each entity Index ..."
python3 -m nordlys.core.data.dbpedia.indexer_dbpedia_entity_type data/config/index_dbpedia_2015_10_entity_hierarchical_type.config.json

"index_name": "dbpedia_2015_10_entity_hierarchical_type",
"entity2type_file": "data/raw-data/dbpedia-2015-10/type2entity-mapping/dbpedia-2015-10-entity_to_type.csv.bz2"        
'''

class IndexerDBpediaTypes(object):
    __DOC_TYPE = "doc"  # we don't make use of types
    __MAPPINGS = {
        # ID_KEY: Elastic.notanalyzed_field(),
        TYPE_KEY: Elastic.analyzed_field(),
    }

    def __init__(self, config):
        self.__elastic = None
        self.__config = config
        self.__model = config.get("model", Elastic.BM25)
        self.__index_name = config["index_name"]
        self.__entity2type_file = config["entity2type_file"]

    @property
    def name(self):
        return self.__index_name

    def __make_type_doc(self, types):
        return  {TYPE_KEY: types}

    def build_index(self, force=False):
        """Builds the index.
        :param force: True iff it is required to overwrite the index (i.e. by creating it by force); False by default.
        :type force: bool
        :return:
        """
        self.__elastic = Elastic(self.__index_name)
        self.__elastic.create_index(mappings=self.__MAPPINGS, force=force)


        prefix = URIPrefix()

        # For indexing types in bulk
        entities_bulk = {}  # dict from type id to type(=doc)

        # process type2entity file
        last_entity = None
        types = []
        lines_counter = 0
        entities_counter = 0
        with FileUtils.open_file_by_type(self.__entity2type_file) as f:
            for line in f:
                line = line.decode()  # o.w. line is made of bytes
                if not line.startswith("<"):  # bad-formed lines in dataset
                    continue

                obj = subj = type = entity = None
                if len(line.rstrip().split())==2:

                    obj, subj = line.rstrip().split()

                    type = prefix.get_prefixed(subj)  # subject prefixed
                    entity = prefix.get_prefixed(obj)

                    # use only DBpedia Ontology native types (no bibo, foaf, schema, etc.)
                    if not type.startswith(DBO_PREFIX):
                        continue

                if (last_entity is not None and entity != last_entity) or (len(line.rstrip().split())<2):
                    # moving to new entity, so:
                    # create a doc for this entity, with all the types, and store it in a bulk
                    entities_counter += 1

                    last_entity = Mongo.unescape(last_entity)
                    last_entity = utf8_helper.truncateUTF8length(last_entity, 511)  # id e document mishavad entity name dar inja. max len rooye 511 byte mizaram! rooye retrieve ham badan ino dar nazar migiram!

                    entities_bulk[last_entity] = self.__make_type_doc(types) # last_entity is doc id in index
                    types = []  # important to reset it

                    if entities_counter % BULK_LEN == 0:  # index the bulk of BULK_LEN docs
                        self.__elastic.add_docs_bulk(entities_bulk)
                        entities_bulk.clear()  # NOTE: important to reset it
                        PLOGGER.info("\tIndexing a bulk of {} docs (entities)... OK. "
                                     "{} entities already indexed.".format(BULK_LEN, entities_counter))

                last_entity = entity
                types.append(type)

                lines_counter += 1
                if lines_counter % 10000 == 0:
                    # PLOGGER.info("\t{}K lines processed".format(lines_counter // 1000))
                    pass
                pass

        # index the last type
        entities_counter += 1

        PLOGGER.info("\n\tFound {}-th (last) entitiy: {}\t\t with # of types: {}".format(entities_counter, last_entity,
                                                                                         len(types)))

        entities_bulk[last_entity] = self.__make_type_doc(types)
        self.__elastic.add_docs_bulk(entities_bulk)  # a tiny bulk :)
        # no need to reset neither types nor entities_bulk :P
        # PLOGGER.info("Indexing a bulk of {} docs (types)... OK.".format(BULK_LEN))

        PLOGGER.info("\n### Indexing all {} found docs (entities)... Done.".format(entities_counter))


# -------
# Main script
def main(args):
    config = FileUtils.load_config(args.config)
    type2entity_file = os.path.expanduser(os.path.join(config.get("entity2type_file", "")))
    if not os.path.isfile(type2entity_file):
        exit(1)

    indexer = IndexerDBpediaTypes(config)
    indexer.build_index(force=True)
    PLOGGER.info("Index build: <{}>".format(indexer.name))


def arg_parser(description=None):
    """Returns a 1-uple with the parsed paths to the entity-to-type file."""
    default_description = "It indexes DBpedia types storing the abstracts of their respective assigning entities."
    description_str = description if description else default_description
    parser = argparse.ArgumentParser(description=description_str)

    parser.add_argument("config", help="config file", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # main()
    main(arg_parser())
