if [ $1 = 'dbpedia' ]
then
        echo "############ Building DBpedia index ..."
        python3 -m nordlys.core.data.dbpedia.indexer_dbpedia data/config/index_dbpedia_2015_10.config.json
elif [ $1 == 'types' ]
then
        echo "############ Building DBpedia types index ..."
        python3 -m nordlys.core.data.dbpedia.indexer_dbpedia_types data/config/index_dbpedia_2015_10_types.config.json
elif [ $1 == 'entity_type' ]
then
        echo "############ Building DBpedia Hierarchical types for each entity Index ..."
        python3 -m nordlys.core.data.dbpedia.indexer_dbpedia_entity_type data/config/index_dbpedia_2015_10_entity_hierarchical_type.config.json
elif [ $1 == 'dbpedia_uri' ]
then
        echo "############ Building DBpedia URI-only index ..."
        python3 -m nordlys.core.data.dbpedia.indexer_dbpedia_uri data/config/index_dbpedia_2015_10_uri.config.json
fi
