Entity Linking
================

Identifying named entities in queries and linking them to the corresponding entry in the knowledge base is known as the task of entity linking in queries (ELQ). Given a query q, return one or multiple interpretations of the query, each interpretation consists of a set of mention-entity pairs.

Entity retrieval is a core building block of semantic search.  Given a search query, entity retrieval is the task of returning a ranked list of entities from an underlying knowledge base.

The following ELQ methods are implemented in Nordlys:

- **CMNS**:  Thee baseline method that performs entity linking based on the overall popularity of entities as link targets, i.e., the commonness feature [:ref:`Hasibi et al., 2015 <ref_Hasibi2015>`].

- **LTR-greedy**: The recommended method (with respect to both efficiency and effectiveness) by Hasibi et al. [:ref:`Hasibi et al., 2016 <ref_Hasibi2017>`], which employs a learning-to-rank model with various textual and semantic similarity features. Please note that the implemented method in Nordlys is slightly different from the one presented in [:ref:`Hasibi et al., 2016 <ref_Hasibi2017>`] (i.e. the features and index). The corresponding files for this method are under `data/el`. Specifically:

  - ``config_ltr.json`` holds the config file
  - ``model.txt`` is the trained model 

Usage
-----

- :doc:`Command line usage <el_usage>`
- :ref:`API usage <api_el>`



References
----------

.. _ref_Hasibi2015:

- Faegheh Hasibi, Krisztian Balog, Svein Erik Bratsberg. 2015. **Entity Linking in Queries: Tasks and Evaluation**. In: *ACM SIGIR International Conference on the Theory of Information Retrieval (ICTIR ’15)*. [`PDF <http://hasibi.com/files/ictir2015-elq.pdf>`_]

.. _ref_Hasibi2017:

- Faegheh Hasibi, Krisztian Balog, Svein Erik Bratsberg. 2017. **Entity Linking in Queries: Efficiency vs. Effectiveness**. In: *39th European Conference on Information Retrieval (ECIR ’17)*. [`PDF <http://hasibi.com/files/ecir2017-elq.pdf>`_]

.. _ref_Ogilvie2003:

- Paul Ogilvie and Jamie Callan. 2003. **Combining Document Representations for Known-Item Search**. In: *26th annual international ACM SIGIR conference on Research and development in information retrieval (SIGIR '03)*.
