# HAnDS framework

## Inputs

The framework requires three inputs:
1. [Linked text corpus (Wikipedia)](#wikipedia)
2. [Knowledge Base (Freebase)](#freebase)
3. [Type Hierarchy](#type_hierarchy)


## Wikipedia

We used the September 2016 dump of Wikipedia (enwiki-20160920-pages-articles-multistream.xml.bz2) for the experiments reported in the paper. The details of the preprocessing steps are mentioned below, which can be used to pre-process any other Wikipedia dump. The POS tagged preprocessed Wikipedia dump (20160920) is available at: (ADD URL)


### Preprocessing steps

1. Convert Wikipedia xml.bz2 dump to html format.

```
python ../wikiextractor/ -l -q -b 10M --filter_disambig_pages ../../datasets/Wikipedia_xml_dump/enwiki-20160920-pages-articles-multistream.xml.bz2 -o ../../datasets/Wikipedia_html_dump/

```

2. Convert Wikipedia html dump to a easy-to-use json format along with sentence segmentation. Please make sure that the Stanford CoreNLP server is listening on localhost port 9000.

```
python process_wiki_dump.py ../../datasets/Wikipedia_html_dump/ ../../datasets/processed_wikipedia/
```

## Freebase

Process Freebase and prepare a set of dictionaries to reduce frequent querying to Freebase SPARQL endpoint during the dataset generation process. The essential information needed from Freebase are Wikipedia article title to Freebase types and Wikipedia article redirect information. Make sure SPARQL endpoint is listening at http://localhost:8890/sparql/

1. Create a Wikipedia title to Freebase type and article redirect dictionary.

```
python wiki_title_to_freebase_mapping.py ../../datasets/all_wikipedia_titles/enwiki-latest-all-titles-in-ns0 ../../datasets/pickle_objects/freebase_preprocessed.pickle
```

2. Create a master dictionary which contains the above two generated information along with different surface names a Wikipedia article title is mentioned. The resultant dictionary will be used throughout the data creation process.

```
python title_to_surface_names.py ../../datasets/Wikipedia_html_dump/ ../../datasets/pickle_objects/freebase_preprocessed.pickle ../../datasets/pickle_objects/master_dictionary.pickle

```

## Type Hierarchy

We use two Type hierarchies. The first hierarchy is a variant of the FIGER type hierarchy with 118 types and the second type hierarchy is a variant of the TypeNet type hierarchy.

These hierarchies are available at: datasets/hierarchy/figer_types.map and datasets/hierarchy/typenet_types.txt

## Stages of HAnDS Framework

1. Stage 1

```
# Variant of the FIGER hierarchy.
python hands_stage_one.py ../../datasets/pickle_objects/master_dictionary.pickle ../../datasets/hierarchy/figer_types.map ../../datasets/processed_wikipedia/ ../../datasets/HAnDS_figer_types_stage_one/
# Variant of the TypeNET hierarchy.
python hands_stage_one.py ../../datasets/pickle_objects/master_dictionary.pickle ../../datasets/hierarchy/typenet_types.map ../../datasets/processed_wikipedia/ ../../datasets/HAnDS_typenet_types_stage_one/
```

2. Stage 2

```
# Variant of the FIGER hierarchy.
python hands_stage_two.py ../../datasets/pickle_objects/master_dictionary.pickle ../../datasets/hierarchy/figer_types.map  ../../datasets/HAnDS_figer_types_stage_one/ ../../datasets/HAnDS_figer_types_stage_one_state_two/
# Variant of the TypeNET hierarchy.
python hands_stage_two.py ../../datasets/pickle_objects/master_dictionary.pickle ../../datasets/hierarchy/typenet_types.map  ../../datasets/HAnDS_typenet_types_stage_one/ ../../datasets/HAnDS_typenet_types_stage_one_state_two/

```

3. Stage 3

3a) Convert document level annotations to sentence level annotations.
```
# Variant of the FIGER hierarchy.
python documents_to_sentences.py ../../datasets/HAnDS_figer_types_stage_one_state_two/ ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences/
# Variant of the TypeNET hierarchy.
python documents_to_sentences.py ../../datasets/HAnDS_typenet_types_stage_one_state_two/ ../../datasets/HAnDS_typenet_types_stage_one_state_two_sentences/
```
3b) Filter sentences
```
# Variant of the FIGER hierarchy.
python hands_stage_three.py ../../datasets/start_token_list.txt ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences/ ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_stage_three/
# Variant of the TypeNET hierarchy.
python hands_stage_three.py ../../datasets/start_token_list.txt ../../datasets/HAnDS_typenet_types_stage_one_state_two_sentences/ ../../datasets/HAnDS_typenet_types_stage_one_state_two_sentences_stage_three/
```

4. Post processing

This code will convert freebase types to the types present in the hierarchy based on a type mapping. This also removed sentences from the test set from the generated corpus.

```
# Variant of the FIGER hierarchy.
python hands_post_processing.py ../../datasets/pickle_objects/master_dictionary.pickle ../../datasets/hierarchy/figer_types.map ../../datasets/1k-WFB-g/1k-WFB-g_complete.json  ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_stage_three/ ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp/
# Variant of the TypeNET hierarchy.
python hands_post_processing.py ../../datasets/pickle_objects/master_dictionary.pickle ../../datasets/hierarchy/typenet_types.map ../../datasets/1k-WFB-g/1k-WFB-g_complete.json  ../../datasets/HAnDS_typenet_types_stage_one_state_two_sentences_stage_three/ ../../datasets/HAnDS_typenet_types_stage_one_state_two_sentences_stage_three_pp/
```

The post processed dataset is the final datasets generated using HAnDS framework.

5. Analysis of generated dataset

```
# Variant of the FIGER hierarchy.
python analyse_sentences.py complete_data ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp/ --label_distribution_file=../../stats/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp_labels.txt --pickle_path=../../stats/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp_label_reference.pickle
# Variant of the TypeNET hierarchy.
python analyse_sentences.py complete_data ../../datasets/HAnDS_typenet_types_stage_one_state_two_sentences_stage_three_pp/ --label_distribution_file=../../stats/HAnDS_typenet_types_stage_one_state_two_sentences_stage_three_pp_labels.txt --pickle_path=../../stats/HAnDS_typenet_types_stage_one_state_two_sentences_stage_three_pp_label_reference.pickle
```

6. Randomly sample 2 million sentences for model training. 
```
python random_sample_sentences.py ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp/ ../../stats/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp_label_reference.pickle ../../datasets/sampled_datasets/RS_fner_2.json.gz 2000000
```


7. (Optional) Get same randomly sampled sentences annotated with NDS approach. For this step, first Wikipedia processed datasets needs to converted into sentences, then it needs to be post processed.
```
python retain_sentences_from_sentences.py ../../datasets/processed_wikipedia_sentences_pp/ ../../datasets/sampled_datasets/RS_fner_2.json.gz ../../datasets/sampled_datasets/RS_NDS_from_2.json.gz
```

8. Get randomly sampled sentences annotated without stage three. For this step, stage 3b is skipped. 
a) Analyze sentences
```
python analyse_sentences.py complete_data ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_pp/ --label_distribution_file=../../stats/HAnDS_figer_types_stage_one_state_two_sentences_pp_labels.txt --pickle_path=../../stats/HAnDS_figer_types_stage_one_state_two_sentences_pp_label_reference.pickle
```
b) Randomly sample sentences
```
python random_sample_sentences.py ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_pp/ ../../stats/HAnDS_figer_types_stage_one_state_two_sentences_pp_label_reference.pickle ../../datasets/sampled_datasets/RS_only_stage_one_and_two_2.json.gz 2000000
```

### Qualitative analysis

```
# Variant of the FIGER hierarchy.
python qualitative_analysis.py ../../datasets/HAnDS_figer_types_stage_one_state_two_sentences_stage_three_pp/ ../../datasets/processed_wikipedia_sentences_pp/
# Variant of the TypeNET hierarchy.
python qualitative_analysis.py ../../datasets/HAnDS_typenet_types_stage_one_state_two_sentences_stage_three_pp/ ../../datasets/processed_wikipedia_sentences_pp/
```
