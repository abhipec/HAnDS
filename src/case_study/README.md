## Analysis for the case study: Entity Detection in the Fine Entity Typing Setting


1. The coverage analysis of the FIGER hierarchy is completely manual, and some details are written inside typenet_parsing.py file

2. The coverage analysis of the TypeNET hierarchy is based on the freebase mapping between Freebase to FIGER hierarchy and graph traversal, the results for this analysis can be obtained using the following code:
```
python typenet_parsing.py ../../datasets/case_study/typenet_structure.txt ../../datasets/case_study/figer_type_in_conll.map ../../datasets/case_study/typenet_not_presnet_in_conll.txt
```
