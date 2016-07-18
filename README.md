# KR-EAR
Code of IJCAI2016: "Knowledge Representation Learning with Entities, Attributes and Relations"

Evaluation Results
==========

Evaluation results on entity prediction.

| Model      |     MeanRank(Raw) |   MeanRank(Filter)   |	Hit@10(Raw)	| Hit@10(Filter)|
| :-------- | --------:| :------: | :------: |:------: |
| TransE 			| 259		| 200		| 35.8	| 53.0 |
| TransH			| 282		| 224		| 33.9	| 50.2 |
| TransR 			| 260		| 200		| 37.0	| 56.1 |
| KR-EAR(TransE)	| 186		| 133		| 38.5	| 54.5 |	
| KR-EAR(TransR)	| 172		| 118		| 39.5	| 57.3 |


DATA
==========

We provide FB24k dataset used for the task knowledge base completion in data.zip, using the input format required by our codes. 


Datasets are needed in the folder data/ in the following format

Dataset contains six files:



+ train-rel.txt: training file of relations, format (e1, e2, rel).

+ test-rel.txt: test file of relations, same format as train-rel.txt.

+ train-attr.txt: training file of attributes, format (e1, val, attar).

+ test-attr.txt: test file of attributes, same format as train-attr.txt.

+ entity2id.txt: all entities and corresponding ids, one per line.

+ relation2id.txt: all relations and corresponding ids, one per line.

+ attribute2id.txt: all attributes and corresponding ids, one per line.

+ val2id.txt: : all values and corresponding ids, one per line.

+ attribute_val.txt: the value set of each attribute



COMPILE
==========

Just type make in the folder ./

RUN
==========

You need to  type the following command in each model folder:

For training:

	./main

For testing:

	./test

	./test_attr

You can also change the parameters when training.

-n : the embedding size of entities, relations

-m :  the embedding size of values 

-margin: the margin length








==CITE==

If you use the code, please kindly cite the following paper:

Yankai Lin, Zhiyuan Liu, Maosong Sun. Knowledge Representation Learning with Entities, Attributes and Relations. International Joint Conference on Artificial Intelligence (IJCAI 2016).
