# KR-EAR
Code of IJCAI2016: "Knowledge Representation Learning with Entities, Attributes and Relations"

New: Add PTransE (EMNLP 2015) code!






===== DATA =====

I provide FB24k  datasets used for the task of knowledge base completion with the input format of my code in data.zip.



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



===== CODE =====

In the folder KR-EAR(TransE)/, KR-EAR(TransR)/.



===== COMPILE =====

Just type make in the folder ./



== TRAINING ==

For training, You need follow the step below:





TransE:

	call the program Train_TransE in folder TransE/
	
TransH:
	call the program Train_TransH in folder TransH/

TransR:

	1:	Train the unif method of TransE as initialization.

	2:  call the program Train_TransR in folder TransR/

CTransR:

	1:	Train the unif method of TransR as initialization.

	2:  run the bash run.sh with relation number in folder cluster/ to cluster the triples in the trainning data.

		i.e.

			bash run.sh 10

	3:  call the program Train_cTransR in folder CTransR/

You can also change the parameters when running Train_TransE, Train_TransR, Train_CTransR.

-size : the embedding size k, d

-rate : learing rate

-method: 0 - unif, 1 - bern



== TESTING ==

For testing, You need follow the step below:


TransR:

	call the program Test_TransR with method as parameter in folder TransR/

CTransR:

	call the program Test_CTransR with method as parameter in folder CTransR/

It will evaluate on test.txt and report mean rank and Hits@10




==CITE==

If you use the code, you should cite the following paper:

Yankai Lin, Zhiyuan Liu, Maosong Sun. Knowledge Representation Learning with Entities, Attributes and Relations. International Joint Conference on Artificial Intelligence (IJCAI 2016).