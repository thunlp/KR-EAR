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

===== RUN =====

You can also change the parameters when running Train_TransE, Train_TransR, Train_CTransR.

-n : the embedding size of entities, relations

-m :  the embedding size of values 

-margin: the margin length








==CITE==

If you use the code, you should cite the following paper:

Yankai Lin, Zhiyuan Liu, Maosong Sun. Knowledge Representation Learning with Entities, Attributes and Relations. International Joint Conference on Artificial Intelligence (IJCAI 2016).