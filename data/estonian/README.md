# Estonian resources for MultiGEC-2025 

Estonian data contains resources from two L2 learner corpora: 

* EKI Estonian L2 learner parallel error corpus (Institute of Estonian Language) 

* Error-annotated subset of Estonian Interlanguage Corpus (Tallinn University) 

There is an additional external resource available, which mostly contains single sentences and their corrections:

* Parallel corpus of errors by learners of Estonian (University of Tartu) 


The first two data sets comprise full texts. The sentences have first been error annotated in the M2 format using adapted ERRANT classification and then converted into parallel sentences. The third data set includes text extracts of single or multiple sentences which have been annotated with possible correction(s). 


## EKI Estonian L2 learner parallel error corpus 

The corpus has been automatically generated from the Error-annotated Estonian L2 learner corpus. The materials for the error annotated corpus originate from the Estonian learner corpus EMMA, containing writings from Estonian learner assessment test (7th grade, 254 texts), basic school final exam (9th grade, 251 texts) and state exam (12th grade, 998 texts) provided by the Education and Youth Board. The error-annotation has been carried out by applying an ERRANT-M2 annotation scheme, adapted to the Estonian specifics. The corpus includes a total of 1,503 essays of A2, B1 and B2 level in divided into subsets of 1,202-150-151 for train, dev and test. The reference sets are differentiated on the basis of whether optional error corrections are taken into account or not. 


## Error-annotated subset of Estonian Interlanguage Corpus (EIC) 

This subcorpus consists of 258 texts and 3,721 sentences retrieved from the [Estonian Interlanguage Corpus](https://elle.tlu.ee/tools) compiled at the Tallinn University. The pseudonymized texts include narrative/descriptive and argumentative writings as well as informal and formal letters representing the proficiency levels A2–C1. Previously, they were manually error-tagged in the CoNLL-U format, indicating the error type, scope, and correction. The annotations have been converted to the M2 format using an adapted version of the ERRANT tagset, which can be found [here](https://github.com/tlu-dt-nlp/EstGEC-L2-Corpus). While the previous format was limited to one error annotation per sentence, up to two new annotations have been added in the process. The parallel version of the data set has been generated from the three annotation versions (if fewer corrections were proposed, they are repeated in the ref2 and/or ref3 files). The data has been split to the train, dev and test set as follows: 206-26-26 texts and 2,951-431-441 sentences. 

## University of Tartu Parallel Corpus of Errors by Learners of Estonian (UTL2)

This parallel corpus includes scrambled sentences from various texts with one correction per each sentence; less than 10% of the segments consist of more than one sentence. The original corpus in XML can be found [here](https://cl.ut.ee/korpused/veakorpus/index.php?lang=en). The data is in an XML format (with a DTD included, explaining the content) and the data includes also the level of the language learner, their mother tongue and type of text are specified in the files. The data consists of a total of 9,005 sentences and their corrections.
