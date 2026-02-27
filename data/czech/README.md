# GECCC dataset modified for MultiGEC 2025

Grammar Error Correction Corpus for Czech (GECCC, available from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4639) consists of 83 058 sentences in 12 564 texts, and covers four diverse domains, including essays written by native students, informal website texts, essays written by Romani ethnic minority children and teenagers and essays written by non-native speakers. For MultiGEC, the corpus is split into four parts according to the domain, corresponding to the four folders: NatForm, NatWebInf, Romani and SecLearn. The description below applies to all the four subcorpora.

# files from GECCC used:

- *paragraph.input*, *paragraph.gold* – aligned detokenized paragraphs: input with original noisy paragraphs and gold with one or two of their (alternative) corrections (tab-separated)
- *paragraph.meta* – each line in this file specifies a pointer to the meta-description file meta.tsv that contains metadata such as the user domain of the sentence writer or the original document the sentence belongs to. Specifically, i-th line of paragraph.meta stores the pointer for the i-th line of paragraph.input and paragraph.gold. Note that paragraphs are in their files in original order and that paragraphs can be joined into documents using information from the paragraph.meta file (all paragraphs belonging to the same document have the same ID)
- *meta.tsv* – see below in "original metadata"

# steps:

- for each train, dev and test folder
	- paragraph.meta was column-merged (paste in UNIX) with paragraph.input and paragraph.gold, separately
	- any alternative annotations in paragraph.gold were moved to a separate file
	- paragraphs and texts in input, gold and alternative gold were reformatted to match the MultiGEC 2025 data format, using the reformat_text.py script
   	- all text files were split according to the domain of the text into four folders: NatForm, NatWebInf, Romani and SecLearn, preserving their names and using the split_by_domain.py script and the domains.txt file, extracted from meta.tsv, listing the assignment of text IDs to the domain  
	
- added: LICENSE, metadata_template.yaml, README.md (this document), meta.tsv, reformat_text.py, split_by_domain.py, domains.txt   

# original metadata

The meta-description file *meta.tsv* is a tab-separated file that stores meta-information about each record. Specifically, it has these columns:
- Filename – ID of the record that is used for linking from the .meta files
- Domain – one of 4 user domains (*Natives Formal*, *Natives Web Informal*, *Romani* and *Second Learners*)
- Age – either a 1- or 2-digit string specifying the age of the writer, or 4-digit string indicating writer age range (e.g. 1519 stays for age between 15 and 19) or *Unknown*
- Sex – either *F* (female), *M* (male) or *Unknown*
- isSlavic – either *Yes* if the writer comes from a Slavic language group, *No* if he does not or Unknown
- Source – name of the corpus the original noisy document comes from.
- OriginalName – ID of the document storing the original sentences regarding the Source. The main purpose of this is to allow linking the GECCC dataset records to their original records that sometimes provide more metadata than Age, Sex and isSlavic that we chose.
- NewlyAnnotated – whether the document is annotated with new annotations or the annotation were not newly created and come from the original source
