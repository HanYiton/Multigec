To fix an issue in the assignment of texts in the train split to one of the four domains, the following files were replaced by corrected versions. They are based on the more recent version of the source corpus (GECCC, dated 2022-09-28, http://hdl.handle.net/11234/1-4861), namely on the paragraph.input, paragraph.gold and pragraph.meta datasets in the data/train/ folder.

NatForm:
cs-natform-orig-train.md
cs-natform-ref1-train.md

NatWebInf:
cs-natwebinf-orig-train.md
cs-natwebinf-ref1-train.md
cs-natwebinf-ref2-train.md

Romani:
cs-romani-orig-train.md
cs-romani-ref1-train.md
cs-romani-ref2-train.md
metadata.yaml

SecLearn:
cs-seclearn-orig-train.md
cs-seclearn-ref1-train.md
cs-seclearn-ref2-train.md
metadata.yaml

The data/meta.tsv file was correct in the previous version.
The statistics had to be changed for the Romani and SecLearn subcorpora as follows:

Romani:
reference_essays_2: 
  n_essays: 
    total: 247 (was 1025)
    train: 0 (was 778)

SecLearn:
reference_essays_2: 
  n_essays: 
    total: 450 (was 604)
    train: 183 (was 337)

The number of (original) texts / words in the four subcorpora is as follows:

NatForm: 391 / 74K
NatWebInf: 6157 / 139K
Romani: 3599 / 282K
SecLearn: 2407 / 344K