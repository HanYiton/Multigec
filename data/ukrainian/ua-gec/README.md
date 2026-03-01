# UA-GEC: Grammatical Error Correction and Fluency Corpus for the Ukrainian Language

UA-GEC consists of texts written by a diverse set of contributors. There are
three sources for these texts:

- Essays written on a given prompt.
- Translations of text snippets into Ukrainian.
- Arbitrarily authored text donations.

The last category is the largest and most diverse. It contains social media
posts, essays, technical documents, advertisements, chats, etc.

Corrected versions were annotated by professional linguists. There are two
different annotations for each text in the dev and test sets. The training set
has only one annotation.

The initial annotation was done for both grammatical error correction and
fluency. Each edit was manually classified into 22 error type categories. To
create a minimal edits GEC-only version of the corpus, fluency edits were
reverted and the resulting texts were proofread for correctness (to ensure that
reverting the annotation doesn't leave a sentence in a broken state).


## Statistics

| Split     | Correcion style | References | Documents |
|:---------:|:---------------:|:----------:|----------:|
| train     | minimal         |  1         | 1,706     |
| dev       | minimal         |  1, 2      | 87        |
| test      | minimal         |  1, 2      | 79        |
| **TOTAL** | minimal         |  1, 2      | 1,872     |
| train     | fluency         |  3         | 1,706     |
| dev       | fluency         |  3, 4      | 87        |
| test      | fluency         |  3, 4      | 79        |
| **TOTAL** | fluency         |  3, 4      | 1,872     |


## References

* [UA-GEC paper](https://aclanthology.org/2023.unlp-1.12/)

* [UA-GEC data and code](https://github.com/grammarly/ua-gec)
