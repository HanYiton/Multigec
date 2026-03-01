#!/usr/bin/env python3
"""Prepare data files for the Ukrainian GEC task."""

import yaml
import ua_gec


def save_split(docs, split_name, base_ref, num_annotators):
    print(f"Saving {split_name} split with {len(docs)} documents")
    f_out_src = open(f'uk-ua-gec-orig-{split_name}.md', 'w')
    f_out_tgts = []
    for ref in range(1, num_annotators + 1):
        f_out = open(f'uk-ua-gec-ref{ref + base_ref}-{split_name}.md', 'w')
        f_out_tgts.append(f_out)

    for doc in docs:
        header = f"### essay_id = {doc.doc_id}\n"
        ref = doc.meta.annotator_id
        if ref == 1:
            f_out_src.write(header)
            f_out_src.write(doc.source + '\n')

        f_out_tgt = f_out_tgts[ref - 1]
        f_out_tgt.write(header)
        f_out_tgt.write(doc.target + '\n')


def update_metadata(metadata_by_ref, layer, train, dev, test):
    correction_style = 'minimal' if layer == ua_gec.AnnotationLayer.GecOnly else 'fluency'
    base_ref = 0 if layer == ua_gec.AnnotationLayer.GecOnly else 2
    for annotator_id in [1, 2]:
        ref_key = f"reference_essays_{base_ref + annotator_id}"
        metadata_by_ref[ref_key] = {
                'correction_style': correction_style,
                'n_essays': {
                    'total': count_docs(train + dev + test, annotator_id),
                    'train': count_docs(train, annotator_id),
                    'dev': count_docs(dev, annotator_id),
                    'test': count_docs(test, annotator_id),
                }
        }


def count_docs(docs, annotator_id):
    return sum(1 for doc in docs if doc.meta.annotator_id == annotator_id)

def main():

    # Save GEC-only and GEC+fluency as separate references
    metadata_by_ref = {}
    for layer in [ua_gec.AnnotationLayer.GecOnly, ua_gec.AnnotationLayer.GecAndFluency]:
        # Some docs in train set have two annotators, but we can't represent
        # that in the current format. So just skip them.
        train_data = [doc for doc in ua_gec.Corpus('train', layer)
                      if doc.meta.annotator_id == 1]

        # Original UA-GEC data is split into train/test.
        # The UNLP2023 shared task further splits original test into dev and test.
        # We re-use this same partitioning scheme here. Specifically:
        # docs 0002..0430 -> 1504 sentences -> dev
        # docs 0434..     -> 1351 sentences -> test
        dev_data = []
        test_data = []
        for doc in ua_gec.Corpus('test', layer):
            if int(doc.doc_id) <= 430:
                dev_data.append(doc)
            else:
                test_data.append(doc)
        base_ref = 0 if layer == ua_gec.AnnotationLayer.GecOnly else 2

        save_split(train_data, 'train', base_ref, num_annotators=1)
        save_split(dev_data, 'dev', base_ref, num_annotators=2)
        save_split(test_data, 'test', base_ref, num_annotators=2)
        
        update_metadata(metadata_by_ref, layer, train_data, dev_data, test_data)

    print(yaml.dump(metadata_by_ref, indent=2))


if  __name__ == '__main__':
    main()
