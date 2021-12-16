import multiprocessing as mp
import os
import os.path as osp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xmltodict
from negbio.chexpert.constants import *
from tqdm import tqdm

from chexpert_labeler.loader import Loader
from chexpert_labeler.stages.aggregate import Aggregator
from chexpert_labeler.stages.classify import Classifier
from chexpert_labeler.stages.extract import Extractor


class Doc:
    def __init__(self, fd):
        self.parsed = xmltodict.parse(fd.read())

    def get_id(self):
        return self.parsed['eCitation']['uId']['@id']

    def get_findings(self):
        parsed_text = self.get_report_text()
        findings = parsed_text["FINDINGS"] if "FINDINGS" in parsed_text else ""
        return findings

    def get_report_text(self):
        abstract_text = self.parsed['eCitation']['MedlineCitation']['Article']['Abstract']['AbstractText']
        dic_abstract_text = {}
        for val in abstract_text:
            if "#text" in val:
                dic_abstract_text[val['@Label']] = val['#text']
        return dic_abstract_text


def parse_findings_from_xml(root_dir, img_name):
    report_id = img_name.split('_')[0][3:]  # CXR<num>_... -> <num>
    with open(os.path.join(root_dir, 'reports', f"{report_id}.xml"), 'r') as fd:
        doc = Doc(fd)
    findings = doc.get_findings()
    findings = findings.split('. ')
    findings = [i+'.\n' for i in findings]
    findings[-1] = findings[-1].replace('..', '.')
    findings = [i.replace('"', '') for i in findings]
    for idx in range(len(findings)):
        if ',' in findings[idx]:
            findings[idx] = '"' + findings[idx][:-1] + '"\n'
    with open(os.path.join(root_dir, 'labels', f"{report_id}.csv"), 'w') as fd:
        fd.writelines(findings)
    return True


def write(reports, labels, output_path, verbose=False):
    """Write labeled reports to specified path."""
    labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = labels[:, index]

    if verbose:
        print(f"Writing reports and labels to {output_path}.")
    labeled_reports[[REPORTS] + CATEGORIES].to_csv(output_path,
                                                   index=False)


def labeling(reports_path, extractor, classifier, aggregator):

    extract_impression = False
    output_path = reports_path.replace('.csv', '_labeled.csv')
    reports_path = Path(reports_path)
    output_path = Path(output_path)

    loader = Loader(reports_path, extract_impression)

    # Load reports in place.
    loader.load()
    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    labels = aggregator.aggregate(loader.collection)

    if labels.size != 0:
        write(loader.reports, labels, output_path, False)

    pass


def main_sub(indices, root_dir, img_names, phase, extractor, classifier, aggregator):
    # error_report_path = 'error_report.txt'

    with tqdm(desc=phase, total=len(indices), ncols=100) as pbar:
        for idx_p in indices:
            img_name = img_names[idx_p]
            report_id = img_name.split('_')[0][3:]  # CXR<num>_... -> <num>
            if osp.isfile(osp.join(root_dir, 'labels', f'{report_id}_labeled.csv')):
                pbar.update(1)
                continue
            findings = parse_findings_from_xml(root_dir, img_name)
            if findings:
                labeling(osp.join(root_dir, 'labels', f"{report_id}.csv"), extractor, classifier, aggregator)
            pbar.update(1)


def main():
    root_dir = '/home/nas1_userA/minseokchoi20/coursework/2021fall/ai604/ai604-cv-project/data/openi'
    num_proc = 5
    phase = sys.argv[1]  # train val test
    target_idx_proc = int(sys.argv[2])

    # labeler
    mention_phrases_dir = 'chexpert_labeler/phrases/mention'
    unmention_phrases_dir = 'chexpert_labeler/phrases/unmention'
    pre_negation_uncertainty_path = 'chexpert_labeler/patterns/pre_negation_uncertainty.txt'
    negation_path = 'chexpert_labeler/patterns/negation.txt'
    post_negation_uncertainty_path = 'chexpert_labeler/patterns/post_negation_uncertainty.txt'
    mention_phrases_dir = Path(mention_phrases_dir)
    unmention_phrases_dir = Path(unmention_phrases_dir)
    verbose = False

    extractor = Extractor(mention_phrases_dir,
                          unmention_phrases_dir,
                          verbose=verbose)
    classifier = Classifier(pre_negation_uncertainty_path,
                            negation_path,
                            post_negation_uncertainty_path,
                            verbose=verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=verbose)

    with open(osp.join(root_dir, f'{phase}_img_names.txt'), 'r') as f:
        img_names = f.readlines()
    img_names = [i[:-1] for i in img_names]

    total_file_len = len(img_names)
    indices = np.arange(total_file_len)
    split_array = np.array_split(indices, num_proc)

    main_sub(split_array[target_idx_proc], root_dir, img_names, phase,
             extractor, classifier, aggregator)


if __name__ == '__main__':
    main()
