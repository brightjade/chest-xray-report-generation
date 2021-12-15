import os
import os.path as osp
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

from negbio.chexpert.constants import *
from chexpert_labeler.stages.aggregate import Aggregator
from chexpert_labeler.stages.classify import Classifier
from chexpert_labeler.stages.extract import Extractor
from chexpert_labeler.loader import Loader
from pathlib import Path


def parse_findings_from_txt(study_txt_path):
    # print(study_txt_path)
    with open(study_txt_path, 'r') as f:
        report = f.readlines()
    finding_idx = -1
    impression_idx = -1  # or CONCLUSION:
    for idx, line in enumerate(report):
        if 'FINDINGS:' in line:
            finding_idx = idx
        elif 'IMPRESSION:' in line or 'CONCLUSION:' in line:
            impression_idx = idx
            break

    if finding_idx != -1:
        if impression_idx != -1:
            findings = report[finding_idx:impression_idx]
        else:
            findings = report[finding_idx:]
        findings[0] = findings[0].replace('FINDINGS: ', '').replace('FINDINGS:', '')
        # pop_zero = False
        # for idx in range(len(findings)):
        #     if findings[idx] == '\n':
        #         findings[idx] = '  '
        #         if idx == 0:
        #             pop_zero = True
        #     elif findings[idx] == ' \n':
        #         findings[idx] = '  '
        #         if idx == 0:
        #             pop_zero = True
        # if pop_zero:
        #     findings.pop(0)
        findings = ''.join(findings).replace('\n \n ', '  ').replace('\n', '')
        findings = findings.replace('"', '')
        findings = findings.split('  ')
        findings = [line+'\n' for line in findings]
        findings = [line for line in findings if line != '\n']
        findings = [line.lstrip(' ') for line in findings]
        for line_idx in range(len(findings)):
            if ',' in findings[line_idx]:
                findings[line_idx] = '"' + findings[line_idx][:-1] + '"\n'
        with open(study_txt_path.replace('.txt', '.csv'), 'w') as f:
            f.writelines(findings)
        return True
    return False


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
    reports_path = reports_path.replace('.txt', '.csv')
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


def main_sub(indices, patient_dirs, extractor, classifier, aggregator):
    error_report_path = 'error_report.txt'
    subdir = osp.basename(osp.dirname(patient_dirs[0]))

    with tqdm(desc=osp.basename(subdir), total=len(indices), ncols=100) as pbar:
        for idx_p in indices:
            patient_dir = patient_dirs[idx_p]
            study_txt_list = os.listdir(patient_dir)
            study_txt_list = [name for name in study_txt_list if '.txt' in name]
            study_txt_paths = [osp.join(patient_dir, name) for name in study_txt_list]
            for study_txt_path in study_txt_paths:
                if osp.isfile(study_txt_path.replace('.txt', '_labeled.csv')):
                    continue
                findings = parse_findings_from_txt(study_txt_path)
                if findings:
                    labeling(study_txt_path, extractor, classifier, aggregator)
                    # try:
                    #     labeling(study_txt_path)
                    # except:
                    #     with open(error_report_path, 'a') as f:
                    #         f.write(study_txt_path + '\n')
            pbar.update(1)


def main():
    root_dir = '/home/nas1_userE/gyuhyeonsim/physionet.org/files/mimic-cxr/2.0.0/files'
    subdirs = [osp.join(root_dir, f'p1{i}') for i in range(10)]
    num_proc = 1
    target_idx_subdir = int(sys.argv[1])
    target_idx_proc = int(sys.argv[2])
    # root
    # p10 ~ p19
    # s~

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

    for idx_subdir, subdir in enumerate(subdirs):
        if target_idx_subdir != idx_subdir:
            continue
        patient_list = os.listdir(subdir)
        patient_list = [name for name in patient_list if not 'index.html' in name]
        patient_dirs = [osp.join(subdir, name) for name in patient_list]

        total_file_len = len(patient_dirs)
        indices = np.arange(total_file_len)
        split_array = np.array_split(indices, num_proc)

        main_sub(split_array[target_idx_proc], patient_dirs,
                 extractor, classifier, aggregator)


if __name__ == '__main__':
    main()
