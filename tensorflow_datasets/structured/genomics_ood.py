# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""genomics_ood dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_DOWNLOAD_URL = 'gs://tfds-data/downloads/genomics_ood/genomics_ood.zip'

_CITATION = """
@inproceedings{ren2019likelihood,
  title={Likelihood ratios for out-of-distribution detection},
  author={Ren, Jie and 
  Liu, Peter J and 
  Fertig, Emily and 
  Snoek, Jasper and 
  Poplin, Ryan and 
  Depristo, Mark and 
  Dillon, Joshua and 
  Lakshminarayanan, Balaji},
  booktitle={Advances in Neural Information Processing Systems},
  pages={14707--14718},
  year={2019}
}
"""

_DESCRIPTION = """
Bacteria identification based on genomic sequences holds the promise of early 
detection of diseases, but requires a model that can output low confidence 
predictions on out-of-distribution (OOD) genomic sequences from new bacteria 
that were not present in the training data.
We introduce a genomics dataset for OOD detection that allows other researchers 
to benchmark progress on this important problem.
New bacterial classes are gradually discovered over the years. Grouping classes 
by years is a natural way to mimic the in-distribution and OOD examples. 
The dataset contains genomic sequences sampled from 10 bacteria 
classes that were discovered before the year 2011 as in-distribution classes, 60 bacteria 
classes discovered between 2011-2016 as OOD for validation, and another 60 
different bacteria classes discovered after 2016 as OOD for test, in total 130 
bacteria classes. 
The details of the dataset can be found in the paper supplemental. 
"""


class GenomicsOod(tfds.core.GeneratorBasedBuilder):
  """Genomic sequence dataset for out-of-distribution (OOD) detection.

  In-distribution data contains genomic sequences randomly sampled from genomes
  of 10 bacteria classes that were discovered before the year of 2011. Training,
  validation, and test data are provided for the in-distribution classes.

  OOD data contains genomic sequences randomly sampled from genomes of 60
  bacteria classes discovered between 2011-2016, and another 60 different
  bacteria classes discovered after 2016, as OOD validation and test datasets,
  respectively. Note that no OOD classes are available at the training time.

  The genomic sequence is 250 long, composed by characters of {A, C,
  G, T}. The sample size of each class is 100,000 in the training and 10,000 for
  the validation and test sets.

  For each example, the features include:
  seq: the input DNA sequence composed by {A, C, G, T}.
  label_id: the predictive target, i.e., the index of bacteria class.
  label_name: the name of the bacteria class corresponding to label_id
  seq_info: the source of the DNA sequence, i.e., the genome name, NCBI
  accession number, and the position where it was sampled from.
  domain: if the bacteria is in-distribution (in), or OOD (ood)
  """

  VERSION = tfds.core.Version('0.1.1')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict({
            'seq': tfds.features.Text(),
            'label_id': tf.int32,
            'label_name': tfds.features.Text(),
            'seq_info': tfds.features.Text(),
            'domain': tfds.features.Text()
        }),
        # If there's a common (input, target) tuple from the features,
        # specify them here. They'll be used if as_supervised=True in
        # builder.as_dataset.
        supervised_keys=('seq', 'label_id'),
        # Homepage of the dataset for documentation
        homepage='https://github.com/google-research/google-research/tree/master/genomics_ood',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    data_path = dl_manager.download_and_extract(_DOWNLOAD_URL)
    return [
        tfds.core.SplitGenerator(
            name='train',  # name=tfds.Split.TRAIN,
            gen_kwargs={
                'filename': os.path.join(data_path, 'before_2011_in_tr.txt')
            },
        ),
        tfds.core.SplitGenerator(
            name='validation',
            gen_kwargs={
                'filename':
                    os.path.join(data_path, 'between_2011-2016_in_val.txt')
            },
        ),
        tfds.core.SplitGenerator(
            name='test',
            gen_kwargs={
                'filename': os.path.join(data_path, 'after_2016_in_test.txt')
            },
        ),
        tfds.core.SplitGenerator(
            name='validation_ood',
            gen_kwargs={
                'filename':
                    os.path.join(data_path, 'between_2011-2016_ood_val.txt')
            },
        ),
        tfds.core.SplitGenerator(
            name='test_ood',
            gen_kwargs={
                'filename': os.path.join(data_path, 'after_2016_ood_test.txt')
            },
        ),
    ]

  def _generate_examples(self, filename):
    """Yields examples."""
    with tf.io.gfile.GFile(filename) as f:
      reader = csv.DictReader(f, delimiter='\t')
      for row_id, row in enumerate(reader):
        example = {}
        example['seq'] = row['seq']
        example['label_id'] = row['label_id']
        example['label_name'] = row['label_name']
        example['seq_info'] = row['seq_info']
        example['domain'] = row['domain']

        yield row_id, example
