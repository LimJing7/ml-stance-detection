import argparse
import jsonlines
from yaml import parse

from processors.ans import ANSProcessor
from processors.argmin import ArgMinProcessor
from processors.arc import ARCProcessor
from processors.asap import ASAPProcessor
from processors.combnlpcc import CombNLPCCProcessor
from processors.fnc1 import FNC1Processor
from processors.iac1 import IAC1Processor
from processors.ibmcs import IBMCSProcessor
from processors.nlpcc import NLPCCProcessor
from processors.perspectrum import PerspectrumProcessor
from processors.xstance import XStanceProcessor
from processors.rita import RItaProcessor
from processors.semeval2016t6 import SemEval2016t6Processor
from processors.snopes import SnopesProcessor
from processors.trans_nlpcc import TransNLPCCProcessor
from processors.twitter2015 import Twitter2015Processor
from processors.twitter2017 import Twitter2017Processor
from processors.vast import VASTProcessor
from processors.tnlpcc import tNLPCCProcessor



PROCESSORS = {
  'stance': {'ans': ANSProcessor,
             'arc': ARCProcessor,
             'argmin': ArgMinProcessor,
             'asap': ASAPProcessor,
             'comb_nlpcc': CombNLPCCProcessor,
             'fnc1': FNC1Processor,
             'iac1': IAC1Processor,
             'ibmcs': IBMCSProcessor,
             'nlpcc': NLPCCProcessor,
             'perspectrum': PerspectrumProcessor,
             'r-ita': RItaProcessor,
             'semeval2016t6': SemEval2016t6Processor,
             'snopes': SnopesProcessor,
             'trans_nlpcc': TransNLPCCProcessor,
             'twitter2015': Twitter2015Processor,
             'twitter2017': Twitter2017Processor,
             'vast': VASTProcessor,
             'xstance': XStanceProcessor,
             'tnlpcc': tNLPCCProcessor}
}


def convert_to_jsonl(dataset, input_dir, output_dir):
    if dataset.startswith('xstance'):
        lang = dataset.split('-')[-1]
        processor = PROCESSORS['stance']['xstance'](lang)
    else:
        processor = PROCESSORS['stance'][dataset]()


    for split in {'train', 'dev', 'test'}:
        output_file = f'{output_dir}/{dataset}_{split}.json'
        examples = processor.get_examples(input_dir, split)
        data = [{'uid': i.guid, 'premise': i.topic, 'hypothesis': i.text, 'label': i.label} for i in examples]

        with jsonlines.open(output_file, 'w') as ofile:
            ofile.write_all(data)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datasets",
        default=["arc"],
        nargs="*",
        type=str,
        help="Train dataset(s)."
    )
    parser.add_argument(
        '--input_dir',
        default='/projects/tir5/users/jinglim/stance_datasets'
    )
    parser.add_argument(
        '--output_dir',
        default='/projects/tir5/users/jinglim/stance_datasets/senti-stance'
    )

    args = parser.parse_args()

    for dataset in args.datasets:
        print(f'{dataset=}')
        convert_to_jsonl(dataset, args.input_dir, args.output_dir)


if __name__ == '__main__':
    main()