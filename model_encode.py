import os
import json
import glob
import tqdm
import traceback
from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertConfig, BertModel

from model import OneIE
from config import Config
from util import save_result
from data import IEDatasetEval
from convert import json_to_cs

cur_dir = os.path.dirname(os.path.realpath(__file__))
format_ext_mapping = {'txt': 'txt', 'ltf': 'ltf.xml', 'json': 'json',
                      'json_single': 'json'}

def predict_document(path, model, tokenizer, config, batch_size=20, 
                     max_length=128, gpu=False, input_format='txt',
                     language='english'):
    """
    :param path (str): path to the input file.
    :param model (OneIE): pre-trained model object.
    :param tokenizer (BertTokenizer): BERT tokenizer.
    :param config (Config): configuration object.
    :param batch_size (int): Batch size (default=20).
    :param max_length (int): Max word piece number (default=128).
    :param gpu (bool): Use GPU or not (default=False).
    :param input_format (str): Input file format (txt or ltf, default='txt).
    :param langauge (str): Input document language (default='english').
    """
    test_set = IEDatasetEval(path, max_length=max_length, gpu=gpu,
                             input_format=input_format, language=language)
    test_set.numberize(tokenizer)
    # document info
    info = {
        'doc_id': test_set.doc_id,
        'ori_sent_num': test_set.ori_sent_num,
        'sent_num': len(test_set)
    }
    # prediction result
    result = []
    for batch in DataLoader(test_set, batch_size=batch_size, shuffle=False, 
                            collate_fn=test_set.collate_fn):
        graphs = model.predict(batch)
        for graph, tokens, sent_id, token_ids in zip(graphs, batch.tokens,
                                                     batch.sent_ids,
                                                     batch.token_ids):
            graph.clean(relation_directional=config.relation_directional,
                        symmetric_relations=config.symmetric_relations)
            result.append((sent_id, token_ids, tokens, graph))
    return result, info

def create_bert_model(bert_model_name):
    bert_config = BertConfig.from_pretrained(bert_model_name)
    bert = BertModel(bert_config)

    #print(bert)
    return bert

def load_model_variants(new_bert, model_path, device=0, gpu=False, beam_size=5):
    print('Loading the model from {}'.format(model_path))
    map_location = 'cuda:{}'.format(device) if gpu else 'cpu'
    state = torch.load(model_path, map_location=map_location)

    config = state['config']
    if type(config) is dict:
        config = Config.from_dict(config)
    config.bert_cache_dir = os.path.join(cur_dir, 'bert')
    vocabs = state['vocabs']
    valid_patterns = state['valid']

    # recover the model
    model = OneIE(config, vocabs, valid_patterns)
    model.load_state_dict(state['model'])
    model.beam_size = beam_size
    if gpu:
        model.cuda(device)

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_name,
                                              cache_dir=config.bert_cache_dir,
                                              do_lower_case=False)


    layers = list(model.children())
    layers[0] = new_bert

    new_model = torch.nn.Sequential(*layers)

    #print(list(new_model.children()))

    return new_model, tokenizer, config

def predict(new_bert, model_path, input_path, output_path, log_path=None, cs_path=None,
         batch_size=50, max_length=128, device=0, gpu=False,
         file_extension='txt', beam_size=5, input_format='txt',
         language='english'):
    """Perform information extraction.
    :param model_path (str): Path to the pre-trained model file.
    :param input_path (str): Path to the input directory.
    :param output_path (str): Path to the output directory.
    :param log_path (str): Path to the log file.
    :param cs_path (str): (optional) Path to the cold-start format output directory.
    :param batch_size (int): Batch size (default=50).
    :param max_length (int): Max word piece number for each sentence (default=128).
    :param device (int): GPU device index (default=0).
    :param gpu (bool): Use GPU (default=False).
    :param file_extension (str): Input file extension. Only files ending with the
    given extension will be processed (default='txt').
    :param beam_size (int): Beam size of the decoder (default=5).
    :param input_format (str): Input file format (txt or ltf, default='txt').
    :param language (str): Document language (default='english').
    """
    # set gpu device
    if gpu:
        torch.cuda.set_device(device)
    # load the model from file
    model, tokenizer, config = load_model_variants(new_bert, model_path, device=device, gpu=gpu,
                                          beam_size=beam_size)
    # get the list of documents
    file_list = glob.glob(os.path.join(input_path, '*.{}'.format(file_extension)))
    # log writer
    if log_path:
        log_writer = open(log_path, 'w', encoding='utf-8')
    # run the model; collect result and info
    doc_info_list = []
    progress = tqdm.tqdm(total=len(file_list), ncols=75)
    for f in file_list:
        progress.update(1)
        try:
            doc_result, doc_info = predict_document(
                f, model, tokenizer, config, batch_size=batch_size,
                max_length=max_length, gpu=gpu, input_format=input_format,
                language=language)
            # save json format result
            doc_id = doc_info['doc_id']
            with open(os.path.join(output_path, '{}.json'.format(doc_id)), 'w') as w:
                for sent_id, token_ids, tokens, graph in doc_result:
                    output = {
                        'doc_id': doc_id,
                        'sent_id': sent_id,
                        'token_ids': token_ids,
                        'tokens': tokens,
                        'graph': graph.to_dict()
                    }
                    w.write(json.dumps(output) + '\n')
            # write doc info
            if log_path:
                log_writer.write(json.dumps(doc_info) + '\n')
                log_writer.flush()
        except Exception as e:
            print('not working')
            traceback.print_exc()
            if log_path:
                log_writer.write(json.dumps(
                    {'file': file, 'message': str(e)}) + '\n')
                log_writer.flush()
    progress.close()

    # convert to the cold-start format
    if cs_path:
        print('Converting to cs format')
        json_to_cs(output_path, cs_path)


bert_model_names = ['bert-base-cased', 'bert-large-cased', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-base-cased-finetuned-mrpc']
bert_models = [create_bert_model(bert_model_name) for bert_model_name in bert_model_names]
#oneie_models = [load_model_variants(new_bert, './weights/english.role.v0.3.mdl') for new_bert in bert_models]

for bert_model in bert_models:
    predict(bert_model, './weights/english.role.v0.3.mdl', './input', './output')

#load_model_variants('./weights/english.role.v0.3.mdl')