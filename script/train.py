import argparse
import logging
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from model import paraKor
from dataloader import paraDataModule

## Generation part ##
from typing import Optional
import torch
import transformers
from transformers import AutoModelWithLMHead, PreTrainedTokenizerFast
from fastai.text.all import *
import fastai
import re

from utils import TransformersTokenizer, DropOutput
##################

parser = argparse.ArgumentParser(description='Paraphrasing Model on KoGPT-3')

parser.add_argument('--test',
                    action='store_true',
                    default=False,
                    help='response generation on given user input')

parser.add_argument('--text_file',
                    type=str,
                    default='input_example.txt',
                    help='input strings')

parser.add_argument('--model_params_para',
                    type=str,
                    default='',
                    help='model binary for paraphrasing')

parser.add_argument('--model_params_gen',
                    type=str,
                    default='',
                    help='model binary for generation')

parser.add_argument('--train_para',
                    action='store_true',
                    default=False,
                    help='for training paraphrasing')

parser.add_argument('--load_para',
                    action='store_true',
                    default=False,
                    help='training paraphrasing model from checkpoint')

parser.add_argument('--train_gen',
                    action='store_true',
                    default=False,
                    help='for training generation model')

parser.add_argument('--train_gen_file',
                    type=str,
                    default='gen_example.txt',
                    help='train set for generation model')

parser.add_argument('--gen_epochs', type=int, 
                    help='epochs for fine tuning generation model')

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

parser = paraKor.add_model_specific_args(parser)
parser = paraDataModule.add_model_specific_args(parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()
logging.info(args)

## Generation part ##
tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                    bos_token='</s>', eos_token='</s>', unk_token='<unk>',
                                                    pad_token='<pad>', mask_token='<mask>')
model = AutoModelWithLMHead.from_pretrained("skt/kogpt2-base-v2")
#################

if __name__ == "__main__":
    # Training paraphrasing model
    if args.train_para:
        if args.load_para:
            model = paraKor.load_from_checkpoint(args.model_params_para)
        else:
            model = paraKor(args)
        model.train()
        dm = paraDataModule(args)
        checkpoint_callback = ModelCheckpoint(
            filename="paraKor-{epoch:02d}-{train_loss:.2f}"
            )
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=[checkpoint_callback], gradient_clip_val=1.0)
        trainer.fit(model, dm)
    # Training generation model
    if args.train_gen:
        with open(args.train_gen_file) as f:
          lines = f.read()
          lines = lines[:int(len(lines)/1000)]  # Only using 1/1000 due to memory issue
        #split data
        train=lines[:int(len(lines)*0.9)]
        test=lines[int(len(lines)*0.9):]
        splits = [[0],[1]]
        #init dataloader
        tls = TfmdLists([train,test], TransformersTokenizer(tokenizer), splits=splits, dl_type=LMDataLoader)
        batch = 8
        seq_len = 256
        dls = tls.dataloaders(bs=batch, seq_len=seq_len)
        learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), cbs=[DropOutput], metrics=Perplexity()).to_fp16()
        learn.fine_tune(args.gen_epochs)
        # saving model
        learn.export(f'gen_finetune_{args.gen_epochs}.pkl')
    # Generating then paraphrasing
    if args.test:
        # generation model
        learn = load_learner(args.model_params_gen)
        learn.cuda()

        # paraphrasing model
        model = paraKor.load_from_checkpoint(args.model_params_para)

        # Loading text file for generation
        with open(args.text_file, 'r') as f:
            INPUT = f.read()
        
        # Text generation
        prompt_ids = tokenizer.encode(INPUT)
        inp = tensor(prompt_ids)[None].cuda()
        preds = learn.model.generate(inp,
                                  max_length=156,
                                  pad_token_id=tokenizer.pad_token_id,
                                  eos_token_id=tokenizer.eos_token_id,
                                  bos_token_id=tokenizer.bos_token_id,
                                  repetition_penalty=2.0,       
                                  use_cache=True
                                  )

        # Displaying the generated text
        GEN_TEXT = tokenizer.decode(preds[0].cpu().numpy())
        print('Generated text with the given sentence:')
        print(GEN_TEXT.split('.')[0]+'.')
        for sentence in GEN_TEXT.split('.')[1:-1]:  # [1:-1]; the first and last splits are unnecessary
            print(sentence[1:]+'.')  # [1:]; to delete unnecessary space at the start of the sentence
        print()

        # Displaying the generated text after paraphrasing
        print('Generated text after paraphrasing:')
        print(GEN_TEXT.split('.')[0]+'.')  # Paraphrasing each sentence except the first one
        for sentence in GEN_TEXT.split('.')[1:-1]:  # [1:-1]; the first and last splits are unnecessary
            # Paraphrasing
            model.test(sentence[1:]+'.')  # [1:]; to delete unnecessary space at the start of the sentence
