# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
# Modified from Swin Transformer (https://github.com/microsoft/Swin-Transformer)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------------
from cgitb import handler
import os.path as osp
import random
import warnings
from functools import partial

import numpy as np
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from .collate import collate
from timm.data import create_transform
import torch
from torchvision import transforms as T
import us
import nltk
from timm.data.transforms import str_to_pil_interp 
from .tokenizer import SimpleTokenizer

from sclip.clip import tokenize


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(config, split="train"):
    dataset_train = build_dataset(config=config, split=split)
    us.dprint("successfully build train dataset")

    dc_collate = partial(collate, samples_per_gpu=config.batch_size)
    init_fn = partial(
        worker_init_fn, num_workers=config.num_workers, rank=dist.get_rank(), seed=config.seed
    )
    data_loader_train = wds.WebLoader(
        dataset_train.batched(config.batch_size, dc_collate, partial=False),
        batch_size=None,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.num_workers > 0,
        worker_init_fn=init_fn,
    )

    train_len = len(dataset_train)
    train_nbatches = max(1, train_len // (config.batch_size * dist.get_world_size()))
    data_loader_train = data_loader_train.with_epoch(train_nbatches).with_length(train_nbatches)

    return dataset_train, data_loader_train


def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning,
    and continue."""
    warnings.warn(repr(exn))
    return True


def build_dataset(config, split="train"):
    """
    Args:
        config: CONFIG.data (CONFIG = global config)
    """
    img_transform = build_img_transform(config.img_aug, split=split)
    text_transform = build_text_transform()
    dataset_type = None
    tar_file_list = []
    total_length = 0
    for ds in config.dataset[split]:
        ds_meta = config.dataset.meta[ds]
        if dataset_type is None:
            dataset_type = ds_meta.type
        else:
            assert dataset_type == ds_meta.type, "All datasets must be of the same type"

        prefix = ds_meta.prefix
        path = ds_meta.path
        length = ds_meta.length
        cur_tar_file_list = []
        for tar_file in braceexpand(osp.join(path, prefix)):
            if osp.exists(tar_file):
                cur_tar_file_list.append(tar_file)
        print(f"Found {len(cur_tar_file_list)} files for dataset {ds}")
        tar_file_list.extend(cur_tar_file_list)
        total_length += length
    

    print(f"Found {len(tar_file_list)} files in total for split {split}")
    if split == "train":
        dataset = (
            wds.WebDataset(tar_file_list, repeat=True, handler=warn_and_continue)
            .shuffle(40000)  # datapoint-level shuffle
            .decode("pil", handler=warn_and_continue)
            .rename(
                image="jpg;png;jpeg",
                text="text;txt",
                org_caption="text;txt",
                keep=False,
                handler=warn_and_continue,
            )
            .map_dict(image=img_transform, text=text_transform, handler=warn_and_continue)
            .with_length(total_length)
        )
    else:
        # zero shot classification validation
        dataset = (  # noqa
            wds.WebDataset(tar_file_list, repeat=False, handler=warn_and_continue)
            .shuffle(0)
            .decode('pil', handler=warn_and_continue)
            .rename(image='jpg;png;jpeg', target='target', keep=False)
            .map_dict(image=img_transform, handler=warn_and_continue)
            .slice(dist.get_rank(), total_length, dist.get_world_size())
            .with_length(total_length)
        )
    return dataset


def build_img_transform(config, split="train"):
    if split == "train":
        if not config.deit_aug:
            transform = T.Compose(
                [
                    T.RandomResizedCrop(config.img_size, scale=config.img_scale),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(mean=us.DEFAULT_MEAN, std=us.DEFAULT_STD),
                ]
            )
        else:
            # deit_aug
            transform = create_transform(
                input_size=config.img_size,
                is_training=True,
                color_jitter=config.color_jitter if config.color_jitter > 0 else None,
                auto_augment=config.auto_augment if config.auto_augment != "none" else None,
                re_prob=config.re_prob,
                re_mode=config.re_mode,
                re_count=config.re_count,
            )
    else:
        transform = T.Compose([
                T.Resize(config.img_size, interpolation=str_to_pil_interp(config.interpolation)),
                T.CenterCrop(config.img_size),
                T.ToTensor(),
                T.Normalize(mean=us.DEFAULT_MEAN, std=us.DEFAULT_STD)
            ])

    return transform


def build_text_transform(text_type='word_aug', max_seq_len=77, max_word=3):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
    if text_type == 'clip':
        transform = Tokenize()
    elif text_type == "word_aug":
        # if local_rank == 0:
        #     nltk.download('popular')
        transform = WordAugTokenizeWrapper(
            Tokenize(SimpleTokenizer(), max_seq_len=max_seq_len), 
            max_word=max_word, 
            template_set='simple', 
            word_type='noun'
        )
    return transform


# class Tokenize:
#     """Wrapper class for CLIP tokenize function."""
# 
#     def __init__(self, max_seq_len=77, truncate=True):
#         self.max_seq_len = max_seq_len
#         self.truncate = truncate
# 
#     def __call__(self, texts):
#         expanded_dim = False
#         if isinstance(texts, str):
#             texts = [texts]
#             expanded_dim = True
# 
#         result = tokenize(texts, self.max_seq_len, self.truncate)
# 
#         if expanded_dim:
#             return result[0]
# 
#         return result

class Tokenize:

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True

        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result

class WordAugTokenizeWrapper:
    def __init__(self, tokenize, max_word=3, template_set='full', word_type='noun'):
        self.tokenize = tokenize
        self.max_word = max_word
        from .imagenet_template import (full_imagenet_templates, sub_imagenet_template, simple_imagenet_template,
                                        identity_template)
        assert template_set in ['full', 'subset', 'simple', 'identity']
        if template_set == 'full':
            templates = full_imagenet_templates
        elif template_set == 'subset':
            templates = sub_imagenet_template
        elif template_set == 'simple':
            templates = simple_imagenet_template
        elif template_set == 'identity':
            templates = identity_template
        else:
            raise ValueError
        self.templates = templates
        assert word_type in ['noun', 'noun_phrase']
        self.word_type = word_type

    def get_tag(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in nltk.pos_tag(tokenized):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret

    def get_noun_phrase(self, tokenized):
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)

        chunked = chunker.parse(nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue
        return continuous_chunk

    def __call__(self, text):
        is_img_net = False

        if isinstance(text, bytes):
            text = str(text.decode('utf-8'))
            is_img_net = True

        assert isinstance(text, str)

        tokenized = nltk.word_tokenize(text)
        nouns = []
        if len(tokenized) > 0:
            if self.word_type == 'noun':
                nouns = self.get_tag(tokenized, ['NN', 'NNS', 'NNP', 'VBG', 'VB', 'VBD', 'VBN', 'VBP', 'VBZ'])
            elif self.word_type == 'noun_phrase':
                nouns = self.get_noun_phrase(tokenized)
            else:
                raise ValueError('word_type must be noun or noun_phrase')

        prompt_texts = []
        if len(nouns) > 0:
            select_nouns = np.random.choice(nouns, min(self.max_word, len(nouns)), replace=False)
            prompt_texts = [np.random.choice(self.templates).format(noun) for noun in select_nouns]
        if len(prompt_texts) < self.max_word:
            prompt_texts += [text] * (self.max_word - len(prompt_texts))

        # only select added 
        if is_img_net:
            texts = prompt_texts
        else:
            text = text.strip()
            if not text.endswith('.'):
                text += '.'
            texts = [text] + prompt_texts
        text = ' '.join(texts)
        return self.tokenize(text)
