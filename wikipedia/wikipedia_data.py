import bz2
import collections
import os
import re
import random
from lxml import etree
import download

TRAIN_LOG_DIR = './train_logs'
EVAL_LOG_DIR = './eval_logs'
CKPT_DIR = './ckpts'
WIKI_DOWNLOAD_DIR = './wikipedia'


class Wikipedia:

    def __init__(self, url, cache_dir, vocabulary_size=10000):
        self._cache_dir = os.path.expanduser(cache_dir)
        self._pages_path = os.path.join(self._cache_dir, 'pages.bz2')
        self._vocabulary_path = os.path.join(self._cache_dir, 'vocabulary.bz2')
        if not os.path.isfile(self._pages_path):
            print('Read pages')
            self._read_pages(url)
        if not os.path.isfile(self._vocabulary_path):
            print('Build vocabulary')
            self._build_vocabulary(vocabulary_size)
        with bz2.open(self._vocabulary_path, 'rt') as vocabulary:
            print('Read vocabulary')
            self._vocabulary = [x.strip() for x in vocabulary]
        self._indices = {x: i for i, x in enumerate(self._vocabulary)}

    def __iter__(self):
        with bz2.open(self._pages_path, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                words = [self.encode(x) for x in words]
                yield words

    @property
    def vocabulary_size(self):
        return len(self._vocabulary)

    def encode(self, word):
        return self._indices.get(word, 0)

    def decode(self, index):
        return self._vocabulary[index]

    def _read_pages(self, url):
        _, file_name = os.path.split(url)
        wikipedia_path = os.path.join(self._cache_dir, file_name)
        if not os.path.isfile(wikipedia_path):
            wikipedia_path = download.download(url, wikipedia_path)
        with bz2.open(wikipedia_path) as wikipedia, \
            bz2.open(self._pages_path, 'wt') as pages:
            for _, element in etree.iterparse(wikipedia, tag='{*}page'):
                if element.find('./{*}redirect') is not None:
                    continue
                page = element.findtext('./{*}revision/{*}text')
                words = self._tokenize(page)
                pages.write(' '.join(words) + '\n')
                element.clear()

    def _build_vocabulary(self, vocabulary_size):
        counter = collections.Counter()
        with bz2.open(self._pages_path, 'rt') as pages:
            for page in pages:
                words = page.strip().split()
                counter.update(words)
        common = ['<unk>'] + counter.most_common(vocabulary_size - 1)
        common = [x[0] for x in common]
        with bz2.open(self._vocabulary_path, 'wt') as vocabulary:
            for word in common:
                vocabulary.write(word + '\n')

    TOKEN_REGEX = re.compile(r'[A-Za-z]+|[!?.:,()]')

    @classmethod
    def _tokenize(cls, page):
        words = cls.TOKEN_REGEX.findall(page)
        words = [x.lower() for x in words]
        return words
