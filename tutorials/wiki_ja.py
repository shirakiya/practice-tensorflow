import os
import re
import shlex
import subprocess
import MeCab

res = subprocess.run(shlex.split('mecab-config --dicdir'),
                     stdout=subprocess.PIPE,
                     universal_newlines=True)
mecab_dicdir = res.stdout.strip()

if os.path.exists(os.path.join(mecab_dicdir, 'mecab-ipadic-neologd')):
    mecab = MeCab.Tagger('-d {}/mecab-ipadic-neologd -O wakati'.format(mecab_dicdir))
else:
    mecab = MeCab.Tagger('-d {}/ipadic -O wakati'.format(mecab_dicdir))


def tokenize(text):
    return mecab.parse(text).strip()


def get_files_list(input_path):
    filelist = []
    if os.path.isfile(input_path):
        filelist.append(input_path)
    else:
        for root, _, files in os.walk(input_path):
            filelist.extend([os.path.join(root, f) for f in files])
    return filelist


def __ignore_text(text, regexp=None):
    if not text:
        return True
    elif regexp and regexp.match(text):
        return True
    return False


def create_corpus(input_path, output_path, is_wiki=False):
    files = get_files_list(input_path)
    files_length = len(files)
    output_file = open(output_path, 'a')

    escape_regexp = None
    if is_wiki:
        escape_regexp = re.compile('<doc|</doc')  # after using "re.match()"

    for index, filepath in enumerate(files):
        print('[{count}/{all}] {file} executed...'.format(count=index+1,
                                                          all=files_length,
                                                          file=filepath))
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if __ignore_text(line, escape_regexp):
                    continue
                output_file.write('{}\n'.format(tokenize(line)))

    output_file.close()


if __name__ == '__main__':
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    wiki_path = os.path.join(data_dir, 'wiki.txt')
    output_path = os.path.join(data_dir, 'wiki_tokenized.txt')

    create_corpus(wiki_path, output_path, is_wiki=True)
