import argparse
import os
import re
import shlex
import subprocess
import MeCab

neologd_filename = 'mecab-ipadic-neologd'
res = subprocess.run(shlex.split('mecab-config --dicdir'),
                     stdout=subprocess.PIPE,
                     universal_newlines=True)
mecab_dicdir = res.stdout.strip()

if os.path.exists(os.path.join(mecab_dicdir, neologd_filename)):
    option = '-d {}/{} -O wakati'.format(mecab_dicdir, neologd_filename)
else:
    option = '-d {}/ipadic -O wakati'.format(mecab_dicdir)

mecab = MeCab.Tagger(option)


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


def main(input_path='wiki.txt', output_file='wiki_tokenized.txt', is_wiki=True):
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    wiki_path = os.path.join(data_dir, input_path)
    output_path = os.path.join(data_dir, output_file)

    create_corpus(wiki_path, output_path, is_wiki=is_wiki)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='wiki.txt')
    parser.add_argument('--output-file', type=str, default='wiki_tokenized.txt')
    option = parser.parse_args()

    main(option.input_path, option.output_file, is_wiki=True)
