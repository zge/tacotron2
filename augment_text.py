# Add emotion labels to the part of text in the filelist
#
# Examples:
#   python augment_text.py \
#     --infile filelists/soe/3x/soe_wav-text_train_3x.txt \
#     --colidx 1 \
#     --version 1
#
# Zhenhao Ge, 2020-05-17

import os
import argparse

emo_dict = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3}
delimiters=[',', '.', '!', '?', ';']

def find_next_char_idx(text, idx):
  L = len(text)
  if idx+1 > L:
    raise Exception('index out of range: idx={}, len(text)={}'.format(idx, L))
  elif idx+1 == L:
    return L
  for i in range(idx+1, L):
    if text[i] != ' ':
      return i
  return i

def split_text(text, delimiters=[',','.'], verbose=False):
  sentences, punctuations = [], []
  idx_sent_start = 0
  L = len(text)
  i = 0
  cnt = 0
  while i < L:
    if text[i] in delimiters:

      # get current sentence and punctuation
      idx_sent_end = i # exclusive
      sentences.append(text[idx_sent_start:idx_sent_end])
      idx_punc_start = i
      idx_punc_end = find_next_char_idx(text, idx_punc_start)
      punctuations.append(text[idx_punc_start:idx_punc_end])

      if verbose:
        print('sent {}: [{},{}), punc: [{},{})'.format(cnt,
          idx_sent_start, idx_sent_end, idx_punc_start, idx_punc_end))

      # post-update
      idx_sent_start = idx_punc_end
      cnt += 1
    i += 1
  if idx_sent_start < L:
    sentences.append(text[idx_sent_start:L])
    if verbose:
      print('sent {}: [{},{})'.format(cnt, idx_sent_start, L))
  return sentences, punctuations

def append_punctuation(sentences, punctuations):
  L1, L2 = len(sentences), len(punctuations)
  assert L1-L2 == 0 or L1-L2 == 1, \
    'check #sentences ({}) and #punctuations ({})'.format(L1, L2)
  sentences_with_punctuation = ['' for _ in range(L1)]
  for i in range(L1-1):
    sentences_with_punctuation[i] = sentences[i] + punctuations[i].rstrip()
  if L2 == L1:
    sentences_with_punctuation[L1-1] = sentences[L1-1] + punctuations[L1-1].rstrip()
  else:
    sentences_with_punctuation[L1-1] = sentences[L1 - 1]
  return sentences_with_punctuation

def add_emo_label(sentence, emo_dict, emotion):
  label0 = emo_dict[emotion]
  label1 = label0 + len(emo_dict)
  return '{{{0}}}{1}{{{2}}}'.format(label0, sentence, label1)

def parse_args():
  usage = ('add emotion labels to the part of text in the filelist')
  parser = argparse.ArgumentParser(usage)
  parser.add_argument('-i', '--infile', type=str, help='input filelist')
  parser.add_argument('-c', '--colidx', type=int, help='text column index')
  parser.add_argument('-x', '--version', type=int, help='version')
  parser.add_argument('-v', '--verbose', action='store_true', help='flag to show info')

  return parser.parse_args()

def main():

  # runtime mode
  args = parse_args()

  # # interative mode
  # args = argparse.ArgumentParser()
  # args.infile = 'filelists/soe/3x/soe_wav-text_train_3x.txt'
  # args.colidx = 1
  # args.version = 1
  # args.verbose = False

  outfile = args.infile.replace('text', 'text{}'.format(args.version))

  # print out input arguments
  print('input file list: {}'.format(args.infile))
  print('output file list: {}'.format(outfile))
  print('version: {}'.format(args.version))
  print('verbose: {}'.format(args.verbose))

  # read lines from input file list
  lines = open(args.infile, 'r').readlines()
  lines = [line.rstrip() for line in lines]
  N = len(lines)
  print('#lines in {}: {}'.format(args.infile, N))

  lines_with_emotion = ['' for _ in range(N)]
  for i, line in enumerate(lines):
    # get text and emotion
    parts = line.split('|')
    path, text = parts[0], parts[args.colidx]
    cat = os.path.basename(os.path.dirname(path))
    emotion = cat.split('-')[0]

    # split text into sentences
    sentences, punctuations = split_text(text, delimiters, verbose=args.verbose)
    sentences = append_punctuation(sentences, punctuations)
    sentences = [add_emo_label(sent, emo_dict, emotion) for sent in sentences]

    # merge back to sentences with emotion labels
    text_with_emotion = ' '.join(sentences)
    parts2 = parts[:args.colidx] + [text_with_emotion] + parts[args.colidx+1:]
    lines_with_emotion[i] = '|'.join(parts2)

  # write out the filelist
  open(outfile, 'w').write('\n'.join(lines_with_emotion))
  print('wrote list file: {}'.format(outfile))

if __name__ == '__main__':
  main()