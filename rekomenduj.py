#!/usr/bin/python3

import collections
import pickle
import sys

from sklearn.metrics import pairwise

try:
    import readline
except ImportError:
    from pyreadline import Readline as readline


def normalize(title):
    return title.replace('’', "'").lower()


class Completer:

    def __init__(self, titles):
        self.titles = collections.defaultdict(list)
        for title in titles:
            self.titles[normalize(title)[:2]].append(title)

    def complete(self, text, state):
        if state == 0:
            text = normalize(text)
            candidates = self.titles.get(text[:2], [])
            self.matches = [
                c for c in candidates if normalize(c).startswith(text)]
        return self.matches[state]


def main():
    model = 'model.pickle' if len(sys.argv) < 2 else sys.argv[1]
    with open(model, 'rb') as file:
        titles = pickle.load(file)
        X = pickle.load(file)
    title_dict = {t: i for i, t in enumerate(titles)}
    readline.parse_and_bind('tab: complete')
    readline.set_completer(Completer(titles).complete)
    readline.set_completer_delims('')
    while True:
        title = input('Podaj tytuł> ')
        if not title:
            break
        if title not in title_dict:
            continue
        i = title_dict[title]
        distances = zip(pairwise.cosine_distances(X[i], X)[0], titles)
        for distance, title in sorted(distances)[1:6]:
            print(f'{distance:.4f} {title}')


if __name__ == '__main__':
    main()
