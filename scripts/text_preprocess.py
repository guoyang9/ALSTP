import re
import collections

import numpy as np

def remove_char(value):
    l_temp = []

    value = re.sub('[\[\],!.;#$^*\_——<>/=%&?@"&\'-:]', ' ', str(value))
    l_temp = [i for i in value.split()]
    #for i in value.strip().split():
    #    l_temp.append(i)

    return l_temp

def remove_dup(l):
    """Remove duplicated words, first remove front words."""
    l_temp = []

    i = len(l) - 1
    while i >= 0:
        l[i] = l[i].lower()
        if l[i] not in l_temp:
            l_temp.append(l[i])
        i = i - 1

    l_temp.reverse()

    return l_temp

def remove_stop(l, l_stop):
    """Remove stop words from an example file."""
    l_temp = []
    for i in l:
        i = i.lower()
        if i not in l_stop and (len(l_temp) == 0 or i != l_temp[-1]):
            l_temp.append(i)

    return l_temp

def filter_words(l, count):
    """Filter words in documents less than count."""
    s = set() #Store words frequency less than count
    l_temp = []

    cnt = collections.Counter()
    for sentence in l:
        cnt.update(sentence)

    for word in cnt:
        if cnt[word] < count:
            s.add(word)

    for sentence in l:
        sen_temp = []
        for word in sentence:
            if word not in s:
                sen_temp.append(word)
        l_temp.append(sen_temp)

    return l_temp
