import re
import collections


def _remove_char(value):
    l_temp = []

    value = re.sub(
        '[\[\],!.;#$^*\_——<>/=%&?@"&\'-:]', ' ', str(value))
    l_temp = [i for i in value.split()]
    return l_temp


def _remove_dup(l):
    """ Remove duplicated words, first remove front ones. """
    l_temp = []

    i = len(l) - 1
    while i >= 0:
        l[i] = l[i].lower()
        if l[i] not in l_temp:
            l_temp.append(l[i])
        i = i - 1

    l_temp.reverse()
    return l_temp
    

def _filter_words(l, count):
    """ Filter words in documents less than count. """
    cnt = collections.Counter()
    for sentence in l:
        cnt.update(sentence)

    s = set(word for word in cnt if cnt[word] < count)
    l_temp = [[word 
        for word in sentence if word not in s] for sentence in l]
    return l_temp
