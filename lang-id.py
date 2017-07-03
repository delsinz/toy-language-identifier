# name: Mingyang Zhang
# login: mingyangz
# id: 650242
# team: the_yummy_yawner

import json
import csv
from nltk import wordpunct_tokenize
from collections import defaultdict



def main():
    training_data = read_data('train.json')
    test_data = read_data('test.json')
    result = detect_lang(training_data, test_data)
    write_output(result)
    #dev_data = read_data('dev.json')
    #eval_accuracy(training_data, dev_data)



# file_name: string
# Returns a list of json data
def read_data(file_name):
    data = []
    with open(file_name) as f:
        for line in f:
            data.append(json.loads(line))
    return tuple(data)



# data: list of json data
# Returns a dict {lang0: token_set0, lang1: token_size1, ...}
def make_dict(data):
    vocab_base = defaultdict(set)
    for record in data:
        lang = record['lang']
        text = record['text']
        tokens = wordpunct_tokenize(text)
        for token in tokens:
            vocab_base[lang].add(token.lower())
    return vocab_base



def make_stopwords(data):
    stopwords = defaultdict(dict)
    for record in data:
        lang = record['lang']
        text = record['text']
        tokens = [token.lower() for token in wordpunct_tokenize(text)]

        if not stopwords.get(lang):
            stopwords[lang] = defaultdict(int)

        for token in tokens:
            stopwords[lang][token] += 1

    for key in stopwords.keys():
        profile_stopwords = list(reversed(sorted(stopwords[key], key=stopwords[key].get)))
        stopwords[key] = profile_stopwords[:500]

    return stopwords



def stopwords_based_detection(target, stopwords):
    target_tokens = wordpunct_tokenize(target)
    target_set = set([token.lower() for token in target_tokens])
    lang_ratios = defaultdict(int)

    for lang in stopwords.keys():
        profile_set = stopwords[lang]
        common_elements = target_set.intersection(profile_set)
        lang_ratios[lang] = len(common_elements)

    if(len(lang_ratios) == 0):
        return 'unk'
    candidate = max(lang_ratios, key=lang_ratios.get)
    return candidate


# Detection system that compares the target vocab and profile vocab
def vocab_based_detection(target, vocab_base):
    target_tokens = wordpunct_tokenize(target)
    target_set = set([token.lower() for token in target_tokens])
    lang_ratios = defaultdict(int)

    for lang in vocab_base.keys():
        profile_set = vocab_base[lang]
        common_elements = target_set.intersection(profile_set)
        lang_ratios[lang] = len(common_elements)

    if(len(lang_ratios) == 0):
        return 'unk'
    candidate = max(lang_ratios, key=lang_ratios.get)
    if(lang_ratios[candidate] / len(target_set) < 0.5):
        return 'unk'
    else:
        return candidate



# Detection system based on target token frequency in profile vocab
def freq_based_detection(target, vocab_base):
    target_tokens = wordpunct_tokenize(target)
    normalized_target_tokens = [token.lower() for token in target_tokens]
    lang_freqs = defaultdict(int)

    for lang in vocab_base.keys():
        profile_set = vocab_base[lang]
        for token in normalized_target_tokens:
            if token in profile_set:
                lang_freqs[lang] += 1

    if(len(lang_freqs) == 0):
        return 'unk'
    candidate = max(lang_freqs, key=lang_freqs.get)
    if(lang_freqs[candidate] / len(target_tokens) < 0.5):
        return 'unk'
    else:
        return candidate



# Evaluation function based on accuracy
def eval_accuracy(training_data, test_data):
    vocab_base = make_dict(training_data)
    #stopwords = make_stopwords(training_data)
    true_count = 0
    for target in test_data:
        target_lang = target['lang']
        target_text = target['text']
        detected_lang = vocab_based_detection(target_text, vocab_base)
        if detected_lang == target_lang:
            true_count += 1
    print('Accuracy: ' + str(true_count/len(test_data)))



def detect_lang(training_data, target_data):
    vocab_base = make_dict(training_data)
    output = []
    for target in target_data:
        text = target['text']
        target_dict = {}
        target_dict['docid'] = target['id']
        target_dict['lang'] = vocab_based_detection(text, vocab_base)
        output.append(target_dict)
    return output



def write_output(result):
    with open('output.csv', 'w') as f:
        fieldnames = ['docid', 'lang']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for target_dict in result:
            writer.writerow(target_dict)



if __name__ == '__main__':
    main()
