'''
Copied from: https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py

Modified by: Maxlinn

1. adapt calculation of CHAIR-i and CHAIR-s for Python3, supports for both json and jsonl file input.
2. integrate synonyms.txt to make the script standalone.
3. remove machine-translation based metrics BLEU-n, CIDEr, ROGUE
4. add new metric Recall, which represents the node words(i.e. lemmas of objects) coverage overall.
5. add pickle cache mechanism to make it fast for repetitive evaluations.
'''
# import pandas as pd  # not used; avoid optional dependency
import re
import os
import sys
import nltk
import json
# from pattern.en import singularize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import argparse
import tqdm
import pickle
from collections import defaultdict


def ensure_nltk_resource(path: str, download_name: str) -> None:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(download_name, quiet=True)


def ensure_nltk_dependencies() -> None:
    ensure_nltk_resource('tokenizers/punkt', 'punkt')
    try:
        ensure_nltk_resource('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
    except Exception:
        ensure_nltk_resource('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
    ensure_nltk_resource('corpora/wordnet', 'wordnet')
    try:
        ensure_nltk_resource('corpora/omw-1.4', 'omw-1.4')
    except Exception:
        pass

# copied from: https://github.com/LisaAnne/Hallucination/blob/master/data/synonyms.txt
synonyms_txt = '''
person, girl, boy, man, woman, kid, child, chef, baker, people, adult, rider, children, baby, worker, passenger, sister, biker, policeman, cop, officer, lady, cowboy, bride, groom, male, female, guy, traveler, mother, father, gentleman, pitcher, player, skier, snowboarder, skater, skateboarder, person, woman, guy, foreigner, child, gentleman, caller, offender, coworker, trespasser, patient, politician, soldier, grandchild, serviceman, walker, drinker, doctor, bicyclist, thief, buyer, teenager, student, camper, driver, solider, hunter, shopper, villager
bicycle, bike, bicycle, bike, unicycle, minibike, trike
car, automobile, van, minivan, sedan, suv, hatchback, cab, jeep, coupe, taxicab, limo, taxi
motorcycle, scooter,  motor bike, motor cycle, motorbike, scooter, moped
airplane, jetliner, plane, air plane, monoplane, aircraft, jet, jetliner, airbus, biplane, seaplane
bus, minibus, trolley
train, locomotive, tramway, caboose
truck, pickup, lorry, hauler, firetruck
boat, ship, liner, sailboat, motorboat, dinghy, powerboat, speedboat, canoe, skiff, yacht, kayak, catamaran, pontoon, houseboat, vessel, rowboat, trawler, ferryboat, watercraft, tugboat, schooner, barge, ferry, sailboard, paddleboat, lifeboat, freighter, steamboat, riverboat, battleship, steamship
traffic light, street light, traffic signal, stop light, streetlight, stoplight
fire hydrant, hydrant
stop sign
parking meter
bench, pew
bird, ostrich, owl, seagull, goose, duck, parakeet, falcon, robin, pelican, waterfowl, heron, hummingbird, mallard, finch, pigeon, sparrow, seabird, osprey, blackbird, fowl, shorebird, woodpecker, egret, chickadee, quail, bluebird, kingfisher, buzzard, willet, gull, swan, bluejay, flamingo, cormorant, parrot, loon, gosling, waterbird, pheasant, rooster, sandpiper, crow, raven, turkey, oriole, cowbird, warbler, magpie, peacock, cockatiel, lorikeet, puffin, vulture, condor, macaw, peafowl, cockatoo, songbird
cat, kitten, feline, tabby
dog, puppy, beagle, pup, chihuahua, schnauzer, dachshund, rottweiler, canine, pitbull, collie, pug, terrier, poodle, labrador, doggie, doberman, mutt, doggy, spaniel, bulldog, sheepdog, weimaraner, corgi, cocker, greyhound, retriever, brindle, hound, whippet, husky
horse, colt, pony, racehorse, stallion, equine, mare, foal, palomino, mustang, clydesdale, bronc, bronco
sheep, lamb, ram, lamb, goat, ewe
cow, cattle, oxen, ox, calf, cattle, holstein, heifer, buffalo, bull, zebu, bison 
elephant
bear, panda
zebra
giraffe
backpack, knapsack
umbrella
handbag, wallet, purse, briefcase
tie, bow, bow tie
suitcase, suit case, luggage
frisbee
skis, ski
snowboard
sports ball, ball
kite
baseball bat
baseball glove
skateboard
surfboard, longboard, skimboard, shortboard, wakeboard
tennis racket, racket
bottle
wine glass
cup
fork
knife, pocketknife, knive
spoon
bowl, container
banana
apple
sandwich, burger, sub, cheeseburger, hamburger
orange
broccoli
carrot
hot dog
pizza
donut, doughnut, bagel
cake,  cheesecake, cupcake, shortcake, coffeecake, pancake
chair, seat, stool
couch, sofa, recliner, futon, loveseat, settee, chesterfield 
potted plant, houseplant
bed
dining table, table, desk
toilet, urinal, commode, toilet, lavatory, potty
tv, monitor, televison, television
laptop, computer, notebook, netbook, lenovo, macbook, laptop computer
mouse
remote
keyboard
cell phone, mobile phone, phone, cellphone, telephone, phon, smartphone, iPhone
microwave
oven, stovetop, stove, stove top oven
toaster
sink
refrigerator, fridge, fridge, freezer
book
clock
vase
scissors
teddy bear, teddybear
hair drier, hairdryer
toothbrush
'''


def combine_coco_captions(annotation_path):
    if not os.path.exists('%s/captions_%s2014.json' % (annotation_path, 'val')):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists('%s/captions_%s2014.json' % (annotation_path, 'train')):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open('%s/captions_%s2014.json' % (annotation_path, 'val')))
    train_caps = json.load(open('%s/captions_%s2014.json' % (annotation_path, 'train')))
    all_caps = {'info': train_caps['info'],
                'licenses': train_caps['licenses'],
                'images': val_caps['images'] + train_caps['images'],
                'annotations': val_caps['annotations'] + train_caps['annotations']}

    return all_caps


def combine_coco_instances(annotation_path):
    if not os.path.exists('%s/instances_%s2014.json' % (annotation_path, 'val')):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists('%s/instances_%s2014.json' % (annotation_path, 'train')):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open('%s/instances_%s2014.json' % (annotation_path, 'val')))
    train_instances = json.load(open('%s/instances_%s2014.json' % (annotation_path, 'train')))
    all_instances = {'info': train_instances['info'],
                     'licenses': train_instances['licenses'],
                     'type': train_instances['licenses'],
                     'categories': train_instances['categories'],
                     'images': train_instances['images'] + val_instances['images'],
                     'annotations': val_instances['annotations'] + train_instances['annotations']}

    return all_instances


class CHAIR(object):

    def __init__(self, coco_path):
        ensure_nltk_dependencies()

        self.imid_to_objects = defaultdict(list)  # later become a dict of sets

        self.coco_path = coco_path

        # read in synonyms
        synonyms = synonyms_txt.splitlines()
        synonyms = [s.strip().split(', ') for s in synonyms]
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO

        # common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = ['motor bike', 'motor cycle', 'air plane', 'traffic light', 'street light',
                             'traffic signal', 'stop light', 'fire hydrant', 'stop sign', 'parking meter', 'suit case',
                             'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'wine glass', 'hot dog',
                             'cell phone', 'mobile phone', 'teddy bear', 'hair drier', 'potted plant', 'bow tie',
                             'laptop computer', 'stove top oven', 'hot dog', 'teddy bear', 'home plate', 'train track']

        # Hard code some rules for special cases in MSCOCO
        # qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = ['bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'animal',
                        'cub']
        # qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ['jet', 'train']

        # double_word_dict will map double words to the word they should be treated as in our analysis

        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict['baby %s' % animal_word] = animal_word
            self.double_word_dict['adult %s' % animal_word] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict['passenger %s' % vehicle_word] = vehicle_word
        self.double_word_dict['bow tie'] = 'tie'
        self.double_word_dict['toilet seat'] = 'toilet'
        self.double_word_dict['wine glas'] = 'wine glass'

        self.get_annotations()

    def _load_generated_captions_into_evaluator(self, cap_file, image_id_key, caption_key):
        '''
        Meant to save time so imid_to_objects does not always need to be recomputed.
        '''
        # Read in captions
        self.think_caps, self.answer_caps, self.eval_imids = load_generated_captions(cap_file, image_id_key, caption_key)
        assert len(self.think_caps) == len(self.eval_imids)
        assert len(self.answer_caps) == len(self.eval_imids)

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def caption_to_words(self, caption):
        '''
        Input: caption
        Output: MSCOCO words in the caption
        '''

        # standard preprocessing
        words = nltk.word_tokenize(caption.lower())
        tagged_sent = nltk.pos_tag(words)
        lemmas_sent = []
        wnl = WordNetLemmatizer()
        for tag in tagged_sent:
            wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
        # words = [singularize(w) for w in words]
        words = lemmas_sent

        # replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = ' '.join(words[i:i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet" will fire for "chair" if we do not include this line)
        if ('toilet' in words) & ('seat' in words): words = [word for word in words if word != 'seat']

        # get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) \
                if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]
        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        # return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def get_annotations_from_segments(self):
        '''
        Add objects taken from MSCOCO segmentation masks
        '''

        coco_segments = combine_coco_instances(self.coco_path)
        segment_annotations = coco_segments['annotations']

        # make dict linking object name to ids
        id_to_name = {}  # dict with id to synsets
        for cat in coco_segments['categories']:
            id_to_name[cat['id']] = cat['name']

        for i, annotation in enumerate(segment_annotations):
            sys.stdout.write("\rGetting annotations for %d/%d segmentation masks"
                             % (i, len(segment_annotations)))
            imid = annotation['image_id']

            node_word = self.inverse_synonym_dict[id_to_name[annotation['category_id']]]
            self.imid_to_objects[imid].append(node_word)
        print("\n")

    def get_annotations_from_captions(self):
        '''
        Add objects taken from MSCOCO ground truth captions
        '''

        coco_caps = combine_coco_captions(self.coco_path)
        caption_annotations = coco_caps['annotations']

        for i, annotation in enumerate(caption_annotations):
            sys.stdout.write('\rGetting annotations for %d/%d ground truth captions'
                             % (i, len(coco_caps['annotations'])))
            imid = annotation['image_id']

            _, node_words, _, _ = self.caption_to_words(annotation['caption'])
            # note here is update, so call get_annotations_from_segments first
            self.imid_to_objects[imid].extend(node_words)
        print("\n")

    def get_annotations(self):
        '''
        Get annotations from both segmentation and captions.  Need both annotation types for CHAIR metric.
        '''

        self.get_annotations_from_segments()
        self.get_annotations_from_captions()
        # deduplicate
        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])


    def compute_chair(self, cap_file, image_id_key, caption_key, sample_size=None):
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        Optionally limit evaluation to the first sample_size captions.
        '''
        self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)
    
        imid_to_objects = self.imid_to_objects
        think_caps = self.think_caps
        answer_caps = self.answer_caps
        eval_imids = self.eval_imids

        indices = list(range(len(think_caps)))
        if sample_size is not None:
            if sample_size <= 0:
                raise ValueError("sample_size must be a positive integer")
            sample_size = min(sample_size, len(indices))
            indices = indices[:sample_size]
        iterator = tqdm.tqdm(indices, total=len(indices))

        num_caps = 0.
        num_hallucinated_caps = 0.
        hallucinated_word_count = 0.
        coco_word_count = 0.
        len_caps = 0.
    
        num_recall_gt_objects = 0.
        num_gt_objects = 0.
        num_generated_objects = 0.
    
        answer_num_hallucinated_caps = 0.
        answer_hallucinated_word_count = 0.
        answer_coco_word_count = 0.
        answer_len_caps = 0.
        answer_num_recall_gt_objects = 0.
        answer_num_generated_objects = 0.
    
        sample_hallucination_ratios = []
        answer_sample_hallucination_ratios = []
    
        output = {'sentences': []}
    
        for i in iterator:
            think_caption: str = think_caps[i]
            answer_caption: str = answer_caps[i]
            imid: int = eval_imids[i]
    
            gt_imid_key = imid.split('.jpg')[0][-12:].lstrip('0')
            gt_imid_key = int(gt_imid_key) if gt_imid_key else 0
            gt_objects = imid_to_objects[gt_imid_key]
    
            def evaluate_caption(caption: str):
                words, node_words, idxs, raw_words = self.caption_to_words(caption)
    
                hallucinated_pairs = []
                hallucination_idxs = []
                recall_gt_objects = set()
                hallucinated_word_count_local = 0
    
                for word, node_word, idx in zip(words, node_words, idxs):
                    if node_word not in gt_objects:
                        hallucinated_word_count_local += 1
                        hallucinated_pairs.append((word, node_word))
                        hallucination_idxs.append(idx)
                    else:
                        recall_gt_objects.add(node_word)
    
                unique_generated_objects = len(set(node_words))
                mscoco_token_count = len(words)
                sample_ratio = hallucinated_word_count_local / float(mscoco_token_count) if mscoco_token_count else 0.
    
                metrics = {'CHAIRs': int(hallucinated_word_count_local > 0),
                           'CHAIRi': sample_ratio,
                           'CHAIRi_object': sample_ratio,
                           'Recall': 0.,
                           'Precision': 0.,
                           'F1': 0.,
                           'Len': mscoco_token_count}
    
                if len(gt_objects) > 0:
                    metrics['Recall'] = len(recall_gt_objects) / len(gt_objects)
                    if unique_generated_objects > 0:
                        metrics['Precision'] = len(recall_gt_objects) / unique_generated_objects
                    if (metrics['Precision'] + metrics['Recall']) > 0:
                        metrics['F1'] = 2 * (metrics['Recall'] * metrics['Precision']) / (metrics['Precision'] + metrics['Recall'])
    
                caption_dict = {'caption': caption,
                                'mscoco_hallucinated_words': hallucinated_pairs,
                                'mscoco_gt_words': list(gt_objects),
                                'mscoco_generated_words': list(node_words),
                                'hallucination_idxs': hallucination_idxs,
                                'words': raw_words,
                                'metrics': metrics}
    
                stats = {'hallucinated_word_count': hallucinated_word_count_local,
                         'mscoco_token_count': mscoco_token_count,
                         'recall_gt_count': len(recall_gt_objects),
                         'unique_generated_objects': unique_generated_objects}
    
                return caption_dict, stats
    
            think_info, think_stats = evaluate_caption(think_caption)
            answer_info, answer_stats = evaluate_caption(answer_caption)
    
            num_caps += 1
            len_caps += think_stats['mscoco_token_count']
            coco_word_count += think_stats['mscoco_token_count']
            hallucinated_word_count += think_stats['hallucinated_word_count']
    
            if think_info['metrics']['CHAIRs']:
                num_hallucinated_caps += 1
    
            num_gt_objects += len(gt_objects)
            num_generated_objects += think_stats['unique_generated_objects']
            num_recall_gt_objects += think_stats['recall_gt_count']
    
            sample_hallucination_ratios.append(think_info['metrics']['CHAIRi'])
    
            answer_len_caps += answer_stats['mscoco_token_count']
            answer_coco_word_count += answer_stats['mscoco_token_count']
            answer_hallucinated_word_count += answer_stats['hallucinated_word_count']
            answer_num_generated_objects += answer_stats['unique_generated_objects']
            answer_num_recall_gt_objects += answer_stats['recall_gt_count']
    
            if answer_info['metrics']['CHAIRs']:
                answer_num_hallucinated_caps += 1
    
            answer_sample_hallucination_ratios.append(answer_info['metrics']['CHAIRi'])
    
            cap_dict = {'image_id': imid,
                        'caption': think_info['caption'],
                        'mscoco_hallucinated_words': think_info['mscoco_hallucinated_words'],
                        'mscoco_gt_words': think_info['mscoco_gt_words'],
                        'mscoco_generated_words': think_info['mscoco_generated_words'],
                        'hallucination_idxs': think_info['hallucination_idxs'],
                        'words': think_info['words'],
                        'metrics': think_info['metrics'],
                        'answer_caption': answer_info['caption'],
                        'answer_mscoco_hallucinated_words': answer_info['mscoco_hallucinated_words'],
                        'answer_mscoco_generated_words': answer_info['mscoco_generated_words'],
                        'answer_mscoco_gt_words': answer_info['mscoco_gt_words'],
                        'answer_hallucination_idxs': answer_info['hallucination_idxs'],
                        'answer_words': answer_info['words'],
                        'answer_metrics': answer_info['metrics']}
    
            output['sentences'].append(cap_dict)
    
        chair_s = (num_hallucinated_caps / num_caps) if num_caps else 0.
        chair_i_object = (hallucinated_word_count / coco_word_count) if coco_word_count else 0.
        chair_i_sample = (sum(sample_hallucination_ratios) / len(sample_hallucination_ratios)) if sample_hallucination_ratios else 0.
    
        recall = (num_recall_gt_objects / num_gt_objects) if num_gt_objects else 0.
        precision = (num_recall_gt_objects / num_generated_objects) if num_generated_objects else 0.
        f1 = 0.
        if (precision + recall) > 0:
            f1 = 2 * (recall * precision) / (precision + recall)
        avg_len = (0.01 * len_caps / num_caps) if num_caps else 0.
    
        output['overall_metrics'] = {'CHAIRs': chair_s,
                                     'CHAIRi': chair_i_sample,
                                     'CHAIRi_object': chair_i_object,
                                     'Recall': recall,
                                     'Precision': precision,
                                     'F1': f1,
                                     'Len': avg_len}
    
        answer_chair_s = (answer_num_hallucinated_caps / num_caps) if num_caps else 0.
        answer_chair_i_object = (answer_hallucinated_word_count / answer_coco_word_count) if answer_coco_word_count else 0.
        answer_chair_i_sample = (sum(answer_sample_hallucination_ratios) / len(answer_sample_hallucination_ratios)) if answer_sample_hallucination_ratios else 0.
    
        answer_recall = (answer_num_recall_gt_objects / num_gt_objects) if num_gt_objects else 0.
        answer_precision = (answer_num_recall_gt_objects / answer_num_generated_objects) if answer_num_generated_objects else 0.
        answer_f1 = 0.
        if (answer_precision + answer_recall) > 0:
            answer_f1 = 2 * (answer_recall * answer_precision) / (answer_precision + answer_recall)
        answer_avg_len = (0.01 * answer_len_caps / num_caps) if num_caps else 0.
    
        output['overall_metrics_answer'] = {'CHAIRs': answer_chair_s,
                                            'CHAIRi': answer_chair_i_sample,
                                            'CHAIRi_object': answer_chair_i_object,
                                            'Recall': answer_recall,
                                            'Precision': answer_precision,
                                            'F1': answer_f1,
                                            'Len': answer_avg_len}
    
        return output
    
    
    def compute_chair_token(self, image_id, caption):
        '''
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.
        '''
        # self._load_generated_captions_into_evaluator(cap_file, image_id_key, caption_key)

        imid_to_objects = self.imid_to_objects
        hallucinated_word_count = 0.
        coco_word_count = 0.

        match = re.search(r'_(\d+)\.jpg$', image_id)
        if match:
            number = match.group(1)
            result = int(number.lstrip('0')) if number.lstrip('0') else 0
        words, node_words, idxs, raw_words = self.caption_to_words(caption)
        gt_objects = imid_to_objects[result]
        cap_dict = {'image_id': image_id,
                    'caption': caption,
                    'mscoco_hallucinated_words': [],
                    'mscoco_gt_words': list(gt_objects),
                    'mscoco_generated_words': list(node_words),
                    'hallucination_idxs': [],
                    'words': raw_words
                    }
        # count hallucinated words
        coco_word_count += len(node_words)

        # add
        recall_gt_objects = set()
        for word, node_word, idx in zip(words, node_words, idxs):
            if node_word not in gt_objects:
                hallucinated_word_count += 1
                cap_dict['mscoco_hallucinated_words'].append((word, node_word))
                cap_dict['hallucination_idxs'].append(idx)
            else:
                recall_gt_objects.add(node_word)
        return cap_dict


def load_generated_captions(cap_file, image_id_key: str, caption_key: str):
    # Read in captions
    # it should be list of dict
    ext = os.path.splitext(cap_file)[-1]
    if ext == '.json':
        caps = json.load(open(cap_file))
    elif ext == '.jsonl':
        caps = [json.loads(s) for s in open(cap_file)]
    else:
        raise ValueError(f'Unspported extension {ext} for cap_file: {cap_file}')

    def _flatten(items):
        for item in items:
            if isinstance(item, list):
                yield from _flatten(item)
            else:
                yield item

    tag_pattern = re.compile(r'</?(think|answer)>', re.IGNORECASE)

    def _strip_tags(text):
        if not isinstance(text, str):
            return ''
        return tag_pattern.sub('', text).strip()

    def _extract_segments(response):
        if not isinstance(response, str):
            return '', ''

        think = ''
        answer = ''

        think_match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        if think_match:
            think = think_match.group(1).strip()
        elif '<think>' in response:
            think_part = response.split('<think>', 1)[1]
            if '</think>' in think_part:
                think_part = think_part.split('</think>', 1)[0]
            think = think_part.strip()

        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
        elif '<answer>' in response:
            answer_part = response.split('<answer>', 1)[1]
            if '</answer>' in answer_part:
                answer_part = answer_part.split('</answer>', 1)[0]
            answer = answer_part.strip()
        elif '</think>' in response:
            answer = response.split('</think>', 1)[1].strip()

        return think, _strip_tags(answer)

    # list of int
    imids = []
    think_caps = []
    answer_caps = []
    skipped = 0

    iterable_caps = caps if isinstance(caps, list) else [caps]
    for obj in _flatten(iterable_caps):
        if not isinstance(obj, dict):
            continue

        if image_id_key not in obj:
            continue

        response = obj.get('model_answer', '')
        think, answer = _extract_segments(response)

        if not think:
            think = _strip_tags(obj.get('thinking', ''))

        if not answer:
            caption_val = obj.get(caption_key, '')
            if isinstance(caption_val, str):
                answer = caption_val.strip()

        if not answer:
            answer = _strip_tags(response)

        if not answer:
            skipped += 1
            continue

        imids.append(obj[image_id_key])
        think_caps.append(think)
        answer_caps.append(answer)

    if not answer_caps:
        raise ValueError(f'No valid captions found in {cap_file}')

    if skipped:
        print(f"Filtered out {skipped} captions without valid answers from {cap_file}")

    return think_caps, answer_caps, imids


def save_hallucinated_words(cap_file, cap_dict):
    with open(cap_file, 'w') as f:
        json.dump(cap_dict, f, indent=2, ensure_ascii=False)


def print_metrics(hallucination_cap_dict, quiet=False):
    think_metrics = hallucination_cap_dict.get('overall_metrics', {})
    answer_metrics = hallucination_cap_dict.get('overall_metrics_answer', {})

    if think_metrics:
        if not quiet:
            print('Think captions metrics:')
        for k, v in think_metrics.items():
            k_str = str(k).ljust(12)
            v_str = f'{v * 100:.01f}'
            print(k_str, v_str, sep=': ')

    if answer_metrics:
        if not quiet:
            print()
            print('Answer captions metrics:')
        for k, v in answer_metrics.items():
            k_str = str(k).ljust(12)
            v_str = f'{v * 100:.01f}'
            print(k_str, v_str, sep=': ')

def chair_eval(evaluator, image_id, caption):
    cap_dict = evaluator.compute_chair_token(image_id, caption)
    return cap_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cap_file", type=str, default='/path/to/your/captions',
                        help="path towards json or jsonl saving image ids and their captions in list of dict.")
    parser.add_argument("--image_id_key", type=str, default="image_id",
                        help="in each dict of cap_file, which key stores image id of coco.")
    parser.add_argument("--caption_key", type=str, default="caption",
                        help="in each dict of cap_file, which key stores caption of the image.")
    parser.add_argument("--sample_size", type=int, default=None,
                        help="number of captions to evaluate; use all when omitted.")

    parser.add_argument("--cache", type=str, default="chair.pkl",
                        help="pre inited CHAIR evaluator object, for fast loading.")
    parser.add_argument("--coco_path", type=str, default='/root/jzq/Benchmarks/COCO/annotations',
                        # /path/to/coco/annotations
                        help="only use for regenerating CHAIR evaluator object, will be ignored if uses cached evaluator.")

    parser.add_argument("--save_path", type=str, default="./log/outputs.json",
                        help="saving CHAIR evaluate and results to json, useful for debugging the caption model.")

    args = parser.parse_args()

    evaluator = None
    if args.cache and os.path.exists(args.cache):
        try:
            with open(args.cache, 'rb') as f:
                evaluator = pickle.load(f)
            print(f"loaded evaluator from cache: {args.cache}")
        except (ModuleNotFoundError, AttributeError) as exc:
            print(f"failed to load cached evaluator '{args.cache}' due to {exc}. rebuilding...")

    if evaluator is None:
        print(f"cache not setted or not exist yet, building from scratch...")
        evaluator = CHAIR(args.coco_path)
        if args.cache:
            with open(args.cache, 'wb') as f:
                pickle.dump(evaluator, f)
            print(f"cached evaluator to: {args.cache}")

    cap_dict = evaluator.compute_chair(
        args.cap_file,
        args.image_id_key,
        args.caption_key,
        sample_size=args.sample_size,
    )
    print_metrics(cap_dict)

    if args.save_path:
        save_hallucinated_words(args.save_path, cap_dict)
