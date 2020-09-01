##############################
# IMPORT
##############################
from pathlib import Path
import pickle
import random
# !pip install ray
import ray
import xml.etree.ElementTree as ET
# ray.init()
# !pip install xlrd
# !pip install nltk
import nltk
nltk.download('wordnet')
import glob
# !pip install tqdm
from tqdm import tqdm
import pandas as pd
# !pip install gensim
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

##############################
# PREPROCESS FUNCTIONS
##############################

# lemmatizer - noun lemma -- https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
def lemma(word): return nltk.stem.WordNetLemmatizer().lemmatize(word)

# preprocss the word - lowercase and lemma
def pre(word): return lemma(word.lower())

def check_word(word): return " " not in word and "." not in word and "-" not in word and "/" not in word

##############################
# LOAD RELATEDNESS DATASETS
##############################

# load the files
def load_similarity_datasets():
    """Load all (13) datasets which can be used to test word interchangeable similarity
    """
    print("Loading relatedness datasets...")
    sim_data = {}
    for file_path in glob.glob("../data/word-sim/*"):
        file_name = file_path[17:].replace(".txt", "")
        try:
            df = pd.read_csv(file_path, sep="\t", header=None)
            df.columns = ['word_1', 'word_2', 'similarity_score']
        except:
            df = pd.read_csv(file_path, sep=" ", header=None)
            df.columns = ['word_1', 'word_2', 'similarity_score']
        # clean
        df.loc[:, 'word_1'] = df.loc[:, 'word_1'].apply(pre)
        df.loc[:, 'word_2'] = df.loc[:, 'word_2'].apply(pre)
        df = df.loc[df['word_1'] != df['word_2'], :]
        sim_data[file_name] = df
    return sim_data

# load similarity datasets
# similarity_datasets = load_similarity_datasets()

##############################
# LOAD ASSOCIATION DATASET
##############################

def prepare_r123_strength_table(cue_data, cue_name):
    # calculate R123 strength
    responses = cue_data.loc[:, ['R1', 'R2', 'R3']].values.reshape(1, -1)[0]
    responses = [pre(str(x)) for x in responses if str(x) != "nan" if "-" not in str(x)]
    responses = pd.DataFrame.from_dict(Counter(responses), orient='index').reset_index()
    responses.columns = ['response', 'R123']
    responses.loc[:, 'N'] = responses['R123'].sum()
    responses.loc[:, 'R123.Str'] = responses['R123'] / responses['N']
    responses.loc[:, 'cue'] = pre(str(cue_name))
    return responses

def prepare_swow_8500(conf_filter=0.1):
    """Return swow words with r123.str >= conf_filter (default:0.1 leads to 8270 unqiue cues)
    """
    # handle swow-8500
    data = pd.read_excel("../data/association/swow-8500.xlsx")
    swow_8500 = []
    for cue_name, cue_data in tqdm(data.groupby(['cue']), position=0, leave=True, desc="Loading SWOW"):
        if " " in str(cue_name):
            continue
        swow_8500.append(prepare_r123_strength_table(cue_data, cue_name))
    swow_8500 = pd.concat(swow_8500)
    swow_8500 = swow_8500.loc[swow_8500['R123.Str']>=conf_filter]
    return swow_8500

# swow_8500_df = prepare_swow_8500(df)

def clean_eat_dataset(data, conf_filter=0.1):
    data.loc[:, 'occ_conf'] = data.loc[:, 'occ_conf'].astype(float)
    data.loc[:, 'occ_count'] = data.loc[:, 'occ_count'].astype(int)
    data = data.query(f"occ_conf >= {conf_filter}")
    return data

def prepare_eat_dataset(conf_filter=0.1):
    """http://rali.iro.umontreal.ca/rali/?q=en/Textual%20Resources/EAT
    Loading 6673 out of 7182 cues by setting default 0.1 filter over occ_conf
    """
    tree = ET.parse('../data/association/eat-stimulus-response.xml')
    root = tree.getroot()
    eat_table = []
    for stimulus in tqdm(root.findall("stimulus"), position=0, leave=True, desc="Loading EAT"):
        stimulus_word = stimulus.attrib['word']
        if check_word(stimulus_word):
            for res in stimulus.findall("response"):
                res_word = res.attrib['word']
                if check_word(res_word):
                    eat_table.append({'cue': pre(stimulus_word), 'response': pre(res_word), 
                               'occ_count': res.attrib['n'], 'occ_conf': res.attrib['r']})
    eat_table = pd.DataFrame.from_dict(eat_table)
    return clean_eat_dataset(eat_table, conf_filter=0.1)

def load_association_dataset():
    print("Loading association datasets...")
    return {"swow8500": prepare_swow_8500(), "eat": prepare_eat_dataset()}

# association_datasets = load_association_dataset()


##############################
# LOAD ANALOGY DATASET
##############################

def load_google_analogy(to_pick=1000):
    """Load the google analogy dataset
    """
    random.seed(0)
    # load all datasets into dictionary
    google_analogy={}
    with open("../data/analogy/google_analogy_set.txt", 'r') as f:
        for line in f:
            line = line.replace("\n", "")
            if ":" in line: # its a title
                title = line[2:]
                google_analogy[title] = []
            else:
                analogy = [pre(x) for x in line.split() if check_word(x)]
                if len(set(analogy)) == 4:
                    google_analogy[title].append(analogy)
    # from each section, randomly pick the data (proprotion to section size)
    sections = list(google_analogy.keys())
    sections_size = [len(google_analogy[section]) for section in sections]
    sections_selection_size = (np.array(sections_size)/sum(sections_size))*to_pick
    all_picked_dataset = []
    for section, to_pick_from_section in zip(sections, sections_selection_size):
        picked = random.sample(google_analogy[section], int(to_pick_from_section))
        all_picked_dataset.extend(picked)
    return all_picked_dataset
# x = load_google_analogy()

def load_bats_analogy(random_choices=40):
    """Load bats dataset
    """
    random.seed(0)
    file_analogy = []
    for section_path in glob.glob("../data/analogy/BATS_3.0/[0-9]*"):
        if "Inflectional_morphology" in section_path: # skip as we do lemma
            continue
        section_name = section_path[10:]
        # print(f"Section: {section_name}")
        for file_path in glob.glob(section_path+"/*"):
            file_name = file_path.replace(section_path, "")
            # print(f"    File: {file_name}")
            file_analogy_prefix = []
            with open(file_path, 'r') as f:
                for line in f:
                    analogy_prefix = [pre(x) for x in line.split() if check_word(x)]
                    if len(set(analogy_prefix)) == 2:
                        file_analogy_prefix.append(analogy_prefix)
            if len(file_analogy_prefix) > random_choices*1:
                for _ in range(random_choices):
                    a, b = random.sample(file_analogy_prefix, 2)
                    a, b = a.copy(), b.copy()
                    a+=b
                    file_analogy.append(a)
    file_analogy = [x for x in file_analogy if len(set(x)) == 4]
    return file_analogy
# file_analogy = load_bats_analogy()

def load_analogy_datasets():
    print("Loading analogy datasets...")
    return {
        "google_analogy": load_google_analogy(),
        "bats_analogy": load_bats_analogy()
    }

##############################
# WRAPPER
##############################

def load_all_datasets(overwrite=False):
    file_path = "../data/all_datasets.pickle"
    # if already exist load and return
    if Path(file_path).is_file() and not overwrite:
        with open(file_path, "rb") as f:
            all_datasets = pickle.load(f)
    else:
        # create new
        all_datasets = {
            "relatedness_datasets": load_similarity_datasets(),
            "association_datasets": load_association_dataset(),
            "analogy_datasets": load_analogy_datasets()
            }
        # save
        with open(file_path, "wb") as f:
            pickle.dump(all_datasets, f)
    # finally return
    return all_datasets

if __name__ == "__main__":
    load_all_datasets(True)
    print("Done")

