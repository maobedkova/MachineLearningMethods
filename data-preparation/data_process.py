import pandas as pd
import re

def is_it_rt(Tweet):
    pattern = re.compile(r'^rt ')
    match = pattern.match(Tweet)
    if match:
        return True
    else:
        return False

def extract_link(tweet):
    if hasLink(tweet):
        pattern = re.compile(r'http : t . co ........')
        match = pattern.sub(r'', tweet)
        return match
    else:
        return tweet

def hasLink(tweet):
    pattern = re.compile(r'http : t . co ')
    match = pattern.findall(tweet)
    if len(match)>0:
        return True
    else:
        return False

def rep_ret(row):
    if is_it_rt(row):
        Tweet = row
        rest = re.split(' : ', Tweet, 1)
        if len(rest)>1:
            return rest[1]
        else:
            return row
    else: return row

def no_of_exc(tweet):
    pattern = re.compile(r'!')
    matches = pattern.findall(tweet)
    return int(len(matches))

def no_of_qu(tweet):
    pattern = re.compile(r'\?')
    matches = pattern.findall(tweet)
    return  int(len(matches))

import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer


def pipeline(line):
    new_line = ""
    for word in line.split():
        new_line += WordNetLemmatizer().lemmatize(word)
        new_line += " "
    return new_line


if __name__ == "__main__":
    test = pd.read_csv("newTruth_trainingSet.csv", sep=",", names=["Tweet", "Pos", "Lang", "No", "Annotation"])
    train = pd.read_csv("newTruth_testSet.csv", sep=",", names=["Tweet", "Pos", "Lang", "No", "Annotation"])

    merged = pd.concat([test, train])

    eng = merged['Lang'] == "en_UK"
    all_eng_tweets = merged[eng]

    all_eng_tweets['retweet'] = all_eng_tweets.Tweet.apply(is_it_rt)
    all_eng_tweets['Tweet'] = all_eng_tweets.Tweet.apply(rep_ret)

    all_eng_tweets['Link'] = all_eng_tweets.Tweet.apply(hasLink)
    all_eng_tweets['Tweet'] = all_eng_tweets.Tweet.apply(extract_link)
    all_eng_tweets['No_Exc'] = all_eng_tweets.Tweet.apply(no_of_exc)
    all_eng_tweets['No_Que'] = all_eng_tweets.Tweet.apply(no_of_qu)

    smile = re.compile(r': \)')
    sad = re.compile(r': \(')
    laugh = re.compile(r': [Dd]')
    heart = re.compile(r'</3')
    bad = re.compile(r': [Ss] ')
    kiss = re.compile(r': [xX]+')
    smileys = [smile, sad, laugh, heart, bad, kiss]
    smiley_cols = ['smile', 'sad', 'laugh', 'heart', 'bad', 'kiss']

    for smile in smiley_cols:
        all_eng_tweets[str(smile)] = False

    index = 0
    for tweet in all_eng_tweets.Tweet:
        col = 0
        for smiley in smileys:
            match = smiley.findall(tweet)
            if len(match) > 0:
                all_eng_tweets.iloc[index, col + 7] = True
            col += 1
        index += 1

    all_eng_tweets.drop(["Pos", "Lang"], axis=1, inplace=True)

    all_eng_tweets["lemmas"] = all_eng_tweets.Tweet.apply(pipeline)

    all_eng_tweets = all_eng_tweets.sample(frac=1).reset_index(drop=True)

    all_eng_tweets.to_csv("train_test.csv")

