import codecs
import logging
import time
import nltk
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from nltk.util import ngrams
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


def load_raw_dataset(filename=None):
    read_handle = codecs.open(filename, 'r', 'utf-8', errors='replace')
    list_lines = read_handle.readlines()
    read_handle.close()

    dataset_raw = []
    str_tags = list_lines[0].strip()
    tags = str_tags.split('\t')
    for str_line in list_lines[1:]:
        dict_post = {}
        str_split = str_line.strip().split('\t')
        if str_split[-1] == 'humor':
            str_split[-1] = 'fake'
        for i in range(len(tags)):
            dict_post[tags[i]] = str_split[i]
        dataset_raw.append(dict_post)

    return dataset_raw


def tokenize_tweets(dataset=None, list_stopwords=None):
    for dict_tweet in dataset:
        # get text of post in its raw form
        tweet_text = dict_tweet['tweetText']
        if 'https' in tweet_text:
            tweet_text = tweet_text.split('https')[0]
        else:
            tweet_text = tweet_text.split('http')[0]

        # tokenize using newlines into sentence (sent) candidates
        list_sents = tweet_text.split('\n')

        # tokenize sent candidates using nltk sentence tokenizer, which will look to break up text further using
        # period type patterns
        nIndexSent = 0
        while nIndexSent < len(list_sents):
            str_sent = list_sents[nIndexSent]
            list_new_sents = nltk.tokenize.sent_tokenize(text=str_sent)
            if len(list_new_sents) > 0:
                list_sents[nIndexSent] = list_new_sents[0]
                for str_sent in list_new_sents[1:]:
                    list_sents.insert(nIndexSent + 1, str_sent)
                    nIndexSent = nIndexSent + 1
            else:
                list_sents[nIndexSent] = ''
            nIndexSent = nIndexSent + 1

        # tokenize each sent using nltk word tokenizer
        list_sent_tokens = []
        for str_sent in list_sents:
            list_tokens = nltk.tokenize.word_tokenize(text=str_sent, preserve_line=True)
            list_good_tokens = []
            for token in list_tokens:
                if token not in list_stopwords:
                    list_good_tokens.append(token)
            list_sent_tokens.append(list_good_tokens)

        # add token sets to dataset
        dict_tweet['sent_tokens'] = list_sent_tokens


def pos_tag_tweets(dataset=None):
    for dict_tweet in dataset:
        # get token sets for each sent
        list_sent_tokens = dict_tweet['sent_tokens']

        # pos token set
        list_sent_pos_sets = nltk.tag.pos_tag_sents(sentences=list_sent_tokens)

        # add token sets to dataset
        dict_tweet['sent_pos'] = list_sent_pos_sets


def ner_tag_tweets(dataset=None):
    for dict_tweet in dataset:
        # get pos tags for each sent
        list_sent_pos_sets = dict_tweet['sent_pos']

        # ner chunk a pos tagged sent
        list_sent_ner = []
        for list_sent_pos in list_sent_pos_sets:
            dict_NER = {}
            tree_sent = nltk.ne_chunk(list_sent_pos)
            for leaf in tree_sent:
                if isinstance(leaf, nltk.tree.Tree):
                    list_tokens = []
                    list_pos = leaf.leaves()
                    for (str_token, str_pos) in list_pos:
                        list_tokens.append(str_token)
                    str_NER_phrase = u' '.join(list_tokens)
                    str_NER_type = leaf.label()
                    if str_NER_type not in dict_NER:
                        dict_NER[str_NER_type] = []
                    dict_NER[str_NER_type].append(str_NER_phrase)
            list_sent_ner.append(dict_NER)

        # add token sets to dataset
        dict_tweet['sent_ner'] = list_sent_ner


def generate_ngrams(dataset, min_gram, max_gram, allow_pos):
    for dict_tweet in dataset:
        # get pos tags for each sent
        list_sent_pos_sets = dict_tweet['sent_pos']

        # generate ngram features for (a) tokens (b) pos sequences
        list_sent_ngrams = []
        for list_sent_pos in list_sent_pos_sets:
            list_tokens = []
            list_pos = []
            for (str_token, str_pos) in list_sent_pos:
                list_tokens.append(str_token)
                list_pos.append(str_pos)

            list_all_ngrams = []
            for nGram in range(min_gram, max_gram + 1):
                list_ngram = list(ngrams(sequence=list_tokens, n=nGram))

                # convert token list to a phrase
                for i in range(len(list_ngram)):
                    list_ngram[i] = u' '.join(list_ngram[i])
                list_all_ngrams.extend(list_ngram)

                if allow_pos:
                    list_ngram = list(ngrams(sequence=list_pos, n=nGram))

                    # convert pos list to a phrase
                    for i in range(len(list_ngram)):
                        list_ngram[i] = u' '.join(list_ngram[i])
                    list_all_ngrams.extend(list_ngram)

            list_sent_ngrams.append(list_all_ngrams)

        # add token sets to dataset
        dict_tweet['sent_ngrams'] = list_sent_ngrams


def index_features(dataset=None, allow_pos=True, allow_ngrams=True, allow_ner=True):
    dict_feature_index = {}
    list_features = []
    nFeatureID = 0
    for dict_tweet in dataset:

        list_sent_pos_sets = dict_tweet['sent_pos']
        list_sent_ngrams = dict_tweet['sent_ngrams']
        list_sent_ner = dict_tweet['sent_ner']

        # add unigram tokens (stoplist filtered) and pos
        for list_sent_pos in list_sent_pos_sets:
            for (str_token, str_pos) in list_sent_pos:

                if str_token.lower() not in dict_feature_index:
                    str_feature = str_token.lower()
                    dict_feature_index[str_feature] = nFeatureID
                    list_features.append(str_feature)
                    nFeatureID = nFeatureID + 1

                if allow_pos and (str_pos not in dict_feature_index):
                    str_feature = str_pos
                    dict_feature_index[str_feature] = nFeatureID
                    list_features.append(str_feature)
                    nFeatureID = nFeatureID + 1

        # add ngram phrases
        for list_all_ngrams in list_sent_ngrams:
            for str_phrase in list_all_ngrams:

                if allow_ngrams and (str_phrase.lower() not in dict_feature_index):
                    str_feature = str_phrase.lower()
                    dict_feature_index[str_feature] = nFeatureID
                    list_features.append(str_feature)
                    nFeatureID = nFeatureID + 1

        # add ner phrases
        for dict_ner in list_sent_ner:
            for str_ner_type in dict_ner:
                for str_phrase in dict_ner[str_ner_type]:

                    if allow_ner and (str_phrase.lower() not in dict_feature_index):
                        str_feature = str_phrase.lower()
                        dict_feature_index[str_feature] = nFeatureID
                        list_features.append(str_feature)
                        nFeatureID = nFeatureID + 1

    return dict_feature_index, list_features


def calc_count_vector(dataset=None, dict_index=None):
    # compile an index of group names
    index_group = {}
    list_groups = []
    nGroupIndex = 0
    for dict_tweet in dataset:
        if dict_tweet['label'] not in index_group:
            index_group[dict_tweet['label']] = nGroupIndex
            list_groups.append(dict_tweet['label'])
            nGroupIndex = nGroupIndex + 1

    # create count vector (rows = tweet, columns = features, cells = frequency of occurrence) with 0 freq
    list_count_vector = []
    for i in range(len(index_group)):
        list_count_vector.append([0] * len(dict_index))

    # add freq to each occurrence of a feature in a tweet
    for dict_tweet in dataset:

        # get tweets group (document type)
        str_group = dict_tweet['label']
        nGroupIndex = index_group[str_group]

        # get features in tweets
        list_sent_pos_sets = dict_tweet['sent_pos']
        list_sent_ner = dict_tweet['sent_ner']
        list_sent_ngrams = dict_tweet['sent_ngrams']

        # add unigram tokens and pos
        for list_sent_pos in list_sent_pos_sets:
            for (str_token, str_pos) in list_sent_pos:

                str_feature = str_token.lower()
                if str_feature in dict_index:
                    list_count_vector[nGroupIndex][dict_index[str_feature]] += 1

                str_feature = str_pos
                if str_feature in dict_index:
                    list_count_vector[nGroupIndex][dict_index[str_feature]] += 1

        # add ngram phrases
        for list_all_ngrams in list_sent_ngrams:
            for str_phrase in list_all_ngrams:
                str_feature = str_phrase.lower()
                if str_feature in dict_index:
                    list_count_vector[nGroupIndex][dict_index[str_feature]] += 1

        # add ner phrases
        for dict_NER in list_sent_ner:
            for str_NER_type in dict_NER:
                for str_phrase in dict_NER[str_NER_type]:
                    str_feature = str_phrase.lower()
                    if str_feature in dict_index:
                        list_count_vector[nGroupIndex][dict_index[str_feature]] += 1

    return list_count_vector, list_groups


def calc_test_train_matrix(dataset=None, list_group=None, set_features=None):
    # compile an index of group names
    index_group = {}
    nGroupIndex = 0
    for str_group in list_group:
        index_group[str_group] = nGroupIndex
        nGroupIndex = nGroupIndex + 1

    # compile an index of features
    index_features = {}
    nFeatureIndex = 0
    for str_feature in set_features:
        index_features[str_feature] = nFeatureIndex
        nFeatureIndex = nFeatureIndex + 1

    # create feature and label vectors
    X = []
    Y = []
    list_feature_set = list(set_features)
    for dict_tweet in dataset:

        # get tweet group
        str_group = dict_tweet['label']
        nGroupIndex = index_group[str_group]
        Y.append(nGroupIndex)

        # get features in tweets
        list_sent_pos_sets = dict_tweet['sent_pos']
        list_sent_ner = dict_tweet['sent_ner']
        list_sent_ngrams = dict_tweet['sent_ngrams']
        list_freq_vector = [0] * len(list_feature_set)

        # add unigram tokens and pos
        for list_sent_pos in list_sent_pos_sets:
            for (str_token, str_pos) in list_sent_pos:

                str_feature = str_token.lower()
                if str_feature in index_features:
                    list_freq_vector[index_features[str_feature]] += 1

                str_feature = str_pos
                if str_feature in index_features:
                    list_freq_vector[index_features[str_feature]] += 1

        # add ngram phrases
        for list_all_ngrams in list_sent_ngrams:
            for str_phrase in list_all_ngrams:
                str_feature = str_phrase.lower()
                if str_feature in index_features:
                    list_freq_vector[index_features[str_feature]] += 1

        # add ner phrases
        for dict_NER in list_sent_ner:
            for str_NER_type in dict_NER:
                for str_phrase in dict_NER[str_NER_type]:
                    str_feature = str_phrase.lower()
                    if str_feature in index_features:
                        list_freq_vector[index_features[str_feature]] += 1

        X.append(list_freq_vector)

    return X, Y


def logistic_regression():
    lr = LogisticRegression(max_iter=10000)
    return lr


def svm_classify():
    svm = LinearSVC()
    return svm


def knn_classify():
    knn = KNeighborsClassifier()
    return knn


def decision_tree_classify():
    dtc = DecisionTreeClassifier(max_leaf_nodes=16)
    return dtc


def random_forest_classify():
    rf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=6)
    return rf


def calc_precision(test_set=None, predict_set=None, ground_truth=None):
    nTP = 0
    nFP = 0
    for post_index in range(len(test_set)):
        if predict_set[post_index] == ground_truth[post_index]:
            nTP += 1
        else:
            nFP += 1
    if nTP + nFP > 0:
        nP = (1.0 * nTP) / (nTP + nFP)
    else:
        nP = 0.0

    print('\nTP = ', nTP, ' FP = ', nFP, ' P = ', nP, '\n')


def drawDiagram(dataset=None, set_features=None):
    dict_dataframe = {}
    list_feature = list(set_features)
    list_freq = [0] * len(list_feature)

    for feature_index in range(len(list_feature)):
        for tweet in dataset:
            list_freq[feature_index] = list_freq[feature_index] + tweet[feature_index]

    dict_dataframe['list_feature'] = list_feature
    dict_dataframe['list_freq'] = list_freq

    df = pd.DataFrame(dict_dataframe)
    df.plot.barh(x='list_feature', y='list_freq', alpha=0.5, title='top 20 feature occurrence frequency')
    plt.show()


if __name__ == '__main__':
    # make logger (global to STDOUT)
    LOG_FORMAT = '%(levelname) -s %(asctime)s %(message)s'
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger.info('logging started')

    #
    # load raw training dataset
    #
    trainFile = 'mediaeval-2015-trainingset.txt'
    logger.info('loading training dataset: ' + trainFile)
    listTrainDatasetRaw = load_raw_dataset(filename=trainFile)
    logger.info('Number of tweets in raw training dataset = ' + repr(len(listTrainDatasetRaw)))

    #
    # load raw testing dataset
    #
    testFile = 'mediaeval-2015-testset.txt'
    logger.info('loading test dataset: ' + trainFile)
    listTestDatasetRaw = load_raw_dataset(filename=testFile)
    logger.info('Number of posts in raw test dataset = ' + repr(len(listTestDatasetRaw)))

    allow_pos = False
    allow_ner = True
    allow_ngrams = True

    #
    # index all available categorical features (tokens, pos, ngrams, ner) so we can work with a count vector index,
    # not text provide a domain specific stoplist to remove tokens with little discriminating value (common words,
    # punctuation, symbols)
    #
    listStopTokens = nltk.corpus.stopwords.words()
    listStopTokens.extend(
        [':', ';', '[', ']', '"', "'", '(', ')', '.', '?', '#', '@', ',', '`', '``', "''", "'", '-', '--', '*', '|',
         '>', '<', '=', '%', '$', '+', '/', '\\', '&', '!', '!!', '!!!', '.', '..', '...', '“', '”', "'s"])

    #
    # tokenize tweets (sentence and word tokenize)
    #
    tokenize_tweets(dataset=listTrainDatasetRaw, list_stopwords=listStopTokens)
    tokenize_tweets(dataset=listTestDatasetRaw, list_stopwords=listStopTokens)
    print('SNAPSHOT #1: tokenize_tweets complete\n')

    #
    # pos tag tweets
    #
    pos_tag_tweets(dataset=listTrainDatasetRaw)
    pos_tag_tweets(dataset=listTestDatasetRaw)
    print('SNAPSHOT #2: pos_tag_tweets complete\n')

    #
    # ner tag text
    #
    ner_tag_tweets(dataset=listTrainDatasetRaw)
    ner_tag_tweets(dataset=listTestDatasetRaw)
    print('SNAPSHOT #3: ner_tag_tweets complete\n')

    #
    # create n-gram features (bigrams and trigrams)
    #
    generate_ngrams(dataset=listTrainDatasetRaw, min_gram=2, max_gram=3, allow_pos=allow_pos)
    generate_ngrams(dataset=listTestDatasetRaw, min_gram=2, max_gram=3, allow_pos=allow_pos)
    print('SNAPSHOT #4: generate_ngram complete\n')

    #
    # create index_features
    #
    (dict_feature_index_train, list_features_train) = index_features(dataset=listTrainDatasetRaw, allow_pos=allow_pos,
                                                                     allow_ner=allow_ner,
                                                                     allow_ngrams=allow_ngrams)
    (dict_feature_index_test, list_features_test) = index_features(dataset=listTestDatasetRaw, allow_pos=allow_pos,
                                                                   allow_ner=allow_ner,
                                                                   allow_ngrams=allow_ngrams)
    print('SNAPSHOT #5: index_features complete')
    print('number of features in training set = ', len(dict_feature_index_train), '\n')
    print('number of features in testing set = ', len(dict_feature_index_test), '\n')

    #
    # create a count vector. row = tweet. column = feature.
    #
    (array_count_vector, list_group) = calc_count_vector(dataset=listTrainDatasetRaw,
                                                         dict_index=dict_feature_index_train)
    print('SNAPSHOT #6: calc_count_vector complete')
    print('groups number = ', len(array_count_vector), '\n')

    transformer = sklearn.feature_extraction.text.TfidfTransformer(smooth_idf=False, use_idf=True)
    transformer.fit(array_count_vector)

    df_idf = pd.DataFrame(data=transformer.idf_, index=list_features_train, columns=['idf_weights'])
    df_idf = df_idf.sort_values(by=['idf_weights'], ascending=False)
    print(df_idf[0:20], '\n')

    tf_idf = transformer.transform(array_count_vector)

    tf_idf_vector_1 = tf_idf[0]
    df_tf_idf_1 = pd.DataFrame(tf_idf_vector_1.T.todense(), index=list_features_train, columns=['tf-idf'])
    df_tf_idf_1 = df_tf_idf_1.sort_values(by=["tf-idf"], ascending=False)
    print(df_tf_idf_1[:20], '\n')

    tf_idf_vector_2 = tf_idf[1]
    df_tf_idf_2 = pd.DataFrame(tf_idf_vector_2.T.todense(), index=list_features_train, columns=['tf-idf'])
    df_tf_idf_2 = df_tf_idf_2.sort_values(by=["tf-idf"], ascending=False)
    print(df_tf_idf_2[:20], '\n')

    #
    # Run experiment with several topN feature selection thresholds
    #
    for topN in [20, 100, 2000, 10000]:

        print('Processing top ', topN)

        #
        # prepare a categorical feature vector training set for use with a post topic classifier
        # for each post create a vector with its feature freq count
        # use a feature selection strategy of aggregating the topN TF-IDF features from each document class
        #
        set_features = set([])
        for nGroupIndex in range(len(list_group)):
            tf_idf_vector = tf_idf[nGroupIndex]
            df_tf_idf = pd.DataFrame(tf_idf_vector.T.todense(), index=list_features_train, columns=['tf-idf'])
            df_tf_idf = df_tf_idf.sort_values(by=["tf-idf"], ascending=False)
            df_tf_idf = df_tf_idf[:topN]
            for str_feature in df_tf_idf.index:
                set_features.add(str_feature)

        print('Creating training matrix')
        (X_train, Y_train) = calc_test_train_matrix(dataset=listTrainDatasetRaw, list_group=list_group,
                                                    set_features=set_features)
        if topN == 20:
            drawDiagram(dataset=X_train, set_features=set_features)

        print('Creating testing matrix')
        (X_test, Y_test) = calc_test_train_matrix(dataset=listTestDatasetRaw, list_group=list_group,
                                                  set_features=set_features)
        if topN == 20:
            drawDiagram(dataset=X_test, set_features=set_features)

        clf = logistic_regression()
        # clf = svm_classify()
        # clf = knn_classify()
        # clf = decision_tree_classify()
        # clf = random_forest_classify()

        start_time = time.time()

        print('CLF constructed')
        clf.fit(X_train, Y_train)
        print('CLF trained')
        Y_predict = clf.predict(X_test)
        print('CLF predicted')

        end_time = time.time()
        print('\nRunning time of top {} : {} seconds'.format(topN, end_time - start_time))

        #
        # Compute post classification precision
        #
        calc_precision(test_set=X_test, predict_set=Y_predict, ground_truth=Y_test)
