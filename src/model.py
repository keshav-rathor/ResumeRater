import json
import os
from typing import *

import numpy as np
import pandas as pd
from gensim.models import LdaModel
from sklearn.linear_model import LinearRegression

from .info_extractor import InfoExtractor
from .utils import (
    loadDocumentIntoSpacy,
    getAllTokensAndChunks,
    loadDefaultNLP,
)


class RatingModel:
    class RatingModelError(Exception):
        pass

    def __init__(self,_type: Optional[str] = None,
        pre_trained_model_json: Optional[str] = None,
        spacy_nlp: Optional[pd.DataFrame] = None):
        """
        Initialize a pre-trained or empty model
        """

        if _type is None:
            # empty model
            self.model = None
            self.keywords = None
        elif _type == "fixed":
            if pre_trained_model_json is None:
                raise RatingModel.RatingModel.Error("pre_trained_model_json is None")
            self.loadModelFixed(pre_trained_model_json)
        elif _type == "lda":
            if pre_trained_model_json is None:
                raise RatingModel.RatingModel.Error("pre_trained_model_json is None")
            self.loadModelLDA(pre_trained_model_json)
        else:
            raise RatingModel.RatingModelError( "type of test not valid. Either 'fixed' or 'lda'")

        print("Loading nlp tools...")
        if spacy_nlp is None:
            # load default model
            self.nlp = loadDefaultNLP()
        else:
            self.nlp = spacy_nlp

        print("Loading pdf parser...")
        # takes some time
        from tika import parser

        self.parser = parser


    def loadModelLDA(self, model_json: str) -> None:
        """
        Function to load a pre-trained ;da model
        :param model_csv: the json filename of the model
        """
        dirname = os.path.dirname(model_json)
        try:
            with open(model_json, "r") as f:
                j = json.load(f)
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "model_json %s is not a valid path" % model_json
            )

        try:
            path = os.path.join(dirname, j["model_csv"])
            self.model = pd.read_csv(path)
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError(
                "model_csv %s in model_json is not a valid path" % path
            )

        try:
            path = os.path.join(dirname, j["lda"])
            self.lda = LdaModel.load(path)
            self.dictionary = self.lda.id2word
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError("lda %s in model_json is not a valid path" % path)

        try:
            path = os.path.join(dirname, j["top_k_words"])
            self.top_k_words = []
            with open(path, "r") as f:
                for line in f:
                    if line:
                        self.top_k_words.append(line.strip())
        except Exception as e:
            print(e)
            raise RatingModel.RatingModelError("top_k_words %s in model_json is not a valid path" % path)

        self._type = "lda"


    def __keep_top_k_words(self, text):
        return [word for word in text if word in self.top_k_words]


    def __trainKMWM(self,seen_chunks_words: List[str],all_tokens_chunks: List[Any],
        keywords: List[str]) -> Optional[Tuple[List[float], List[float]]]:
        """
        Hidden function to obtain KM and WM scores from keywords
        :param seen_chunks_words: n-grams of words in doc
        :param all_tokens_chunks: list of all tokens and chunks
        :param keywords: keywords to train on
        :return: Optional[Tuple[List[float], List[float]]]: kmscores, wmscores
                                                             if no errors.
                                                             Else None
        """

        # get word2vec correlation matrix of all tokens + keyword_tokens
        keywords_tokenized = self.nlp(" ".join(keywords))
        # prepare word embedding matrix
        pd_series_all = []

        # convert tokens and chunks into word embeddings and put them into a pd.Series
        for tc in all_tokens_chunks:
            name = tc.lemma_.lower()
            pd_series_all.append(pd.Series(tc.vector, name=name))

        # convert keywords into word embeddings and put them into a pd.Series
        for kwt in keywords_tokenized:
            name = kwt.text.lower()
            if name not in seen_chunks_words:
                pd_series_all.append(pd.Series(kwt.vector, name=name))
                seen_chunks_words.append(name)
        # get embedding matrix by concatenating all pd.Series
        embedd_mat_df = pd.concat(pd_series_all, axis=1).reset_index()
        corrmat = embedd_mat_df.corr()

        # top n words correlated to keyword
        top_n = list(range(10, 100, 10))
        km_scores = []
        wm_scores = []
        try:
            for kw in keywords:
                km_similarities = []
                wm_similarities = []
                # for top n words based on correlation to kw
                for n in top_n:
                    cols = np.append(
                        corrmat[kw]
                        .drop(keywords)
                        .sort_values(ascending=False)
                        .index.values[: n - 1],
                        kw,
                    )
                    cm = np.corrcoef(embedd_mat_df[cols].values.T)

                    # KM score
                    # avg of top n correlations wrt kw (less the keyword
                    # itself since it has corr = 1)
                    avg_sim = np.mean(cm[0, :][1:])
                    km_similarities.append(avg_sim)

                    # WM score
                    # avg of top n correlations (without kw)
                    # amongst each other

                    len_minus = (
                        cm.shape[0] - 1
                    )  # cm.shape to remove all the self correlations
                    len_minus_sq = len_minus ** 2
                    # 1. sum the correlations less the
                    # correlations with the keyword
                    # 2. subtract len_minus since there are
                    # len_minus autocorrelations
                    # 3. get mean by dividing the size of the rest
                    # i.e. (len_minus_sq - len_minus)
                    avg_wm = (np.sum(cm[1:, 1:]) - len_minus) / (
                        len_minus_sq - len_minus
                    )
                    wm_similarities.append(avg_wm)

                # get 8th degree of X and perform LR to get intercept
                X = np.array(top_n)
                Xes = [X]
                # for i in range(2, 9):
                #     Xes.append(X ** i)
                X_transformed = np.array(Xes).T

                lm = LinearRegression()

                # KM score
                y = np.array(km_similarities)
                lm.fit(X_transformed, y)
                km_scores.append(lm.intercept_)

                # WM score
                y = np.array(wm_similarities)
                lm.fit(X_transformed, y)
                wm_scores.append(lm.intercept_)

        except Exception as e:
            print(e)
            return None

        return km_scores, wm_scores

    def test(self, filename: str, info_extractor: Optional[InfoExtractor]):
        """
        Test a document and print the extracted information and rating
        :param filename: name of resume file
        :param info_extractor: InfoExtractor object
        """
        if self.model is None:
            raise RatingModel.RatingModelError("model is not loaded or trained yet")
        doc, _ = loadDocumentIntoSpacy(filename, self.parser, self.nlp)

        print("Getting rating...")
        if self._type == "fixed":
            print("working on fixed model")
            if self.keywords is None:
                raise RatingModel.RatingModelError("Keywords not found")

            seen_chunks_words, all_tokens_chunks = getAllTokensAndChunks(doc)

            # scoring
            temp_out = self.__trainKMWM(list(seen_chunks_words), list(all_tokens_chunks), self.keywords)
            if temp_out is None:
                raise RatingModel.RatingModelError(
                    "Either parser cannot detect text or too few words in resume for analysis. Most usually the former."
                )
            km_scores, wm_scores = temp_out
            # average of km/wm scores for all keywords
            km_score = np.mean(km_scores)
            wm_score = np.mean(wm_scores)
            final_score = km_score * wm_score
        elif self._type == "lda":
            if self.lda is None or self.dictionary is None or self.top_k_words is None:
                raise RatingModel.RatingModelError("No LDA found")

            seen_chunks_words, all_tokens_chunks = getAllTokensAndChunks(doc)
            seen_chunks_words, all_tokens_chunks = (
                list(seen_chunks_words),
                list(all_tokens_chunks),
            )

            # scoring
            new_seen_chunks_words = self.__keep_top_k_words(seen_chunks_words)
            bow = self.dictionary.doc2bow(new_seen_chunks_words)
            doc_distribution = np.array(
                [tup[1] for tup in self.lda.get_document_topics(bow=bow)]
            )
            # get keywords and weights
            keywords = []
            all_pair_scores = []
            all_topic_scores = []
            all_diff_scores = []
            # take top 5 topics
            for j in doc_distribution.argsort()[-5:][::-1]:
                topic_prob = doc_distribution[j]
                # take top 5 words for each topic
                st = self.lda.show_topic(topicid=j, topn=5)
                sum_st = np.sum(list(map(lambda x: x[1], st)))
                pair_scores = []
                for pair in st:
                    keywords.append(pair[0])
                    pair_scores.append(pair[1])
                all_pair_scores.append(np.array(pair_scores))
                all_topic_scores.append(np.array(topic_prob))

            all_pair_scores = np.array(all_pair_scores)
            norm_all_pair_scores = all_pair_scores.T / np.sum(all_pair_scores, axis=1)
            norm_all_topic_scores = all_topic_scores / np.sum(all_topic_scores)
            all_diff_scores = (norm_all_pair_scores * norm_all_topic_scores).flatten()
            weights = pd.Series(all_diff_scores, index=keywords)
            weights.sort_values(ascending=False, inplace=True)

            temp_out = self.__trainKMWM(seen_chunks_words, all_tokens_chunks, keywords)
            if temp_out is None:
                print(
                    "Either parser cannot detect text or too few words in resume for analysis. Most usually the former. Skip document."
                )
            km_scores, wm_scores = temp_out

            # average of km/wm scores for all keywords
            km_score = np.dot(weights.values, km_scores)
            wm_score = np.dot(weights.values, wm_scores)

            final_score = km_score * wm_score

        # max_score = self.model["score"].iloc[0] - np.std(self.model["score"])
        # min_score = self.model["score"].iloc[-1]
        mean = np.mean(self.model["score"])
        sd = np.std(self.model["score"])

        rating = min(10, max(0, round(5 + (final_score-mean)/sd, 2)))
        if info_extractor is not None:
            print("-" * 20)

            # info_extractor.extractFromFile(filename)
            output= info_extractor.extractFromFile(filename)

            print("output:----",output)
            print("-" * 20)
        print("Rating: %.1f" % rating)
        # if info_extractor is not None:
        #     print("info extractor is not working")
        #     env = os.environ
        #     subprocess.call([sys.executable, filename], env=env)
        return output


