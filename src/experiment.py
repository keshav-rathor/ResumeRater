import os
import argparse
import random
import json
import math
import subprocess
import time
from collections import defaultdict
from typing import *

import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from nltk import FreqDist
from sklearn.linear_model import LinearRegression
import re
import pandas as pd
# import sys

from collections import Counter, defaultdict
from datetime import datetime
from dateutil import relativedelta

import re
import spacy
import docx

# -----Utils------------------------

def loadDefaultNLP(is_big: bool = True) -> Any:
    """
    Function to load the default SpaCy nlp model into self.nlp
    :param is_big: if True, uses a large vocab set, else a small one
    :returns: nlp: a SpaCy nlp model
    """

    def segment_on_newline(doc):
        for token in doc[:-1]:
            if token.text.endswith("\n"):
                doc[token.i + 1].is_sent_start = True
        return doc

    if is_big:
        nlp = spacy.load("en_core_web_lg")
    else:
        nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(segment_on_newline, before="parser")
    return nlp


def countWords(line: str) -> int:
    """
    Counts the numbers of words in a line
    :param line: line to count
    :return count: num of lines
    """
    count = 0
    is_space = False
    for c in line:
        is_not_char = not c.isspace()
        if is_space and is_not_char:
            count += 1
        is_space = not is_not_char
    return count


def getAllTokensAndChunks(doc) -> Tuple[Set[Any], Set[Any]]:
    """
    Converts a spacy doc into tokens and chunks. Tokens and chunks pass through a customFilter first
    :param doc: a SpaCy doc
    :returns: seen_chunks_words: set of strings seen
    :returns: all_tokens_chunks: set of all tokens and chunks found
    """
    # used to test duplicate words/chunks
    seen_chunks_words = set()
    # collate all words/chunks
    all_tokens_chunks = set()
    # generate all 1-gram tokens
    for token in doc:
        w = token.lemma_.lower()
        if (w not in seen_chunks_words) and customFilter(token):
            all_tokens_chunks.add(token)
            seen_chunks_words.add(w)

    # generate all n-gram tokens
    for chunk in doc.noun_chunks:
        c = chunk.lemma_.lower()
        if (
            len(chunk) > 1
            and (c not in seen_chunks_words)
            and all(customFilter(token) for token in chunk)
        ):
            all_tokens_chunks.add(chunk)
            seen_chunks_words.add(c)

    return seen_chunks_words, all_tokens_chunks


def findDocumentsRecursive(base_dir: str):
    """
    Recursively get all documents from `base_dir`
    :param base_dir: base directory of documents
    :returns out: a list of full file names of the documents
    """
    out: List[str] = []

    # check if base_dir is a proper dir
    if not os.path.isdir(base_dir):
        return None

    for d in os.listdir(base_dir):
        full_path = os.path.join(base_dir, d)
        if os.path.isdir(full_path):
            out.extend(findDocumentsRecursive(full_path))
        else:
            for end in (".pdf", ".docx"):
                if full_path.endswith(end):
                    out.append(full_path)
    return out


def generateDFFromData(
    data: Dict[Any, Any],
    filename: str,
    save_csv: bool = False
) -> pd.DataFrame:
    """
    Generates DF for model creation
    :param data: dictionary of data
    :param filename: what to save model as
    :param save_csv: whether to save the model as csv
    :returns data_df: the model df
    """
    data_df = pd.DataFrame(data=data)
    data_df.sort_values(by=["score"], ascending=False, inplace=True)
    data_df.reset_index(inplace=True)
    if save_csv:
        data_df.to_csv(filename)
    return data_df


def getDocxText(filename: str) -> str:
    """
    Get the text from a docx file
    :param filename: docx file
    :returns fullText: text of file
    """
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        txt = para.text
        fullText.append(txt)
    return "\n".join(fullText)


def getPDFText(filename: str, parser) -> str:
    """
    Get the text from a pdf file
    :param filename: pdf file
    :param parser: pdf parser
    :returns fullText: text of file
    """
    raw = parser.from_file(filename)
    new_text = raw["content"]
    if "title" in raw["metadata"]:
        title = raw["metadata"]["title"]
        new_text = new_text.replace(title, "")
    return new_text


def loadDocumentIntoSpacy(f: str, parser, spacy_nlp):
    """
    Convert file into spacy Document
    :param f: filename
    :param parser: pdf_parser
    :param spacy_nlp: nlp model
    :returns nlp_doc: nlp doc
    :returns new_text: text of file
    """
    if f.endswith(".pdf"):
        new_text = getPDFText(f, parser)
    elif f.endswith(".docx"):
        new_text = getDocxText(f)
    else:
        return None, None

    # new_text = "\n".join(
    #     [line.strip() for line in new_text.split("\n") if len(line) > 1]
    # )
    new_text = re.sub("\n{3,}", "\n", new_text)
    new_text = str(bytes(new_text, "utf-8").replace(b"\xe2\x80\x93", b""), "utf-8")
    # convert to spacy doc
    return spacy_nlp(new_text), new_text

# -----------------------------------------------

# ------------Extractor----------------------------
WORDS_LIST = {
    "Work": ["(Work|WORK)", "(Experience(s?)|EXPERIENCE(S?))", "(History|HISTORY)"],
    "Education": ["(Education|EDUCATION)", "(Qualifications|QUALIFICATIONS)"],
    "Skills": [
        "(Skills|SKILLS)",
        "(Proficiency|PROFICIENCY)",
        "LANGUAGE",
        "CERTIFICATION",
    ],
    "Projects": ["(Projects|PROJECTS)"],
    "Activities": ["(Leadership|LEADERSHIP)", "(Activities|ACTIVITIES)"],
}

def __init__(self, spacy_nlp_model, parser):
    self.nlp = spacy_nlp_model
    self.parser = parser

def extractFromFile(self, filename):
    doc, text = loadDocumentIntoSpacy(filename, self.parser, self.nlp)
    self.extractFromText(doc, text, filename)

def extractFromText(self, doc, text, filename):
    l = []
    name =findName(doc, filename)
    if name is None:
        name = ""
    email =findEmail(doc)

    if email is None:
        email = ""
    number = findNumber(doc)
    if number is None:
        number = ""
    city =findCity(doc)
    if city is None:
        city = ""
    categories =extractCategories(text)
    workAndEducation =findWorkAndEducation(
        categories, doc, text, name
    )
    totalWorkExperience =getTotalExperienceFormatted(
        workAndEducation["Work"]
    )
    totalEducationExperience = getTotalExperienceFormatted(
        workAndEducation["Education"]
    )
    allSkills = ", ".join(extractSkills(doc))
    print("Name: %s" % name)
    print("Email: %s" % email)
    print("Number: %s" % number)
    print("City/Country: %s" % city)
    print("\nWork Experience:")
    print(totalWorkExperience)
    l.append(name)
    l.append(email)
    l.append(number)
    for w in workAndEducation["Work"]:
        print(" - " + w)
    print("\nEducation:")
    print(totalEducationExperience)
    for e in workAndEducation["Education"]:
        print(" - " + e)
    print("\nSkills:")
    print(allSkills)
    print('******' * 3)
    print(l)
    print('******' * 3)
    return l

def extractSkills(doc) -> List[str]:
    """
    Helper function to extract skills from spacy nlp text
    :param doc: object of `spacy.tokens.doc.Doc`
    :return: list of skills extracted
    """
    tokens = [token.text for token in doc if not token.is_stop]
    data = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "constants/skills.csv")
    )
    skills = list(data.columns.values)
    skillset = []
    # check for one-grams
    for token in tokens:
        if token.lower() in skills:
            skillset.append(token)

    # check for bi-grams and tri-grams
    for token in doc.noun_chunks:
        token = token.text.lower().strip()
        if token in skills:
            skillset.append(token)
    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def extractCategories(text) -> Dict[str, List[Tuple[int, int]]]:
    """
    Helper function to extract categories like EDUCATION and EXPERIENCE from text
    :param text: text
    :return: Dict[str, List[Tuple[int, int]]]: {category: list((size_of_category, page_count))}
    """
    data = defaultdict(list)
    page_count = 0
    prev_count = 0
    prev_line = None
    prev_k = None
    for line in text.split("\n"):
        line = re.sub(r"\s+?", " ", line).strip()
        for (k, wl) in WORDS_LIST.items():
            # for each word in the list
            for w in wl:
                # if category has not been found and not a very long line
                # - long line likely not a category
                if countWords(line) < 10:
                    match = re.findall(w, line)
                    if match:
                        size = page_count - prev_count
                        # append previous
                        if prev_k is not None:
                            data[prev_k].append((size, prev_count, prev_line))
                        prev_count = page_count
                        prev_k = k
                        prev_line = line
        page_count += 1

    # last item
    if prev_k is not None:
        size = page_count - prev_count - 1  # -1 cuz page_count += 1 on prev line
        data[prev_k].append((size, prev_count, prev_line))

    # choose the biggest category (reduce false positives)
    for k in data:
        if len(data[k]) >= 2:
            data[k] = [max(data[k], key=lambda x: x[0])]
    return data



def findWorkAndEducation(categories, doc, text, name) -> Dict[str, List[str]]:
    inv_data = {v[0][1]: (v[0][0], k) for k, v in categories.items()}
    line_count = 0
    exp_list = defaultdict(list)
    name = name.lower()

    current_line = None
    is_dot = False
    is_space = True
    continuation_sent = []
    first_line = None
    unique_char_regex = "[^\sA-Za-z0-9\.\/\(\)\,\-\|]+"

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        match = re.search(r"^.*:", line)
        if match:
            line = line[match.end():].strip()

        # get first non-space line for filtering since
        # sometimes it might be a page header
        if line and first_line is None:
            first_line = line

        # update line_countfirst since there are `continue`s below
        line_count += 1
        if (line_count - 1) in inv_data:
            current_line = inv_data[line_count - 1][1]
        # contains a full-blown state-machine for filtering stuff
        elif current_line == "Work":
            if line:
                # if name is inside, skip
                if name == line:
                    continue
                # if like first line of resume, skip
                if line == first_line:
                    continue
                # check if it's not a list with some unique character as list bullet
                has_dot = re.findall(unique_char_regex, line[:5])
                # if last paragraph is a list item
                if is_dot:
                    # if this paragraph is not a list item and the previous line is a space
                    if not has_dot and is_space:
                        if line[0].isupper() or re.findall(r"^\d+\.", line[:5]):
                            exp_list[current_line].append(line)
                            is_dot = False

                else:
                    if not has_dot and (
                            line[0].isupper() or re.findall(r"^\d+\.", line[:5])
                    ):
                        exp_list[current_line].append(line)
                        is_dot = False
                if has_dot:
                    is_dot = True
                is_space = False
            else:
                is_space = True
        elif current_line == "Education":
            if line:
                # if not like first line
                if line == first_line:
                    continue
                line = re.sub(unique_char_regex, '', line[:5]) + line[5:]
                if len(line) < 12:
                    continuation_sent.append(line)
                else:
                    if continuation_sent:
                        continuation_sent.append(line)
                        line = " ".join(continuation_sent)
                        continuation_sent = []
                    exp_list[current_line].append(line)

    return exp_list

def findNumber(doc):
    """
    Helper function to extract number from nlp doc
    :param doc: SpaCy Doc of text
    :return: int:number if found, else None
    """
    for sent in doc.sents:
        num = re.findall(r"\(?\+?\d+\)?\d+(?:[- \)]+\d+)*", sent.text)
        if num:
            for n in num:
                if len(n) >= 8 and (
                        not re.findall(r"^[0-9]{2,4} *-+ *[0-9]{2,4}$", n)
                ):
                    return n
    return None

def findEmail(doc):
    """
    Helper function to extract email from nlp doc
    :param doc: SpaCy Doc of text
    :return: str:email if found, else None
    """
    for token in doc:
        if token.like_email:
            return token.text
    return None


def findCity(doc):
    counter = Counter()
    """
    Helper function to extract most likely City/Country from nlp doc
    :param doc: SpaCy Doc of text
    :return: str:city/country if found, else None
    """
    for ent in doc.ents:
        if ent.label_ == "GPE":
            counter[ent.text] += 1

    if len(counter) >= 1:
        return counter.most_common(1)[0][0]
    return None

def findName(doc, filename) -> Optional[str]:
    """
    Helper function to extract name from nlp doc
    :param doc: SpaCy Doc of text
    :param filename: used as backup if NE cannot be found
    :return: str:NAME_PATTERN if found, else None
    """
    to_chain = False
    all_names = []
    person_name = None

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            if not to_chain:
                person_name = ent.text.strip()
                to_chain = True
            else:
                person_name = person_name + " " + ent.text.strip()
        elif ent.label_ != "PERSON":
            if to_chain:
                all_names.append(person_name)
                person_name = None
                to_chain = False
    if all_names:
        return all_names[0]
    else:
        try:
            base_name_wo_ex = os.path.splitext(os.path.basename(filename))[0]
            return base_name_wo_ex + " (from filename)"
        except:
            return None


def getNumberOfMonths(datepair) -> int:
    """
    Helper function to extract total months of experience from a resume
    :param date1: Starting date
    :param date2: Ending date
    :return: months of experience from date1 to date2
    """
    # if years
    # if years
    date2_parsed = False
    if datepair.get("fh", None) is not None:
        gap = datepair["fh"]
    else:
        gap = ""
    try:
        present_vocab = ("present", "date", "now")
        if "syear" in datepair:
            date1 = datepair["fyear"]
            date2 = datepair["syear"]

            if date2.lower() in present_vocab:
                date2 = datetime.now()
                date2_parsed = True

            try:
                if not date2_parsed:
                    date2 = datetime.strptime(str(date2), "%Y")
                date1 = datetime.strptime(str(date1), "%Y")
            except:
                pass
        elif "smonth_num" in datepair:
            date1 = datepair["fmonth_num"]
            date2 = datepair["smonth_num"]

            if date2.lower() in present_vocab:
                date2 = datetime.now()
                date2_parsed = True

            for stype in ("%m" + gap + "%Y", "%m" + gap + "%y"):
                try:
                    if not date2_parsed:
                        date2 = datetime.strptime(str(date2), stype)
                    date1 = datetime.strptime(str(date1), stype)
                    break
                except:
                    pass
        else:
            date1 = datepair["fmonth"]
            date2 = datepair["smonth"]

            if date2.lower() in present_vocab:
                date2 = datetime.now()
                date2_parsed = True

            for stype in (
                    "%b" + gap + "%Y",
                    "%b" + gap + "%y",
                    "%B" + gap + "%Y",
                    "%B" + gap + "%y",
            ):
                try:
                    if not date2_parsed:
                        date2 = datetime.strptime(str(date2), stype)
                    date1 = datetime.strptime(str(date1), stype)
                    break
                except:
                    pass

        months_of_experience = relativedelta.relativedelta(date2, date1)
        months_of_experience = (
                months_of_experience.years * 12 + months_of_experience.months
        )
        return months_of_experience
    except Exception as e:
        return 0

def getTotalExperience(experience_list) -> int:
    """
    Wrapper function to extract total months of experience from a resume
    :param experience_list: list of experience text extracted
    :return: total months of experience
    """
    exp_ = []
    for line in experience_list:
        line = line.lower().strip()
        # have to split search since regex OR does not capture on a first-come-first-serve basis
        experience = re.search(
            r"(?P<fyear>\d{4})\s*(\s|-|to)\s*(?P<syear>\d{4}|present|date|now)",
            line,
            re.I,
        )
        if experience:
            d = experience.groupdict()
            exp_.append(d)
            continue

        experience = re.search(
            r"(?P<fmonth>\w+(?P<fh>.)\d+)\s*(\s|-|to)\s*(?P<smonth>\w+(?P<sh>.)\d+|present|date|now)",
            line,
            re.I,
        )
        if experience:
            d = experience.groupdict()
            exp_.append(d)
            continue

        experience = re.search(
            r"(?P<fmonth_num>\d+(?P<fh>.)\d+)\s*(\s|-|to)\s*(?P<smonth_num>\d+(?P<sh>.)\d+|present|date|now)",
            line,
            re.I,
        )
        if experience:
            d = experience.groupdict()
            exp_.append(d)
            continue
    experience_num_list = [getNumberOfMonths(i) for i in exp_]
    total_experience_in_months = sum(experience_num_list)
    return total_experience_in_months


def getTotalExperienceFormatted(exp_list) -> str:
    months = getTotalExperience(exp_list)
    if months < 12:
        return str(months) + " months"
    years = months // 12
    months = months % 12
    return str(years) + " years " + str(months) + " months"

# ---------------------------------------------------


#-------------Model--------------------------------
def test(self, filename: str):
        """
        Test a document and print the extracted information and rating
        :param filename: name of resume file
        :param info_extractor: InfoExtractor object
        """
        if self.model is None:
            print("model is not loaded or trained yet")
        doc, _ = loadDocumentIntoSpacy(filename, self.parser, self.nlp)

        print("Getting rating...")
        if self._type == "fixed":
            print("working on fixed model")
            if self.keywords is None:
                print("Keywords not found")

            seen_chunks_words, all_tokens_chunks = getAllTokensAndChunks(doc)

            # scoring
            temp_out = self.__trainKMWM(list(seen_chunks_words), list(all_tokens_chunks), self.keywords)
            if temp_out is None:
                print("Either parser cannot detect text or too few words in resume for analysis. Most usually the former." )
            km_scores, wm_scores = temp_out
            # average of km/wm scores for all keywords
            km_score = np.mean(km_scores)
            wm_score = np.mean(wm_scores)
            final_score = km_score * wm_score
        elif self._type == "lda":
            if self.lda is None or self.dictionary is None or self.top_k_words is None:
                print("No LDA found")

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
        x=extractFromText(filename)
        if x is not None:
            print("-" * 20)
            # info_extractor.extractFromFile(filename)
            skill=extractFromFile(filename)
            print("Skill:----",skill)
            print("-" * 20)
        print("Rating: %.1f" % rating)
        # if info_extractor is not None:
        #     print("info extractor is not working")
        #     env = os.environ
        #     subprocess.call([sys.executable, filename], env=env)
#---------------------------------------------------

def customFilter(token):
    customized_stop_words = [
        "work",
        "work experience",
        "work history",
        "experience",
        "education",
        "mobile",
        "phone no.",
        "phone number",
        "email",
        "number",
        "num",
        "professional",
        "career",
        "history",
        "histories",
        "skill",
        "skills",
        "activity",
        "activities",
        "curriculum",
        "tool",
        "tools",
        "language",
        "languages",
        "profile",
        "qualification",
        "qualifications",
        "certificate",
        "certificates",
        "certifications",
        "certification",
        "information",
        "intern",
        "internship",
        "volunteer",
        "award",
        "awards",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
    ]
    token_lower = token.lemma_.lower()
    return all([
            not token.is_space,  # not space
            not token.is_punct,  # not punct
            not token.is_bracket,  # not bracket
            not token.is_quote,  # not quote
            not token.is_currency,  # not currency
            not token.like_num,  # not number
            not token.like_url,  # not url
            not token.like_email,  # not email
            not token.is_oov,  # not out of vocab
            not token.is_stop,  # not a stopword
            token.is_alpha,  # is alphabetical
            token.has_vector,
            token_lower not in customized_stop_words,
            token.pos_ not in ["ADV", "ADJ", "INTJ", "PART", "PRON", "X"],
            token.ent_type_
            not in ["PERSON", "ORG", "DATE","CARDINAL","TIME",],  # not amongst these categories
        ])

