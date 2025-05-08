from collections import Counter
from sentence_transformers import SentenceTransformer, util
import ijson
import json
import math
import os
import spacy


# english natural language processor made by spaCy
nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])
# natural language LLM
model = SentenceTransformer("all-MiniLM-L6-v2")

"""
Go through all of the wiki files and process them into JSON. For each article,
extract the title and convert the rest into a cleaned list of tokens.
Args:
    source - The directory where the wiki files are stored
    dir_path - The directory in which to store the output files
Returns: Number of articles processed
"""
def wiki_files_to_json(source, dir_path):
    # A list to store all article data as dictionaries
    articles = []
    # current document ID
    doc_id = 0
    # current file number
    file_num = 0
    # create a directory to stored the cleaned files
    os.makedirs(dir_path, exist_ok=True)

    # loop through all wiki files
    for filename in os.listdir(source):
        filepath = os.path.join(source, filename)
        print("Reading file " + filename + "...")
        with open(filepath, "r", encoding='utf-8') as f:
            title = None
            body_lines = []
            categories = []

            # loop through each file line by line
            for line in f:
                line = line.strip()

                if line:
                    # if an article title is encountered
                    if line.startswith("[[") and line.endswith("]]"):
                        # we already have a title
                        if title:
                            raw_body = " ".join(body_lines)
                            
                            # gets the lemma of each word
                            body = nlp(raw_body)
                            # handles removing punctuation, extra spaces and stop words
                            body = [
                                token.lemma_.lower()
                                for token in body
                                if not token.is_punct and not token.is_space and not token.is_stop 
                                    and not token.lemma_.startswith("http") and not token.lemma_.startswith("www.")
                            ]

                            # add current article to list
                            articles.append({"id": doc_id, "title": title, "body": body, "raw": raw_body, "categories": categories})
                            doc_id += 1
                            # reset title
                            title = None
                        body_lines = []
                        title = line[2:-2]
                    # ignore redirect articles
                    elif "#redirect" in line.lower():
                        title = None
                        continue
                    # store categories
                    elif line.startswith("CATEGORIES:"):
                        categories += line[12:].split(",")
                    # ignore section headers
                    elif not line.startswith('=='):
                        body_lines.append(line)
            # checks for edge case (last article in the file)
            if title:
                raw_body = " ".join(body_lines)

                # gets the lemma of each word
                body = nlp(raw_body)
                # handles removing punctuation, extra spaces and stop words
                body = [
                    token.lemma_.lower()
                    for token in body
                    if not token.is_punct and not token.is_space and not token.is_stop 
                        and not token.lemma_.startswith("http") and not token.lemma_.startswith("www.")
                ]

                articles.append({"id": doc_id, "title": title, "body": body, "raw": raw_body, "categories": categories})
                doc_id += 1

        # write all articles to an output JSON file
        with open(f"{dir_path}/cleaned{file_num:02}.json", "w", encoding='utf-8') as out_f:
            json.dump(articles, out_f, ensure_ascii=False, indent=4)
        articles = []
        file_num += 1

    print("All done :)", doc_id, "articles processed")
    return doc_id


"""
Build a TF-IDF index from all of the article data. This uses the LNC LTN
formulas from Homework 3.
Args: json_dir - The directory where the JSON files are stored
Returns: A tf-idf index created from the JSON data, a dictionary mapping
    document IDs to article titles, a mapping of doc IDs to cleaned body tokens,
    a mapping of doc IDs to raw body text, and a mapping of doc IDs to a list
    of cleaned category tokens
"""
def build_index(json_dir):
    # make sure user has the json files
    if not os.path.isdir(json_dir):
        print(f"ERROR: directory {json_dir} does not exist")
        exit()
    if len(os.listdir(json_dir)) != 80:
        print(f"ERROR: user has not downloaded all JSON files into directory {json_dir}")
        exit()

    # tf-idf index (dict of dicts);  doc_id -> term -> tf-idf
    tfidf = {}
    # dict mapping doc IDs to article titles
    titles = {}
    # dict holding unprocessed document text
    raw = {}
    # dict holding list of processed token from document
    docs = {}
    # dict holding categories for each document
    # cats = {}

    # extract article information from all JSON files 
    for filename in os.listdir(json_dir):
        filepath = os.path.join(json_dir, filename)
        print("Reading file " + filename + "...")
        with open(filepath, "r", encoding='utf-8') as f:
            # loop through every document in the current file
            for doc in ijson.items(f, "item"):
                # keep track of the scores for each term
                scores = {}

                # collections.Counter counts the term frequency of unique terms
                count_freq = Counter(doc["body"])

                # compute log term frequency and count normalization sum
                norm = 0
                for term, tf in count_freq.items():
                    tf_log = 1 + math.log(tf, 10)
                    scores[term] = tf_log
                    norm += tf_log ** 2
                
                # compute cosine normalization
                norm = math.sqrt(norm)
                for term in scores:
                    scores[term] /= norm

                # add current document information to index
                tfidf[doc["id"]] = scores
                titles[doc["id"]] = doc["title"]
                raw[doc["id"]] = doc["raw"]
                docs[doc["id"]] = " ".join(doc["body"])
                # c = [c.strip().split() for c in doc["categories"]]
                # cat = []
                # for sublist in c:
                #     for item in sublist:
                #         if item not in nlp.Defaults.stop_words:
                #             cat.append(item)
                # cats[doc["id"]] = cat

    return tfidf, titles, docs, raw, None # removed categories


"""
Processes the 100 Jeopardy! questions from the given file. Extracts the
answer(s) and gets the cleaned lemma tokens of the question text.
Args: q_file - filename where the questions are stored
Returns: a list of dictionaries containing the cleaned questions, 
    their category, and their answer(s)
"""
def process_questions(q_file):
    cleaned_questions = []
    # read the questions file
    with open(q_file, "r", encoding='utf-8') as f:
        # questions are always separated by two newlines
        questions = f.read().split("\n\n")
        # loop through each question
        for q in questions:
            if q:
                q = q.split("\n")
                # extract the third line containing the answer(s)
                answers = q.pop(-1).split("|")

                # process the question text
                category = q.pop(0)
                original_question = q.pop(0)
                category_doc = nlp(category)

                # get the lemma of the category
                category_cleaned = []
                for token in category_doc:
                    # ignore the "Alex" blurbs
                    if token.text == "(":
                        break
                    if not token.is_punct and not token.is_space and not token.is_stop:
                        category_cleaned.append(token.lemma_.lower())

                cleaned = nlp(original_question)

                # remove punctuation, spaces, and stopwords
                cleaned = [
                    token.lemma_.lower()
                    for token in cleaned
                    if not token.is_punct and not token.is_space and not token.is_stop 
                ]

                # add cleaned question to list of questions
                cleaned_questions.append({"question": cleaned, "answer": answers, "original": original_question, "category": category_cleaned})
    return cleaned_questions
    

"""
Computes the tf-idf score for each question and get the top X articles based on
that score.
Args:
    questions - list of dicts containing question tokens and expected answers
    tfidf - TF-IDF index containing scores for each document
    titles - dictionary mapping doc IDs to article titles
    N - total number of documents in the vocabulary
    num - cutoff for number of titles to return. "Return top 'num' results"
    docs - dict mapping doc IDs to cleaned body tokens
    raw - dict mapping doc IDs to raw body text
    categories - dict mapping doc IDs to list of cleaned category tokens
"""
def answer_questions(questions, tfidf, titles, N, num, docs, raw, categories):
    # keep track of the correct answers and their positions
    positions = []
    positions1 = []
    hit_count = 0
    hit_count1 = 0
    # loop through each question
    for question in questions:
        # q_scores_category = {}

        # q_freq_category = Counter(question["category"])

        # for term, tf in q_freq_category.items():
        #     # log term frequency
        #     tf_log = 1 + math.log(tf, 10)
        #     # document frequency
        #     df = 0
        #     for doc in tfidf:
        #         if term in tfidf[doc]:
        #             df += 1
        #     # inverse document frequency
        #     if df > 0:
        #         idf = math.log(N/df, 10)
        #         # store tf-idf score
        #         q_scores_category[term] = tf_log * idf

        # category_scores = {}
        # for doc_id in tfidf:
        #     score = 0
        #     for term, weight in q_scores_category.items():
        #         if term in tfidf[doc_id]:
        #             score += weight * tfidf[doc_id][term]
        #     category_scores[doc_id] = score

        # # sort by score (descending)
        # sorted_scores = sorted(category_scores.items(), key=lambda item: item[1], reverse=True)

        # category_top_docs = [
        #     doc_id for doc_id, score in sorted_scores
        #     if score > 0
        # ]

        #----------------------------------------

        # post_category_reduction = int(len(tfidf) * .5)
        # category_top_docs = [doc_id for doc_id, _ in sorted_scores[:post_category_reduction]]

        # keep track of the scores for each term
        q_scores = {}

        # collections.Counter counts the term frequency of the query
        q_freq = Counter(question["question"])

        # compute tf-idf score for question
        for term, tf in q_freq.items():
            # log term frequency
            tf_log = 1 + math.log(tf, 10)
            # document frequency
            df = 0
            for doc in tfidf:
                if term in tfidf[doc]:
                    df += 1
            # inverse document frequency
            if df > 0:
                idf = math.log(N/df, 10)
                # store tf-idf score
                q_scores[term] = tf_log * idf

        # compute score for documents
        question_scores = {}
        for doc_id in tfidf:
            score = 0
            for term, weight in q_scores.items():
                if term in tfidf[doc_id]:
                    score += weight * tfidf[doc_id][term]
            question_scores[doc_id] = score

        # sort by score (descending)
        sorted_scores = sorted(question_scores.items(), key=lambda item: item[1], reverse=True)

        # get top 500
        check1 = [titles[doc_id] for doc_id, _ in sorted_scores[:500]]

        # remove docs with score of 0
        question_top_docs = [
            doc_id for doc_id, score in sorted_scores
            if score > 0
        ]

        # get top 100 nonzero
        question_top_docs = question_top_docs[:1000]

        # post_question_reduction = int(len(category_top_docs) * 1)
        # question_top_docs = [doc_id for doc_id, _ in sorted_scores[:num]]

        # final_scores = {}
        # for doc_id in tfidf:
        #     final_scores[doc_id] = (.2 * category_scores[doc_id]) + (.8 * question_scores[doc_id])
        # sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        # combined_top_docs = [doc_id for doc_id, _ in sorted_scores[:num]]

        # potential_doc_ids = list(set(combined_top_docs + question_top_docs + category_top_docs))
        potential_top_docs = [raw[doc_id] for doc_id in question_top_docs]

        # rerank top 1000 documents using an LLM cosine similarity
        question_embedding = model.encode(question["original"], convert_to_tensor=True)
        doc_embeddings = model.encode(potential_top_docs, convert_to_tensor=True)

        scores = util.cos_sim(question_embedding, doc_embeddings)[0]
        rerank_scores = {doc_id: float(score) for doc_id, score in zip(question_top_docs, scores)}

        reranked = sorted(rerank_scores.items(), key=lambda x: x[1], reverse=True)
        top_docs = [titles[doc_id] for doc_id, _ in reranked[:num]]

        # tfidf_scores = [score for _, score in sorted_scores]

        # max_tfidf = max(tfidf_scores)
        # min_tfidf = min(tfidf_scores)
        # normalized = {
        #     doc_id: (score - min_tfidf) / (max_tfidf - min_tfidf + 1e-8)
        #     for (doc_id, score) in sorted_scores
        # }

        # weight = 0.5

        # final_scores = {
        #     doc_id: weight * normalized.get(doc_id, 0) + (1 - weight) * rerank_scores.get(doc_id, 0)
        #     for doc_id in question_top_docs
        # }

        # get the titles for the associated doc IDs
        query_answers = top_docs # [titles[doc_id] for doc_id, _, _ in top_reranked]
        print(f"Getting top {num} results for category: \"{question["category"]}\", question \"{question["original"]}\":")

        # determine if the correct answer was in the top X results
        found = False
        for i, title in enumerate(check1):
            if title in question["answer"]:
                print(f"HIT: correct article title \"{title}\" appears in top {num} results at position {i+1} for tfidf")
                positions1.append(i+1)
                found = True
                hit_count1 += 1
                break
        if not found:
            print(f"MISS: Expected answer \"{question["answer"]}\" not found in top {num} results for tfidf")
            positions1.append(None)
        
        # determine if the correct answer was in the top X results
        found = False
        for i, title in enumerate(query_answers):
            if title in question["answer"]:
                print(f"HIT: correct article title \"{title}\" appears in top {num} results at position {i+1} for reranking")
                positions.append(i+1)
                found = True
                hit_count += 1
                break
        if not found:
            print(f"MISS: Expected answer \"{question["answer"]}\" not found in top {num} results for reranking")
            positions.append(None)
        
        print("~~~~~")
    
    print(f"Final score: {hit_count}/100, {hit_count}% hit rate for reranking")
    print(f"Final score: {hit_count1}/100, {hit_count1}% hit rate for tfidf")

    # shows the hit distribution for reranking from 1 to 500
    t = {1:0, 20:0, 50:0, 100:0, 250:0, 500:0}
    for pos in positions:
        if pos:
            if pos == 1:
                t[1] += 1
            elif pos <= 20:
                t[20] += 1
            elif pos <= 50:
                t[50] += 1
            elif pos <= 100:
                t[100] += 1
            elif pos <= 250:
                t[250] += 1
            elif pos <= 500:
                t[500] += 1
    print(t)

    # shows the hit distribution for just tf-idf from 1 to 500
    t = {1:0, 20:0, 50:0, 100:0, 250:0, 500:0}
    for pos in positions1:
        if pos:
            if pos == 1:
                t[1] += 1
            elif pos <= 20:
                t[20] += 1
            elif pos <= 50:
                t[50] += 1
            elif pos <= 100:
                t[100] += 1
            elif pos <= 250:
                t[250] += 1
            elif pos <= 500:
                t[500] += 1
    print(t)


def main():
    #### DO NOT RUN THIS ####
    #### This was used to create the cleaned JSON files
    #### It only needed to be ran once
    print("Processing raw wiki files...")
    # N = wiki_files_to_json("articles", "cleaned_articles")
    N = 135572
    print("Done!", N, "articles processed")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Construct the tf-idf index
    print("Constructing the tf-idf index...")
    tfidf_index, articles_dict, docs, raw, cats = build_index("cleaned_articles")
    print("Done!", len(articles_dict), "document scores computed")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Process the Jeopardy questions
    print("Processing the jeopardy questions...")
    questions = process_questions("questions.txt")
    print("Done!", len(questions), "questions processed")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    # Query the index with the questions and get the results
    print("Querying the index for question answers...")
    answer_questions(questions, tfidf_index, articles_dict, N, 500, docs, raw, cats)
    print("All done!")


if __name__ == "__main__":
    main()
