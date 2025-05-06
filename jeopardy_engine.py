from collections import Counter
import json
import math
import os
import spacy


# english natural language processor made by spaCy
nlp = spacy.load('en_core_web_sm', disable=["ner", "parser"])

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

            # loop through each file line by line
            for line in f:
                line = line.strip()

                if line:
                    # if an article title is encountered
                    if line.startswith("[[") and line.endswith("]]"):
                        # we already have a title
                        if title:
                            body = " ".join(body_lines)
                            
                            # gets the lemma of each word
                            body = nlp(body)
                            # handles removing punctuation, extra spaces and stop words
                            body = [
                                token.lemma_.lower()
                                for token in body
                                if not token.is_punct and not token.is_space and not token.is_stop 
                                    and not token.lemma_.startswith("http") and not token.lemma_.startswith("www.")
                                    and token.lemma_ not in "=|"
                            ]

                            # add current article to list
                            articles.append({"id": doc_id, "title": title, "body": body})
                            doc_id += 1
                            # reset title
                            title = None
                        body_lines = []
                        title = line[2:-2]
                    # ignore redirect articles
                    elif "#redirect" in line.lower():
                        title = None
                        continue
                    # ignore section headers
                    elif not line.startswith('=='):
                        body_lines.append(line)
            # checks for edge case (last article in the file)
            if title:
                body = " ".join(body_lines)

                # gets the lemma of each word
                body = nlp(body)
                # handles removing punctuation, extra spaces and stop words
                body = [
                    token.lemma_.lower()
                    for token in body
                    if not token.is_punct and not token.is_space and not token.is_stop 
                        and not token.lemma_.startswith("http") and not token.lemma_.startswith("www.")
                        and token.lemma_ not in "=|"
                ]

                articles.append({"id": doc_id, "title": title, "body": body})
                doc_id += 1

        # write all articles to an output JSON file
        with open(dir_path + "/cleaned" + str(file_num) + ".json", "w", encoding='utf-8') as out_f:
            json.dump(articles, out_f, ensure_ascii=False, indent=4)
        articles = []
        file_num += 1

    print("All done :)", doc_id, "articles processed")
    return doc_id


"""
Build a TF-IDF index from all of the article data. This uses the LNC LTN
formulas from Homework 3.
Args: json_dir - The directory where the JSON files are stored
Returns: A tf-idf index created from the JSON data and a dictionary mapping
    document IDs to article titles.
"""
def build_index(json_dir):
    # tf-idf index (dict of dicts);  doc_id -> term -> tf-idf
    tfidf = {}
    # dict mapping doc IDs to article titles
    titles = {}

    # extract article information from all JSON files 
    for filename in os.listdir(json_dir):
        filepath = os.path.join(json_dir, filename)
        print("Reading file " + filename + "...")
        with open(filepath, "r", encoding='utf-8') as f:
            data = json.load(f)

        # loop through every document in the current file
        for doc in data:
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

    return tfidf, titles


"""
Processes the 100 Jeopardy! questions from the given file. Extracts the
answer(s) and gets the cleaned lemma tokens of the question text.
Args: q_file - filename where the questions are stored
Returns: a list of dictionaries containing the cleaned questions and
    their answer(s)
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
                q = " ".join(q)
                original_question = q
                cleaned = nlp(q)
                # remove punctuation, spaces, and stopwords
                cleaned = [
                    token.lemma_.lower()
                    for token in cleaned
                    if not token.is_punct and not token.is_space and not token.is_stop 
                ]
                # add cleaned question to list of questions
                cleaned_questions.append({"question": cleaned, "answer": answers, "original": original_question})
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
"""
def answer_questions(questions, tfidf, titles, N, num):
    # keep track of the correct answers and their positions
    positions = []
    hit_count = 0
    # loop through each question
    for question in questions:
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
        final_scores = {}
        for doc_id in tfidf:
            score = 0
            for term, weight in q_scores.items():
                if term in tfidf[doc_id]:
                    score += weight * tfidf[doc_id][term]
            final_scores[doc_id] = score

        # sort by score (descending)
        sorted_scores = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        # slice out the top X scores
        sorted_scores = sorted_scores[:num]
        # get the titles for the associated doc IDs
        query_answers = [titles[doc_id] for doc_id, _ in sorted_scores]
        print(f"Getting top {num} results for question \"{question["original"]}\":")
        
        # determine if the correct answer was in the top X results
        found = False
        for i, title in enumerate(query_answers):
            if title in question["answer"]:
                print(f"HIT: correct article title \"{title}\" appears in top {num} results at position {i+1}")
                positions.append(i+1)
                found = True
                hit_count += 1
                break
        if not found:
            print(f"MISS: Expected answer \"{question["answer"]}\" not found in top {num} results")
            positions.append(None)
        
        print("~~~~~")
    print(positions)
    print(f"Final score: {hit_count}/100, {hit_count}% hit rate")

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




def main():
    #### DO NOT RUN THIS ####
    #### This was used to create the cleaned JSON files
    #### It only needed to be ran once
    print("Processing raw wiki files...")
    #N = wiki_files_to_json("articles", "cleaned_articles")
    N = 137471
    print("Done!", N, "articles processed")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Construct the tf-idf index
    print("Constructing the tf-idf index. This will take up to two minutes...")
    tfidf_index, articles_dict = build_index("cleaned_articles")
    print("Done!", len(articles_dict), "document scores computed")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # Process the Jeopardy questions
    print("Processing the jeopardy questions...")
    questions = process_questions("questions.txt")
    print("Done!", len(questions), "questions processed")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    # Query the index with the questions and get the results
    print("Querying the index for question answers...")
    answer_questions(questions, tfidf_index, articles_dict, N, 500)
    print("All done!")


if __name__ == "__main__":
    main()
