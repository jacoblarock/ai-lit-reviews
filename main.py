import requests
import xmltodict
import ollama
import json
from arxiv import Client, Search, Result
import pickle
import os

model = "qwen3:14b"

def create_queries(topic: str) -> list[str]:
    with open("prompts/create_queries.txt", encoding="utf-8") as file:
        prompt = file.read().replace("TOPIC", topic)
    while True:
        resp = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        try:
            content = resp["message"]["content"]
            start_index = content.rfind("[")
            end_index = content.rfind("]") + 1
            content = content[start_index:end_index]
            return json.loads(content)
        except:
            print("retrying")

def query_articles(search_terms: list[str], previous_results: list[Result]) -> list[Result]:
    client = Client()
    results = []
    for term in search_terms:
        print(term)
        search = Search(
            query=term,
            max_results=10
        )
        resp = client.results(search)
        for result in resp:
            if result.entry_id in [x.entry_id for x in previous_results]:
                continue
            results.append(result)
    return results

def deduplicate_results(results: list[Result]) -> list[Result]:
    seen = set()
    unique = []
    for result in results:
        if result.entry_id not in seen:
            seen.add(result.entry_id)
            unique.append(result)
    return unique

def assess_article_by_abstract(article: Result, topic: str) -> bool:
    title = article.title
    abstract = article.summary
    with open("prompts/assess_article_by_abstract.txt", encoding="utf-8") as file:
        prompt = file.read().replace("TOPIC", topic).replace("TITLE", title).replace("ABSTRACT", abstract)
    while True:
        resp = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        answer = resp["message"]["content"]
        if answer[-3:] == "yes":
            return True
        elif answer[-2:] == "no":
            return False
        print(answer)
        print("retrying")

def write_articles_to_file(articles: list[Result], filename: str) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for idx, article in enumerate(articles, start=1):
            f.write(f"Article {idx}\n")
            f.write(f"Title: {article.title}\n")
            f.write(f"Authors: {', '.join([author.name for author in article.authors])}\n")
            f.write(f"Link: {article.entry_id}\n")
            f.write(f"DOI: {article.doi}\n") 
            f.write("Abstract:\n")
            f.write(f"{article.summary}\n")
            f.write("-" * 80 + "\n\n")

def determine_article_categories(approved_articles: list[Result], topic: str) -> dict[str,list]:
    print("Determing article categories")
    abstracts = [(i, article.summary) for i, article in enumerate(approved_articles)]
    with open("prompts/determine_article_categories.txt", encoding="utf-8") as file:
        prompt = file.read().replace("TOPIC", topic).replace("ABSTRACTS", json.dumps(abstracts))
    while True:
        resp = ollama.chat(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        try:
            content = resp["message"]["content"]
            start_index = content.rfind("{")
            end_index = content.rfind("}") + 1
            content = content[start_index:end_index]
            return json.loads(content)
        except:
            print("retrying")

def main():
    topic = "Explainable AI in the area of audio deepfake detection."
    used_queries = []
    if not os.path.isfile("temp/article_metadata.pkl") or not os.path.isfile("temp/queries.pkl"):
        article_metadata = []
        while len(article_metadata) < 50:
            queries = create_queries(topic)
            used_queries += queries
            article_metadata += query_articles(queries, article_metadata)
            print(len(article_metadata))
        article_metadata = deduplicate_results(article_metadata)
        print(len(article_metadata))
        with open("temp/article_metadata.pkl", "wb") as file:
            pickle.dump(article_metadata, file)
        with open("temp/queries.pkl", "wb") as file:
            pickle.dump(used_queries, file)
    else:
        with open("temp/article_metadata.pkl", "rb") as file:
            article_metadata = pickle.load(file)
        with open("temp/queries.pkl", "rb") as file:
            used_queries = pickle.load(file)
    if not os.path.isfile("temp/abstract_filtered.pkl"):
        abstract_filtered = []
        for article in article_metadata:
            article_res = assess_article_by_abstract(article, topic)
            if article_res:
                print(article.title)
                print("article approved")
                abstract_filtered.append(article)
            else:
                print(article.title)
                print("article filtered out")
        with open("temp/abstract_filtered.pkl", "wb") as file:
            pickle.dump(abstract_filtered, file)
    else:
        with open("temp/abstract_filtered.pkl", "rb") as file:
            abstract_filtered = pickle.load(file)
    if not os.path.isfile("temp/article_categories.pkl"):
        categories = determine_article_categories(abstract_filtered, topic)
        with open("temp/article_categories.pkl", "wb") as file:
            pickle.dump(categories, file)
    else:
        with open("temp/article_categories.pkl", "rb") as file:
            categories = pickle.load(file)
    print(categories)

if __name__ == "__main__":
    main()