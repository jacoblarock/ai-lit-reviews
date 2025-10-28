import requests
import ollama
import json
from arxiv import Client, Search, Result
import pickle
import os
import shutil
from bs4 import BeautifulSoup
import sys
from latexcompiler import LC

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

def generate_biblatex_entry(result: Result, key: str) -> str:
    # Extract key components
    eprint = result.entry_id.split('/')[-1]
    # authors = ", ".join([a.name for a in result.authors])
    authors = f"{result.authors[0]} et al."
    try:
        year = eprint.split('.')[0][:4]
    except Exception:
        year = "????"
    entry = (
        f"@article{{{key},\n"
        f"  title = {{{result.title}}},\n"
        f"  author = {{{authors}}},\n"
        f"  eprint = {{{eprint}}},\n"
        f"  eprinttype = {{arxiv}},\n"
        f"  year = {{{year}}},\n"
        f"  doi = {{{result.doi}}}\n"
        f"}}"
    )
    return entry

def create_bibliography(sources: list[Result]) -> str:
    out = ""
    for i, res in enumerate(sources):
        out += generate_biblatex_entry(res, str(i))
        out += "\n"
    return out

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

def make_methods_section(used_queries: list[str], abstract_filtered: list[Result], topic: str) -> str:
    print("making methods section")
    article_titles = [x.title for x in abstract_filtered]
    with open("prompts/make_methods_section.txt", "r", encoding="utf-8") as file:
        prompt = file.read()\
            .replace("QUERIES", str(used_queries))\
            .replace("TITLES", str(article_titles))\
            .replace("TOPIC", topic)
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
    )
    c = resp["message"]["content"]
    return c[c.find("</think>")+8:]

def prepare_subsection_articles(abstract_filtered: list[Result], subsection_contains: list[int]) -> dict[int,Result]:
    subsection_articles: dict[int,Result] = {}
    for i in subsection_contains:
        subsection_articles[i] = abstract_filtered[i]
    return subsection_articles

def summarize_article(article_contents: str, topic: str, section_name: str):
    with open("prompts/summarize_article.txt", encoding="utf-8") as file:
        prompt = file.read().replace("TOPIC", topic).replace("SECTION", section_name).replace("ARTICLE", article_contents)
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    c = resp["message"]["content"]
    return c[c.find("</think>")+8:]

def make_subsection_summaries(subsection_articles: dict[int,Result], topic: str, section_name: str) -> dict[int,str]:
    print("making summaries for subsection: " + section_name)
    article_summaries = {}
    for i in subsection_articles.keys():
        article = subsection_articles[i]
        short_id = article.get_short_id()
        article_contents_raw = requests.get(
            f"https://arxiv.org/html/{short_id}"
        ).content
        soup = BeautifulSoup(article_contents_raw, features="html.parser")
        article_elements = soup.find_all("p", {"class": "ltx_p"})
        if article_elements:
            article_contents ="\n".join([str(e) for e in article_elements])
        if not article_elements:
            print("skipping:", i)
            continue
        article_summaries[i] = summarize_article(article_contents, topic, section_name)
        print("summarized:", i)
    return article_summaries

def make_results_subsection(subsection_summaries: dict[int,str], topic: str, section_name: str) -> str:
    print("making subsection:", section_name)
    with open("prompts/make_results_subsection.txt", encoding="utf-8") as file:
        prompt = file.read().replace("TOPIC", topic).replace("SECTION", section_name).replace("SUMMARIES", str(subsection_summaries))
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    c = resp["message"]["content"]
    return c[c.find("</think>")+8:]

def make_results_intro(results_subsections: dict[str,str], topic: str) -> str:
    print("making results introduction")
    joined_subsections = "\n".join(results_subsections.values())
    with open("prompts/make_results_intro.txt", encoding="utf-8") as file:
        prompt = file.read().replace("TOPIC", topic).replace("SUBSECTIONS", str(joined_subsections))
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    c = resp["message"]["content"]
    return c[c.find("</think>")+8:]

def make_discussion_limitations(results_intro: str, results_subsections: dict[str,str], topic: str) -> str:
    print("discussion limitations")
    results_section = "\n".join(
        [results_intro] + [x for x in results_subsections.values()]
    )
    with open("prompts/discussion_limitations.txt", encoding="utf-8") as file:
        prompt = file.read().replace("RESULTS", results_section).replace("TOPIC", str(topic))
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    c = resp["message"]["content"]
    return c[c.find("</think>")+8:]

def make_discussion_future_directions(results_intro: str, results_subsections: dict[str,str], topic: str) -> str:
    print("discussion future directions")
    results_section = "\n".join(
        [results_intro] + [x for x in results_subsections.values()]
    )
    with open("prompts/discussion_future_directions.txt", encoding="utf-8") as file:
        prompt = file.read().replace("RESULTS", results_section).replace("TOPIC", str(topic))
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    c = resp["message"]["content"]
    return c[c.find("</think>")+8:]

def make_discussion_intro(discussion_subsections: list[str], topic: str) -> str:
    print("making discussion introduction")
    joined_subsections = "\n".join(discussion_subsections)
    with open("prompts/make_discussion_intro.txt", encoding="utf-8") as file:
        prompt = file.read().replace("TOPIC", topic).replace("SUBSECTIONS", str(joined_subsections))
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    c = resp["message"]["content"]
    return c[c.find("</think>")+8:]


def compile(sections: list[str], title: str, author: str):
    with open("templates/main.tex", "r", encoding="utf-8") as file:
        template = file.read()
    out = template
    out = out.replace("TITLE", title)
    out = out.replace("AUTHOR", author)
    out = out.replace("SECTIONS", "\n".join(sections))
    if not os.path.isdir("temp/result/"):
        os.mkdir("temp/result/")
    with open("temp/result/main.tex", "w", encoding="utf-8") as file:
        file.write(out)
    shutil.copyfile("temp/citations.bib", "temp/result/citations.bib")
    os.chdir("temp/result/")
    LC.compile_document(
        tex_engine="pdflatex",
        bib_engine="bibtex",
        no_bib=True,
        path="main.tex",
        folder_name="compile"
    )

def main():
    topic = "Explainable AI in the area of audio deepfake detection."
    if not os.path.isdir("temp/"):
        os.mkdir("temp/")
    used_queries = []
    # create queries and search for articles until there are at least 50 results for filtering
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
    # filter articles by the titles and abstracts for their relevance to the topic
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
            abstract_filtered: list[Result] = pickle.load(file)
    if not os.path.isfile("temp/citations.bib"):
        with open("temp/citations.bib", "w", encoding="utf-8") as file:
            bibliography = create_bibliography(abstract_filtered)
            file.write(bibliography)
    else:
        with open("temp/citations.bib", "r", encoding="utf-8") as file:
            bibliography = file.read()
    # generate the methods section based on previous results
    if not os.path.isfile("temp/methods.txt"):
        methods = make_methods_section(used_queries, abstract_filtered, topic)
        with open("temp/methods.txt", "w", encoding="utf-8") as file:
            file.write(methods)
    else:
        with open("temp/methods.txt", "r", encoding="utf-8") as file:
            methods = file.read()
    # categorize the articles into subsections of the results section
    if not os.path.isfile("temp/article_categories.pkl"):
        categories = determine_article_categories(abstract_filtered, topic)
        with open("temp/article_categories.pkl", "wb") as file:
            pickle.dump(categories, file)
    else:
        with open("temp/article_categories.pkl", "rb") as file:
            categories = pickle.load(file)
    # create summaries of articles to prepare for the results section
    sections_subsections_summaries: dict[str,dict[int,str]] = {}
    for i, category in enumerate(categories.keys()):
        if not os.path.isfile(f"temp/subsection_summaries_{i}.pkl"):
            subsection_articles = prepare_subsection_articles(abstract_filtered, categories[category])
            subsection_summaries = make_subsection_summaries(subsection_articles, topic, category)
            with open(f"temp/subsection_summaries_{i}.pkl", "wb") as file:
                pickle.dump(subsection_summaries, file)
        else:
            with open(f"temp/subsection_summaries_{i}.pkl", "rb") as file:
                subsection_summaries = pickle.load(file)
        sections_subsections_summaries[category] = subsection_summaries
    results_subsections: dict[str,str] = {}
    for i, category in enumerate(categories.keys()):
        if not os.path.isfile(f"temp/subsection_{i}.txt"):
            subsection_contents = make_results_subsection(
                sections_subsections_summaries[category],
                topic,
                category
            )
            with open(f"temp/subsection_{i}.txt", "w", encoding="utf-8") as file:
                file.write(subsection_contents)
        else:
            with open(f"temp/subsection_{i}.txt", "r", encoding="utf-8") as file:
                subsection_contents = file.read()
        results_subsections[category] = subsection_contents
    if not os.path.isfile(f"temp/results_intro.txt"):
        results_intro = make_results_intro(results_subsections, topic)
        with open(f"temp/results_intro.txt", "w", encoding="utf-8") as file:
            file.write(results_intro)
    else:
        with open(f"temp/results_intro.txt", "r", encoding="utf-8") as file:
            results_intro = file.read()
    if not os.path.isfile(f"temp/discussion_limitations.txt"):
        discussion_limitations = make_discussion_limitations(results_intro, results_subsections, topic)
        with open(f"temp/discussion_limitations.txt", "w", encoding="utf-8") as file:
            file.write(discussion_limitations)
    else:
        with open(f"temp/discussion_limitations.txt", "r", encoding="utf-8") as file:
            discussion_limitations = file.read()
    if not os.path.isfile(f"temp/discussion_future_directions.txt"):
        discussion_future_directions = make_discussion_future_directions(results_intro, results_subsections, topic)
        with open(f"temp/discussion_future_directions.txt", "w", encoding="utf-8") as file:
            file.write(discussion_future_directions)
    else:
        with open(f"temp/discussion_future_directions.txt", "r", encoding="utf-8") as file:
            discussion_future_directions = file.read()
    if not os.path.isfile(f"temp/discussion_intro.txt"):
        discussion_intro = make_discussion_intro([discussion_limitations, discussion_future_directions], topic)
        with open(f"temp/discussion_intro.txt", "w", encoding="utf-8") as file:
            file.write(discussion_intro)
    else:
        with open(f"temp/discussion_intro.txt", "r", encoding="utf-8") as file:
            discussion_intro = file.read()
    document_sections = [
        methods,
        results_intro
    ] + [
        x for x in results_subsections.values()
    ] + [
        discussion_intro,
        discussion_limitations,
        discussion_future_directions,
    ]
    compile(
        document_sections,
        topic,
        "Bot"
    )


if __name__ == "__main__":
    main()