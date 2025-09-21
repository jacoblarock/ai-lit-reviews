# AI Literature Reviews
This project attempts to programmatically guide a large language model (LLM) through the process of creating a systematic literature review, leveraging the reasoning capabilities of state-of-the art models .
## Methods
Currently, the following steps are implemented or planned:
- Searching for relevant articles using generated queries in the ArXiv database.
- Filtering of articles for their relevance for inclusion in the study.
- Generation of the methods section with the previous steps taken as context.
- Systematic categorization of the filtered articles for use in subsections in the review.
- Contextual article summarization as preprocessing before the results section.
- Segmented generation of the results section using the previous steps as context.
- Evaluation of potential subsections for the discussion section.
- Segmented generation of the discussion subsections.
- With the discussion as context, the extrapolation of a conclusion.
The use of locally hosted LLMs is made possible with the Ollama library.