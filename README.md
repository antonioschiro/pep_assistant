![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

# PEP Assistant

PEP Assistant is a Python-based tool for making your code compliant to Python Enhancement Proposals (PEPs) and to the Python common best practices.
It uses Retrieval Augmentated Generation (RAG) to get PEP rules and useful python code examples.

## Features

- **PEP Scraping:** Downloads and stores the content of selected PEPs.
- **GitHub Scraping:** Recursively scrapes Python source files from Flask and Python standard library repos.
- **Chunking:** Splits documents and code into manageable chunks using configurable strategies (classic, Python-aware, semantic
    (optional, not used as default)).
- **Vector Database:** Embeds and stores chunks in two ChromaDB vectorstores for a faster and clean similarity search.
    The first one contains chunks from PEP webpages scraping. The last one contains chunks from the python libraries.
    The source of each chunk, by the way, is stored into metadatas.
- **Extensible:** Easily add new sources or chunking strategies.
- **Self RAG Agent:** The agent can evaluate its own generated code and decide if it's hallucinated, uncomplete or complete. 
    How it works:
    - If the answer is not complete, it will try to regenerate the answer with the same retrieved documents.
    - If the answer is hallucinated, it will repeat the documents retrieving.
    - A maximum number of iterations is provided in order to avoid infinite loops.
- **Custom templates:** Homemade templates to improve retrieval and generation.
- **Performance evaluation and testing:** Tests the agent via Langsmith by simply run a script. 

## Project Structure

```
.env
app.py
requirements.txt
rag_folder/
    codes_folder/
        chroma.sqlite3
        flask_codes.txt
        standardlib_codes.txt
        ...
    pep_folder/
        chroma.sqlite3
        pep008_document.txt
        ...
utils/
    graph.py
    nodes.py
    scripts.py
    templates.py
tests/
    data_collection.py
    datasets.py
    run_tests.py
```

## Setup

1. **Clone the repository:**
    ```sh
    git clone <your-repo-url>
    cd pep_assistant
    ```

2. **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Set environment variables:**
    - Create a `.env` file with the following (adjust the path as needed):
      ```
      absolute_path=/absolute/path/to/your/project/root
      ```

4. **Run the scripts:**
    - The main logic is in [`utils/scripts.py`](utils/scripts.py). Running this script will:
        - Scrape PEPs and code repositories if vectorstores do not exist.
        - Chunk and embed the documents.
        - Store them in ChromaDB vectorstores.

    ```sh
    python utils/scripts.py
    ```

## Usage

- **PEP Scraping:** Downloads and stores PEP documents in `rag_folder/pep_folder/`.
- **Code Scraping:** Downloads and stores code from Flask, scikit-learn, and the Python standard library in `rag_folder/codes_folder/`.
- **Vectorstore Creation:** Chunks and embeds documents, storing them in `chroma.sqlite3` files for fast retrieval.
- **Vectorstore Creation:** Chunks and embeds documents, storing them in `chroma.sqlite3` files for fast retrieval.
- **Run the agent:** Fix the style of your code by running `app.py` file.

## Customization

- **Add new PEPs:** Edit the `pep_urls` list in [`utils/scripts.py`](utils/scripts.py).
- **Add new repositories:** Modify the URLs for `flask_url`, `sklearn_url`, or add new ones.
- **Change chunking strategy:** Adjust the `SplitterType` in the `create_chunks` and `create_vectorstore` functions.
- **Edit templates:** Changing the custom templates to adjust the documents retrieval and the response generation.
- **Add new tests:** Add more test code snippets to evaluate the agent.

## Testing

Test scripts are located in the [`tests/`](tests/) directory.

## Dependencies

- Python 3.8+
- Selenium
- BeautifulSoup4
- langchain
- chromadb
- huggingface-hub
- dotenv
- webdriver-manager

See [`requirements.txt`](requirements.txt) for the full list.

## License

[MIT License](LICENSE)