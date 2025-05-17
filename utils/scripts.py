# Import libraries
import os
import time
import traceback
from urllib.parse import urljoin
from dotenv import load_dotenv
load_dotenv()
from enum import Enum
# Libraries for scraping
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
# Libraries for preparing vector database
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import PythonCodeTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
# Set the key TOKENIZERS_PARALLELISM to avoid deadlocks
os.environ["TOKENIZERS_PARALLELISM"] = "false"
## Function to create Selenium driver.
def create_driver():
    # Set Chrome options for headless mode.
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    # Initialize driver.
    s = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=s, options=chrome_options)
# Global folders paths
absolute_path = os.getenv('absolute_path')
rag_path = os.path.join(absolute_path, 'pep_assistant/rag_folder')
pysnippet_folder = os.path.join(rag_path, 'codes_folder')

# PEP DATA RETRIEVE:
pep_urls = ['https://peps.python.org/pep-0484/',
            'https://peps.python.org/pep-0020/', 
            'https://peps.python.org/pep-0498/', 
            'https://peps.python.org/pep-0572/', 
            'https://peps.python.org/pep-0586/', 
            'https://peps.python.org/pep-0585/', 
            'https://peps.python.org/pep-0257/', 
            'https://peps.python.org/pep-0008/', 
            'https://peps.python.org/pep-0635/', 
            'https://peps.python.org/pep-0622/', 
            'https://peps.python.org/pep-0526/', 
            'https://peps.python.org/pep-0634/', 
            'https://peps.python.org/pep-0563/', 
            'https://peps.python.org/pep-0604/', 
            'https://peps.python.org/pep-0636/', 
            'https://peps.python.org/pep-0589/', 
            'https://peps.python.org/pep-0593/', 
            'https://peps.python.org/pep-0343/', 
            'https://peps.python.org/pep-0649/']

pep_file_names = [url.split('/')[-2].replace('-','') + '_document' for url in pep_urls]
pep_db_folder = os.path.join(rag_path, 'pep_folder')
## FLASK SOURCE CODE RETRIEVE:
flask_url = "https://github.com/pallets/flask/tree/main/src/flask"
flask_file_name = 'flask_codes'
# SK-LEARN SOURCE CODE RETRIEVE:
sklearn_url = "https://github.com/scikit-learn/scikit-learn/tree/main/sklearn"
sklearn_file_name = 'sklearn_codes'
# PYTHON STANDARD LIBRARY MODULES:
standardlib_url = "https://github.com/python/cpython/tree/main/Lib"
standardlib_file_name = 'standardlib_codes'

def scrape_pep_url(url:str, file_name:str, folder_path:str = pep_db_folder) -> None:
    '''This function allows to scrape any PEP webpage and store html content into a .txt file located in folder_path.'''
    if not file_name.endswith('.txt'):
        file_name += '.txt'
    file_path = os.path.join(folder_path, file_name)
    try:
        driver = create_driver()
        # Scrape site:
        driver.get(url)
        driver.implicitly_wait(5)
        html_pep = driver.page_source
        soup = BeautifulSoup(html_pep, 'html.parser')
        # Store the retrieved content in txt file.
        with open(file_path, 'w') as doc:
            tags = soup.find_all()
            doc.writelines([tag.text for tag in tags])
    except Exception as e:
        print(f'Exception occurred: {e}')
    finally:
        driver.quit()

# FUNCTION TO SCRAPE SOURCE CODES ON GITHUB REPOS
def get_github_urls(url: str, max_depth:int = 2)->list[str]:
    '''This function takes in input a GitHub repo URL and it returns the list of source files urls.'''
    urls_list = []
    driver = create_driver()
    def scraping_folder(scraping_url=url, depth=0, max_depth=max_depth):
        if depth > max_depth:
            return
        # Storing files name in list
        nonlocal urls_list
        driver.get(scraping_url)
        try:
            elements_list = WebDriverWait(driver, 20).until(EC.presence_of_all_elements_located((By.CLASS_NAME, "Link--primary")))
        except Exception as e:
            print(f"An error occurring when loading page elements: {e}")
        # Collecting aria labels and urls
        element_attributes = []
        for element in elements_list:
            href = element.get_attribute('href') # contains file relative path
            file_url = urljoin( "https://github.com"+ "/", href) 
            aria_label = element.get_attribute('aria-label')
            element_attributes.append((file_url, aria_label))
        # Iterating on all files inside repo
        for link, aria_label in element_attributes:
            try:
                if link.endswith('.py'):
                    urls_list.append(link)
                elif "(Directory)" in aria_label:
                    time.sleep(1)
                    scraping_folder(link, depth=depth+1)          
            except Exception as e:
                print(f'An exception occured while scraping files: {e}')
                continue
    try:
        scraping_folder()
    except Exception as e:
        print(f"An exception occured while calling {scraping_folder.__name__}:{e}")
    finally:
        driver.quit()
    print(f'Successfully downloaded {len(urls_list)} urls.')
    return urls_list

def scrape_github_urls(urls_list:list[str], file_name: str, folder_path:str=pysnippet_folder, max_attempt:int=3) -> None:
    '''This function retrieve Python code for each GitHub repo URL in urls_list and store their content into a .txt file.'''
    # Check filename. Adding extension if not present
    if not file_name.endswith('.txt'):
        file_name += '.txt'
    # Create director if not present
    os.makedirs(folder_path, exist_ok= True)
    # Create file
    file_path = os.path.join(folder_path, file_name)
    # Appending html content to txt file
    with open(file_path, 'a') as doc:
        for url in urls_list:
            for attempt in range(max_attempt):
                try:
                    driver = create_driver()
                    driver.get(url)
                    element = WebDriverWait(driver,20).until(EC.presence_of_element_located((By.ID, "read-only-cursor-text-area")))
                    snippet = element.text
                    doc.write(snippet)
                    doc.write('\n')
                    break
                except TimeoutException as e:
                    if attempt == 2:
                        print(f"TimeoutException occurred for URL: {url} \n{e}")
                        traceback.print_exc()
                        break
                except Exception as e:
                    print(f'An exception occurred for {url}: \n{e}')
                    break
                finally:
                    driver.quit()
                    time.sleep(1) # Adding one second delay to respect Github rate-limit

## FUNCTIONS TO CREATE VECTOR DATABASES
# Functions to create chunks, embedds and store them in vectorstores
# Defining splitting type
class SplitterType(Enum):
    CLASSIC = "recursive characters splitter"
    PYTHON = "python code splitter"
    SEMANTIC = "semantic splitter"

def create_chunks(folder_path:str, splitter_type:SplitterType=SplitterType.CLASSIC, chunk_size:int=1000, chunk_overlap:int=200)->list:
    '''This function creates chunks starting from txt files stored in a certain folder.'''
    doc_list = []
    # Store documents in a list
    for file in os.listdir(folder_path):
        if file.endswith('.txt'):
            file_path = os.path.join(folder_path, file)
            loader = TextLoader(file_path)
            # Load data into document objects
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = file.split('_')[0].lower() # Adding metadata
            doc_list.extend(docs)
    # Split documents in chunks
    if splitter_type==SplitterType.PYTHON:
        text_splitter=PythonCodeTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
            )
    elif splitter_type==SplitterType.SEMANTIC:
        text_splitter= SemanticChunker(
            HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"),
            breakpoint_threshold_type= "percentile",
            )
    else:
        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    print(f"Text splitter chosen: {type(text_splitter)}")
    chunks = text_splitter.split_documents(doc_list)
    return chunks

def create_vectorstore(vectorstore_folder:str, chunks_folder: str, splitter_type:SplitterType, chunk_size:int=1000, chunk_overlap:int=200)->Chroma:
    '''This function returns a ChromaDB vectorstore starting from chunks.'''
    # Define embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    chunks = create_chunks(chunks_folder, splitter_type, chunk_size, chunk_overlap)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory= vectorstore_folder
    )
    return vectorstore

# Creating txt documents from PEP URLs
if not os.path.exists(os.path.join(pep_db_folder, "chroma.sqlite3")):
    for name, url in zip(pep_file_names, pep_urls):
        scrape_pep_url(url=url, file_name=name)
    #Storing them into vectorDB
    guidelines_vectorstore = create_vectorstore(vectorstore_folder=pep_db_folder, chunks_folder=pep_db_folder, splitter_type=SplitterType.CLASSIC)
# Python libraries scraping and storing data
if not os.path.exists(os.path.join(pysnippet_folder, "chroma.sqlite3")):
    '''
    ## Note: this repo is huge. It takes a while to be scraped.
    # Sklearn
    sklearn_urls = get_github_urls(url=sklearn_url)
    print(sklearn_urls)
    scrape_github_urls(urls_list=sklearn_urls, file_name=sklearn_file_name)
    '''
    # Flask
    flask_urls = get_github_urls(url=flask_url)
    scrape_github_urls(urls_list=flask_urls, file_name=flask_file_name)
    # Standard libraries
    standardlib_urls = get_github_urls(url=standardlib_url, max_depth=0)
    scrape_github_urls(urls_list=standardlib_urls, file_name=standardlib_file_name)
    # Creating chunks from the generated txt files and store them into a vectorDB
    snippets_code_vectorstore = create_vectorstore(vectorstore_folder=pysnippet_folder, chunks_folder=pysnippet_folder, splitter_type = SplitterType.PYTHON)