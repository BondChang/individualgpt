#!/usr/bin/env python3
from flask import Flask, jsonify, render_template, flash, redirect, url_for, Markup, request
from flask_cors import CORS
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
from langchain import PromptTemplate
import os
import argparse
import time
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
import glob
from typing import List
import requests

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

app = Flask(__name__)
CORS(app)
load_dotenv()

persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
chunk_size = 1000
chunk_overlap = 200

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH', 8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS', 4))
llm = None

from constants import CHROMA_SETTINGS


class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"] = "text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".c": (TextLoader, {'autodetect_encoding': True}),
    ".cc": (TextLoader, {'autodetect_encoding': True}),
    ".cpp": (TextLoader, {'autodetect_encoding': True}),
    ".h": (TextLoader, {'autodetect_encoding': True}),
    ".hpp": (TextLoader, {'autodetect_encoding': True}),
    ".java": (TextLoader, {'autodetect_encoding': True}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    with Pool(processes=os.cpu_count()) as pool:
        results = []
        with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, docs in enumerate(pool.imap_unordered(load_single_document, filtered_files)):
                results.extend(docs)
                pbar.update()

    return results


def split_documents(documents: list[Document]) -> tuple[list[Document], list[Document]]:
    # Splits documents for correct Text Splitter
    text_docs, cpp_docs, java_docs = [], [], []
    for doc in documents:
        file_extension = os.path.splitext(doc.metadata["source"])[1]
        if file_extension == ".cc" or file_extension == ".cpp" or file_extension == ".c" or file_extension == ".h" or file_extension == ".hpp":
            cpp_docs.append(doc)
        elif file_extension == ".java":
            java_docs.append(doc)
        else:
            text_docs.append(doc)

    return text_docs, cpp_docs, java_docs


def process_documents(ignored_files: List[str] = []) -> List[Document]:
    """
    Load documents and split in chunks
    """
    print(f"Loading documents from {source_directory}")
    documents = load_documents(source_directory, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    print(f"Loaded {len(documents)} new documents from {source_directory}")
    text_documents, cpp_documents, java_documents = split_documents(documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    cpp_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    java_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JAVA, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(text_documents)
    texts.extend(cpp_splitter.split_documents(cpp_documents))
    texts.extend(java_splitter.split_documents(java_documents))
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    return texts


def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(
                os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False


@app.route('/ingest', methods=['GET'])
def ingest_data():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']])
        print(f"Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("Creating new vectorstore")
        texts = process_documents()
        print(f"Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory,
                                   client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    print(f"Ingestion complete! You can now run privateGPT.py to query your documents")
    return jsonify(response="Success")


@app.route('/get_answer', methods=['POST'])
def get_answer():
    args = parse_arguments()
    query = request.json
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    if llm == None:
        return "Model not downloaded", 400
    # refine_prompt_template = (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n"
    #     "这是原始问题: {question}\n"
    #     "已有的回答: {existing_answer}\n"
    #     "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
    #     "\n\n"
    #     "{context_str}\n"
    #     "\\nn"
    #     "请根据新的文段，进一步完善你的回答。\n\n"
    #     "### Response: "
    # )
    #
    # initial_qa_template = (
    #     "Below is an instruction that describes a task. "
    #     "Write a response that appropriately completes the request.\n\n"
    #     "### Instruction:\n"
    #     "以下为背景知识：\n"
    #     "{context_str}"
    #     "\n"
    #     "请根据以上背景知识, 回答这个问题：{question}。\n\n"
    #     "### Response: "
    # )
    # refine_prompt = PromptTemplate(
    #     input_variables=["question", "existing_answer", "context_str"],
    #     template=refine_prompt_template,
    # )
    # initial_qa_prompt = PromptTemplate(
    #     input_variables=["context_str", "question"],
    #     template=initial_qa_template,
    # )

    alpaca2_prompt_template = (
        "[INST] <<SYS>>\n"
        "You are a helpful assistant. 你是一个乐于助人的助手。\n"
        "<</SYS>>\n\n"
        "{context}\n\n{question} [/INST]"
    )


    input_with_prompt = PromptTemplate(template=alpaca2_prompt_template, input_variables=["context", "question"])
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                     return_source_documents=not args.hide_source,
                                     chain_type_kwargs={"prompt": input_with_prompt})
    # chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="refine",
    #     retriever=retriever, return_source_documents=not args.hide_source,
    #     chain_type_kwargs=chain_type_kwargs)

    if query != None and query != "":
        res = qa(query)
        answer, docs = res['result'], res['source_documents']

        source_data = []
        for document in docs:
            source_data.append({"name": document.metadata["source"]})

        return jsonify(query=query, answer=answer, source=source_data)

    return "Empty Query", 400


@app.route('/upload_doc', methods=['POST'])
def upload_doc():
    if 'document' not in request.files:
        return jsonify(response="No document file found"), 400

    document = request.files['document']
    if document.filename == '':
        return jsonify(response="No selected file"), 400

    filename = document.filename
    save_path = os.path.join('source_documents', filename)
    document.save(save_path)

    return jsonify(response="Document upload successful")


@app.route('/download_model', methods=['GET'])
def download_and_save():
    url = 'https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin'  # Specify the URL of the resource to download
    filename = 'ggml-gpt4all-j-v1.3-groovy.bin'  # Specify the name for the downloaded file
    models_folder = 'models'  # Specify the name of the folder inside the Flask app root

    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    bytes_downloaded = 0
    file_path = f'{models_folder}/{filename}'
    if os.path.exists(file_path):
        return jsonify(response="Download completed")

    with open(file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=4096):
            file.write(chunk)
            bytes_downloaded += len(chunk)
            progress = round((bytes_downloaded / total_size) * 100, 2)
            print(f'Download Progress: {progress}%')
    global llm
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    return jsonify(response="Download completed")


def load_model():
    filename = 'ggml-model-q4_0_q6s_zxl.bin'  # Specify the name for the downloaded file
    models_folder = 'models'  # Specify the name of the folder inside the Flask app root
    file_path = f'{models_folder}/{filename}'
    if os.path.exists(file_path):
        global llm
        callbacks = [StreamingStdOutCallbackHandler()]
        llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
                       verbose=False,  n_threads=16,n_gpu_layers=os.environ.get('N_GPU_LAYERS'))


def main():
    # Parse the command line arguments
    args = parse_arguments()
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
                           verbose=False, n_threads=16)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', n_batch=model_n_batch,
                          callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    refine_prompt_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "这是原始问题: {question}\n"
        "已有的回答: {existing_answer}\n"
        "现在还有一些文字，（如果有需要）你可以根据它们完善现有的回答。"
        "\n\n"
        "{context_str}\n"
        "\\nn"
        "请根据新的文段，进一步完善你的回答。\n\n"
        "### Response: "
    )

    initial_qa_template = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "以下为背景知识：\n"
        "{context_str}"
        "\n"
        "请根据以上背景知识, 回答这个问题：{question}。\n\n"
        "### Response: "
    )
    refine_prompt = PromptTemplate(
        input_variables=["question", "existing_answer", "context_str"],
        template=refine_prompt_template,
    )
    initial_qa_prompt = PromptTemplate(
        input_variables=["context_str", "question"],
        template=initial_qa_template,
    )
    chain_type_kwargs = {"question_prompt": initial_qa_prompt, "refine_prompt": refine_prompt}
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="refine",
        retriever=retriever, return_source_documents=not args.hide_source,
        chain_type_kwargs=chain_type_kwargs)
    # qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    # prompt_template = ("Below is an instruction that describes a task. "
    #                    "Write a response that appropriately completes the request.\n\n"
    #                    "### Instruction:\n{context}\n\n{question}\n\n### Response: ")
    # from langchain import PromptTemplate
    # PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm, chain_type="stuff",
    #     retriever=retriever,
    #     return_source_documents=not args.hide_source,
    #     chain_type_kwargs={"prompt": PROMPT})
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        # for document in docs:
        # print("\n> " + document.metadata["source"] + ":")
        # print(document.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='privateGPT: Ask questions to your documents without an internet connection, '
                    'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    load_model()
    print("LLM0", llm)
    app.run(host="0.0.0.0", debug=False)
    # main()
