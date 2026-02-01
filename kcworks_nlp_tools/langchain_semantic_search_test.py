#! /usr/bin/python

from collections.abc import Generator
from typing import Callable

import argparse
import orjson
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kcworks_nlp_tools.util import timed


def overwrite(text: str, **kwargs):
    """Overwrite the last line of stdout with a string."""
    LINE_UP = "\033[1A"
    LINE_CLEAR = "\x1b[2K"

    print(LINE_UP, end=LINE_CLEAR, flush=True)
    print(text, **kwargs, flush=True)


class JSONSource:
    @staticmethod
    def assemble_content(obj: dict, template: str = "") -> str:
        return f"{obj['subject']}, {template}"

    @staticmethod
    def assemble_metadata(obj: dict) -> dict:
        return {"fast_id": obj["id"], "scheme": obj["scheme"]}

    def load(
        self,
        lazy: bool = True,
        limit: int = -1,
        splitter: RecursiveCharacterTextSplitter | None = None,
    ) -> Generator[Document]:
        """Load json source documents."""
        facets = [
            {
                "file": "./data/fast_subjects/subjects_fast_topical.jsonl",
                "template": "the topic",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_chronological.jsonl",
                "template": "the chronological date or period",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_corporate.jsonl",
                "template": "the organization, group, or movement",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_event.jsonl",
                "template": "the historical or current event",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_formgenre.jsonl",
                "template": "the media form or genre",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_geographic.jsonl",
                "template": "the geographic location or region",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_meeting.jsonl",
                "template": "the meeting, conference, or event for dialogue",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_personal.jsonl",
                "template": "the person",
            },
            {
                "file": "./data/fast_subjects/subjects_fast_title.jsonl",
                "template": "the title of a work",
            },
        ]
        line_count = 0
        if not splitter:
            raise RuntimeError("Text splitter is required")

        for facet in facets:
            with open(facet["file"], "rb") as json_file:
                print("Opened JSON source file...")
                for line in json_file:
                    line_count += 1

                    if limit >= 0 and line_count > limit:
                        break
                    if not line.strip():
                        continue

                    obj = orjson.loads(line)

                    split_docs = splitter.split_text(
                        self.assemble_content(obj, facet["template"])
                    )
                    for index, doc in enumerate(split_docs):
                        result = Document(
                            page_content=doc,
                            metadata=self.assemble_metadata(obj),
                            id=f"{obj['id']}-{index}",
                        )
                        yield (result)
                print("Done yielding JSON documents for search terms...")


class VectorStore:
    """Class to create and manage a vector_store"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        chunk_size=500,
        chunk_overlap=100,
    ):
        """Initialize a Searcher object."""
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        self.store = Chroma(
            collection_name="fast_subjects",
            embedding_function=self.embeddings,
            persist_directory="./database/chroma_langchain_db",
        )

    def store_vectors(self, docs: Generator[Document]) -> tuple[int, int]:
        """Generate embeddings and store in db."""
        seen_counter = 0
        stored_counter = 0
        spinner_characters = ["|", "/", "-", "\\"]
        first_line = True
        with timed(operation_name="store_vectors"):
            while True:
                try:
                    doc = next(docs)
                    seen_counter += 1
                    existing_vectors = self.vector_store.get_by_ids([doc.id])

                    if len(existing_vectors) == 0:
                        self.store.add_documents(documents=[doc], ids=[doc.id])
                        stored_counter += 1
                    current_spinner_char = spinner_characters[
                        seen_counter % len(spinner_characters)
                    ]
                    if first_line:
                        # First line - just print normally
                        print(
                            f"{current_spinner_char}  storing subject {seen_counter} ({stored_counter} new vectors)"
                        )
                        first_line = False
                    else:
                        overwrite(
                            f"{current_spinner_char}  storing subject {seen_counter} ({stored_counter} new vectors)"
                        )
                except StopIteration:
                    overwrite(
                        f"All done storing {seen_counter} subjects as vectors ({stored_counter} new)."
                    )
                    return seen_counter, stored_counter

    def load_docs(self, json_source: JSONSource, limit: int | None = None) -> None:
        """Produce vectors for subject terms to be searched.

        Loading of documents from JSON and splitting is handled in a generator
        which is then passed into the method for producing the embeddings and storing
        them in persistent storage. This allows loading/splitting and embedding/writing
        operations to happen more-or-less in parallel.

        Arguments:
            json_source: JSONSource object handling the source file.
            limit: The number of documents to load from the JSON source file. Default
                is None, in which case all the documents are loaded.
        """
        docs = json_source.load(limit=limit, splitter=self.splitter)
        self.store_vectors(docs, limit=limit)


class Searcher:
    """Class to perform similarity searches"""

    def __init__(self, vector_store: VectorStore):
        "Initializa a Searcher instance."
        self.vector_store = vector_store

    def _search(self, search_string: str, search_func: Callable):
        """Execute the actual search"""

        with timed(operation_name="search"):
            results = search_func(search_string)
            print("-" * 20)
            print(f'Finding a match for "{search_string}"')
            print("-" * 20)
            print("Results")
            print("-" * 20)

        return results

    def search(self, search_string: str):
        "Perform a similarity search"

        print("Starting similarity search...")
        search_func = self.vector_store.similarity_search
        results = self._search(search_string, search_func)

        for idx, result in enumerate(results):
            print(f"{result.page_content}")
            print(f"    {result.metadata['fast_id']}")
            print(f"    {result.metadata['scheme']}")
        print("-" * 20)

        return results

    def search_with_score(self, search_string: str):
        "Perform a similarity search"

        print("Starting similarity search with score...")
        search_func = self.vector_store.similarity_search_with_score
        results = self._search(search_string, search_func)
        print("first result:")
        print(results[0])

        for idx, result in enumerate(results):
            score = result[1]
            doc = result[0]
            print(f"{doc.page_content}")
            print(f"score: {str(score)}")
            print(f"    {doc.metadata['fast_id']}")
            print(f"    {doc.metadata['scheme']}")
        print("-" * 20)

        return results


def main():
    parser = argparse.ArgumentParser(
        description="A test script for subject semantic search with langchain"
    )
    parser.add_argument("search_string", help="String to find matching subjects for")
    parser.add_argument(
        "--generate-docs",
        "-g",
        action="store_true",
        help="Generate subject term vectors before searching",
    )
    parser.add_argument(
        "--limit-terms",
        "-l",
        type=int,
        help="Limit the number of subject term vectors generated (if --generate is True).",
    )
    args = parser.parse_args()

    # get subject terms vector store
    vector_store = VectorStore()
    if args.generate_docs:
        json_source = JSONSource()
        vector_store.load_docs(json_source, limit=args.limit_terms)

    # search for the search string
    searcher = Searcher(vector_store=vector_store.store)
    searcher.search(search_string=args.search_string)
    searcher.search_with_score(search_string=args.search_string)


if __name__ == "__main__":
    main()
