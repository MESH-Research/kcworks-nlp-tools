#! /usr/bin/python

from collections.abc import Generator
from pathlib import Path
from typing import Callable

import argparse
import ast
import orjson
from chromadb import Search, Knn, Rrf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kcworks_nlp_tools.dependencies import extract_fast_files
from kcworks_nlp_tools.util import overwrite, timed, get_package_root

DEFAULT_RESULT_SIZE = 20
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_FETCH_K = 20
DEFAULT_LAMBDA_MULT = 0.5
DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class JSONSource:
    def __init__(self):
        self.package_root = get_package_root()

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
                "file": "data/fast_subjects/subjects_fast_topical.jsonl",
                "template": "the topic",
            },
            {
                "file": "data/fast_subjects/subjects_fast_chronological.jsonl",
                "template": "the chronological date or period",
            },
            {
                "file": "data/fast_subjects/subjects_fast_corporate.jsonl",
                "template": "the organization, group, or movement",
            },
            {
                "file": "data/fast_subjects/subjects_fast_event.jsonl",
                "template": "the historical or current event",
            },
            {
                "file": "data/fast_subjects/subjects_fast_formgenre.jsonl",
                "template": "the media form or genre",
            },
            {
                "file": "data/fast_subjects/subjects_fast_geographic.jsonl",
                "template": "the geographic location or region",
            },
            {
                "file": "data/fast_subjects/subjects_fast_meeting.jsonl",
                "template": "the meeting, conference, or event for dialogue",
            },
            {
                "file": "data/fast_subjects/subjects_fast_personal.jsonl",
                "template": "the person",
            },
            {
                "file": "data/fast_subjects/subjects_fast_title.jsonl",
                "template": "the title of a work",
            },
        ]
        paths = [facet["file"] for facet in facets]
        # ensure JSON source files are available
        file_result = extract_fast_files(paths)
        if len(file_result.missing_files) > 0 or len(file_result.unextracted_files) > 0:
            print("*** MISSING JSON SOURCE FILES ***")
            print(file_result.missing_files + file_result.unextracted_files)
            if len(file_result.extracted_files) > 0:
                print("*** Proceeding with facets that are available")
                facets = [f for f in facets if f["file"] in file_result.extracted_files]
            else:
                raise RuntimeError("No JSON source files available.")

        line_count = 0
        if not splitter:
            raise RuntimeError("Text splitter is required")

        for facet in facets:
            file_path = Path(self.package_root) / facet["file"]
            with open(file_path, "rb") as json_file:
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
        model_name: str = DEFAULT_MODEL_NAME,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
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
                    existing_vectors = self.store.get_by_ids([doc.id])

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
        self.store_vectors(docs)


class Searcher:
    """Class to perform similarity searches"""

    def __init__(self, vector_store: VectorStore):
        "Initializa a Searcher instance."
        self.vector_store = vector_store

    @staticmethod
    def _print_results(results: list, scored: bool = False) -> None:
        """Print result list."""

        for idx, result in enumerate(results):
            doc = result
            if scored:
                doc = result[0]
                score = result[1]
            print(f"{doc.page_content}")
            if scored:
                print(f"score: {str(score)}")
            print(f"    {doc.metadata['fast_id']}")
            print(f"    {doc.metadata['scheme']}")
        print("-" * 20)

    def _search(
        self,
        search_string: str,
        search_func: Callable,
        result_size: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs,
    ):
        """Execute the actual search"""

        with timed(operation_name="search"):
            results = search_func(search_string, k=result_size, filter=filter, **kwargs)
            print("-" * 20)
            print(f'Finding a match for "{search_string}"')
            print("-" * 20)
            print("Results")
            print("-" * 20)

        return results

    def search(
        self,
        search_string: str,
        result_size: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs,
    ):
        "Perform a similarity search"
        print("Starting similarity search...")
        search_func = self.vector_store.similarity_search
        results = self._search(
            search_string, search_func, result_size=result_size, filter=filter, **kwargs
        )

        self._print_results(results)

        return results

    def search_with_score(
        self,
        search_string: str,
        result_size: int = 10,
        filter: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
        **kwargs,
    ):
        "Perform a similarity search with distance score."
        print("Starting similarity search with distance score...")
        search_func = self.vector_store.similarity_search_with_score
        results = self._search(
            search_string,
            search_func,
            result_size=result_size,
            filter=filter,
            where_document=where_document,
            **kwargs,
        )
        print("first result:")
        print(results[0])

        self._print_results(results, scored=True)

        return results

    def search_marginal_relevance(
        self,
        search_string: str,
        result_size: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        """Perform a max marginal relevance search.

        This search algorithm takes into account both document similarity
        *and* diversity in returned documents.
        """
        print("Starting similarity search with distance score...")
        search_func = self.vector_store.max_marginal_relevance_search
        results = self._search(
            search_string, search_func, result_size=10, fetch_k=20, lambda_mult=0.5
        )

        self._print_results(results)

        return results

    def search_hybrid(
        self,
        search_string: str,
        result_size: int = 10,
        filter: dict[str, str] | None = None,
        where_document: dict[str, str] | None = None,
    ):
        """Perform a hybrid search."""
        hybrid_rank = Rrf(
            ranks=[
                Knn(query="query", return_rank=True, limit=300),
                Knn(query="query learning applications", key="sparse_embedding"),
            ],
            weights=[2.0, 1.0],  # Dense 2x more important
            k=60,
        )

        search = (
            Search()
            # .where((K("language") == "en") & (K("year") >= 2020))
            .rank(hybrid_rank)
            .limit(10)
            # .select(K.DOCUMENT, K.SCORE, "title", "year")
        )

        with timed(operation_name="hybrid_search"):
            print("-" * 20)
            print(f'Finding a match for "{search_string}"')
            print("-" * 20)
            results = self.vector_store.hybrid_search(search)
            print("Results")
            print("-" * 20)

        self._print_results(results)

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
    parser.add_argument(
        "--model_name",
        "-m",
        default=DEFAULT_MODEL_NAME,
        help="Limit the number of subject term vectors generated (if --generate is True).",
    )
    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Max number of tokens in each document. Default is 500.",
    )
    parser.add_argument(
        "--chunk_overlap",
        "-o",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Number of tokens to overlap between chunks in split documents. Default is 100.",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=DEFAULT_RESULT_SIZE,
        help="Number of search results to return. Default is 10.",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        help="Filter to constrain search results (dict as string, e.g. \"{'scheme': 'topical'}\")",
    )
    parser.add_argument(
        "--where",
        "-w",
        type=str,
        help=(
            "Where clause to restrict results to matching documents (dict as string). (Only applies to scored similarity, MMR, and hybrid searches.)"
        ),
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=DEFAULT_FETCH_K,
        help=(
            "Number of documents to pass to the MMR algorithm. (Only applies to max-marginal-relevance searches.) Default is 20."
        ),
    )
    parser.add_argument(
        "--lambda-mult",
        dest="lambda_mult",
        type=float,
        default=DEFAULT_LAMBDA_MULT,
        help="Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. (Only applies to max-marginal-relevance searches.)",
    )
    args = parser.parse_args()

    filter_dict = ast.literal_eval(args.filter) if args.filter is not None else None
    where_dict = ast.literal_eval(args.where) if args.where is not None else None
    chunk_size = args.chunk_size or DEFAULT_CHUNK_SIZE
    chunk_overlap = args.chunk_overlap or DEFAULT_CHUNK_OVERLAP

    # get subject terms vector store
    vector_store = VectorStore(
        model_name=args.model_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if args.generate_docs:
        json_source = JSONSource()
        vector_store.load_docs(json_source, limit=args.limit_terms)

    # search for the search string
    searcher = Searcher(vector_store=vector_store.store)

    searcher.search(
        search_string=args.search_string,
        result_size=args.size,
        filter=filter_dict,
    )
    searcher.search_with_score(
        search_string=args.search_string,
        result_size=args.size,
        filter=filter_dict,
        where_document=where_dict,
    )
    searcher.search_marginal_relevance(
        search_string=args.search_string,
        result_size=args.size,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
    )
    # searcher.search_hybrid(
    #     search_string=args.search_string,
    #     result_size=args.size,
    #     filter=filter_dict,
    #     where_document=where_dict,
    # )


if __name__ == "__main__":
    main()
