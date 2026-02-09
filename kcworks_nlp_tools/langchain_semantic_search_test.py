#! /usr/bin/python

# When run directly (e.g. python kcworks_nlp_tools/langchain_semantic_search_test.py), add project root to sys.path so the package is importable.
import sys
from pathlib import Path

if not __package__:
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Callable

import argparse
import ast
import orjson
from chromadb import Search, Knn, Rrf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kcworks_nlp_tools import config
from kcworks_nlp_tools.dependencies import extract_fast_files
from kcworks_nlp_tools.util import overwrite, timed, get_package_root


class VectorStore(ABC):
    """Abstract interface for the store object passed to Searcher.

    Implementations (ChromaVectorStore, OpenSearchVectorStore) delegate to their
    backend. Searcher calls only these four methods.
    """

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents most similar to the query."""
        ...

    @abstractmethod
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return documents and similarity scores."""
        ...

    @abstractmethod
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents balanced for similarity and diversity."""
        ...

    @abstractmethod
    def hybrid_search(
        self,
        query_string: str,
        k: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return documents from hybrid (e.g. vector + keyword) search."""
        ...


def _model_slug(model_name: str) -> str:
    """Return the slug for model_name from config.MODEL_SLUGS. Raises if the model is not in the config."""
    if model_name not in config.MODEL_SLUGS:
        raise ValueError(
            f"Unknown model {model_name!r}. Add it to MODEL_SLUGS in config.py (model name -> short slug for DB dirs)."
        )
    return config.MODEL_SLUGS[model_name]


def _chroma_persist_directory(algorithm: str, model_slug: str) -> str:
    """Persist directory under package database dir: chroma_langchain_db-{model_slug}-{algorithm}. Creates the directory if needed."""
    path = (
        Path(get_package_root())
        / "database"
        / f"{config.CHROMA_DB_DIR_NAME}-{model_slug}-{algorithm}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


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
        facets = list(config.FACETS)
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

                    if limit and limit >= 0 and line_count > limit:
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


class ChromaVectorStore(VectorStore):
    """Chroma-backed vector store: create and manage vectors, delegate search to Chroma."""

    def __init__(
        self,
        model_name: str = config.DEFAULT_MODEL_NAME,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        distance_algorithm: str = config.DEFAULT_DISTANCE_ALGORITHM,
    ):
        """Initialize a ChromaVectorStore.

        Args:
            model_name: HuggingFace model for embeddings.
            chunk_size: Max tokens per chunk.
            chunk_overlap: Overlap between chunks.
            distance_algorithm: Chroma comparison algorithm: 'l2', 'cosine', or 'ip'.
                Determines persist directory (e.g. ...-cosine) and collection metadata.
        """
        if distance_algorithm not in config.CHROMA_DISTANCE_ALGORITHMS:
            raise ValueError(
                f"distance_algorithm must be one of {config.CHROMA_DISTANCE_ALGORITHMS}, got {distance_algorithm!r}"
            )
        self.distance_algorithm = distance_algorithm
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
        )

        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)

        model_slug = _model_slug(
            model_name
        )  # from config.MODEL_SLUGS; used in persist path
        self.persist_directory = _chroma_persist_directory(
            distance_algorithm, model_slug
        )
        print(f"** vector store path: {self.persist_directory}")
        self.store = Chroma(
            collection_name="fast_subjects",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_metadata={"hnsw:space": distance_algorithm},
        )
        self._warmup_store()

    def _warmup_store(self) -> None:
        """Run two searches so Chroma and the wrapper absorb any second-call one-time cost."""
        with timed("warmup search (Chroma index load)"):
            self.store.similarity_search(
                "warmup searches in artificial intelligence semantic retrieval",
                k=config.DEFAULT_RESULT_SIZE,
            )
            self.store.similarity_search(
                "warmup searches in artificial intelligence semantic retrieval",
                k=config.DEFAULT_RESULT_SIZE,
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

    def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        return self.store.similarity_search(query, k=k, filter=filter, **kwargs)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        return self.store.similarity_search_with_score(
            query, k=k, filter=filter, **kwargs
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 10,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        **kwargs: Any,
    ) -> list[Document]:
        return self.store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, **kwargs
        )

    def hybrid_search(
        self,
        query_string: str,
        k: int = 10,
        filter: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        hybrid_rank = Rrf(
            ranks=[
                Knn(query="query", return_rank=True, limit=300),
                Knn(query="query learning applications", key="sparse_embedding"),
            ],
            weights=[2.0, 1.0],
            k=60,
        )
        search = Search().rank(hybrid_rank).limit(k)
        return self.store.hybrid_search(search)


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
        print("Starting max marginal relevance search...")
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
        with timed(operation_name="hybrid_search"):
            print("-" * 20)
            print(f'Finding a match for "{search_string}"')
            print("-" * 20)
            results = self.vector_store.hybrid_search(
                search_string, k=result_size, filter=filter
            )
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
        default=config.DEFAULT_MODEL_NAME,
        help="Embedding model name (e.g. sentence-transformers/...). Determines DB subdir via config.MODEL_SLUGS. Default: %(default)s.",
    )
    parser.add_argument(
        "--chunk_size",
        "-c",
        type=int,
        default=config.DEFAULT_CHUNK_SIZE,
        help="Max number of tokens in each document. Default is 500.",
    )
    parser.add_argument(
        "--chunk_overlap",
        "-o",
        type=int,
        default=config.DEFAULT_CHUNK_OVERLAP,
        help="Number of tokens to overlap between chunks in split documents. Default is 100.",
    )
    parser.add_argument(
        "--size",
        "-s",
        type=int,
        default=config.DEFAULT_RESULT_SIZE,
        help="Number of search results to return. Default is 10.",
    )
    parser.add_argument(
        "--filter",
        "-f",
        type=str,
        help="Filter to constrain search results (dict as string, e.g. \"{'scheme': 'topical'}\")",
    )
    parser.add_argument(
        "--facet",
        type=str,
        choices=config.FACET_SCHEME_NAMES,
        help="Restrict search to this scheme (facet) only. Sets filter to scheme=<facet>.",
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
        default=config.DEFAULT_FETCH_K,
        help=(
            "Number of documents to pass to the MMR algorithm. (Only applies to max-marginal-relevance searches.) Default is 20."
        ),
    )
    parser.add_argument(
        "--lambda-mult",
        dest="lambda_mult",
        type=float,
        default=config.DEFAULT_LAMBDA_MULT,
        help="Number between 0 and 1 that determines the degree of diversity among the results with 0 corresponding to maximum diversity and 1 to minimum diversity. (Only applies to max-marginal-relevance searches.)",
    )
    parser.add_argument(
        "--distance",
        "-d",
        dest="distance_algorithm",
        choices=config.CHROMA_DISTANCE_ALGORITHMS,
        default=config.DEFAULT_DISTANCE_ALGORITHM,
        help="Chroma distance algorithm for the vector store. Uses a separate DB directory per algorithm (e.g. ...-cosine). Default: %(default)s.",
    )
    parser.add_argument(
        "--backend",
        "-b",
        choices=("chroma", "opensearch"),
        default=config.VECTOR_STORE_BACKEND,
        help="Vector store backend (chroma or opensearch). Default from env VECTOR_STORE_BACKEND or chroma.",
    )
    args = parser.parse_args()

    filter_dict = ast.literal_eval(args.filter) if args.filter is not None else None
    if args.facet is not None:
        filter_dict = {**(filter_dict or {}), "scheme": args.facet}
    where_dict = ast.literal_eval(args.where) if args.where is not None else None
    chunk_size = args.chunk_size or config.DEFAULT_CHUNK_SIZE
    chunk_overlap = args.chunk_overlap or config.DEFAULT_CHUNK_OVERLAP
    backend = args.backend

    # get subject terms vector store (persist dir and collection use --distance)
    with timed("initializing VectorStore"):
        if backend == "chroma":
            vector_store = ChromaVectorStore(
                model_name=args.model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                distance_algorithm=args.distance_algorithm,
            )
        elif backend == "opensearch":
            raise NotImplementedError(
                "OpenSearch backend is not implemented yet. Use --backend chroma."
            )
        else:
            raise ValueError(f"Unknown backend: {backend!r}")
    with timed("generating docs from JSONSource"):
        if args.generate_docs is True:
            with timed("initializing JSONSource"):
                json_source = JSONSource()
            with timed("loading docs from JSONSource"):
                vector_store.load_docs(json_source, limit=args.limit_terms)

    # search for the search string
    with timed("initializing Searcher"):
        searcher = Searcher(vector_store=vector_store)

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
