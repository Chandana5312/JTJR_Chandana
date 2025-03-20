import os
from langchain_openai import AzureOpenAIEmbeddings
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import QueryAnswerType, QueryCaptionType, QueryType, VectorizedQuery
from azure.search.documents.models import VectorizableTextQuery
import tiktoken
import dotenv

dotenv.load_dotenv('../.env')


class SearchAgent():
    def __init__(self):
        self.azure_search_credential = AzureKeyCredential(os.getenv("AZURE_SEARCH_ADMIN_KEY"))
        self.search_client = SearchClient(endpoint=os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT"),
                             index_name=os.getenv("AZURE_SEARCH_INDEX"),
                             credential=self.azure_search_credential)

    # cl100k_base
    def count_tokens(self,text, model_name=os.getenv("AZURE_OPENAI_EMB_MODEL")):
        print("the model name", model_name)  # Adjust model name if needed

        encoding = tiktoken.encoding_for_model(model_name)

        token_list = encoding.encode(text)

        return len(token_list)


    def get_text_embeddings(self,text: str):
        """
        Generate embeddings for the given text using Azure OpenAI's text embedding model.

        Parameters:
        text (str): Input text for which embeddings need to be generated.

        Returns:
        list: A list representing the embedding vector of the input text.
        """
        embedding_input_tokens=self.count_tokens(text)
        # Initialize the Azure OpenAI embedding model with specified parameters
        embedding_model = AzureOpenAIEmbeddings(
            model=os.getenv('AZURE_OPENAI_EMB_MODEL'),  # Model name
            dimensions=1536,  # Number of dimensions for the embeddings
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),  # Azure OpenAI endpoint
            api_key=os.getenv("AZURE_OPENAI_API_KEY")  # API key for authentication
            # openai_api_version=AZURE_VERSION  # Uncomment if API version is required
        )

        # Generate the embedding vector for the input text
        query_vector = embedding_model.embed_query(text)

        return query_vector,embedding_input_tokens

    def search(self,
                    query_text,
                    has_text=True,
                    has_vector=False,
                    use_semantic_captions=False,
                    top=10):
        """
        Perform a search query using either text-based or vector-based search methods.

        Parameters:
        query_text (str): The search query text.
        has_text (bool): Whether to include text-based search (default: True).
        has_vector (bool): Whether to include vector-based search (default: False).
        use_semantic_captions (bool): Whether to use semantic captions (default: False).
        top (int): Number of top results to retrieve (default: 10).

        Returns:
        list: A list of search results.
        """
        print("-------------------------------------------------------------------")
        print(query_text)
        print("-------------------------------------------------------------------")

        filter = None
        vector_query = None

        if has_vector:
            print("---- Vector")
            embedding,embedding_input_tokens = self.get_text_embeddings(query_text)
            vector_query = [
                VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="description_embedding")
            ]
        else:
            print("---- Text")

        if not has_text:
            print("---- No Text")
            query_text = ""
        else:
            print("---- Text")

        # Use semantic ranker if requested and if retrieval mode is text or hybrid (vectors + text)
        if use_semantic_captions and has_text:
            print("If")
            r = self.search_client.search(
                query_text,
                filter=filter,
                select=["job_role", "job_role_description", "marketing_audience", "function", "seniority"],
                query_type=QueryType.SEMANTIC,
                query_language="en-us",
                query_speller="lexicon",
                semantic_configuration_name="my-semantic-config",
                top=top,
                query_caption="extractive" if use_semantic_captions else None,
                vector_queries=vector_query
            )
        else:
            print("Else")
            print(query_text)
            r = self.search_client.search(
                query_text,
                filter=filter,
                select=["job_role", "job_role_description", "marketing_audience", "function", "seniority"],
                top=top,
                vector_queries=vector_query
            )

        results = list(r)
        return results,embedding_input_tokens
