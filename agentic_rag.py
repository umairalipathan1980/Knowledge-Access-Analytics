import logging
import os
import re
import sys
import warnings
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import requests
# import spacy
import streamlit as st
# from bs4 import BeautifulSoup
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain_chroma import Chroma
from langchain_community.document_loaders import (UnstructuredMarkdownLoader,
                                                  WebBaseLoader)
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_groq.chat_models import ChatGroq
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
# RecursiveCharacterTextSplitter removed - using Docling HierarchicalChunker instead
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
# PyPDF2 removed - using Docling for PDF processing
# from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI


# Set up environment variables
os.environ["USER_AGENT"] = "AgenticRAG/1.0"

os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_API_KEY") or ""
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://haagahelia-poc-gaik.openai.azure.com/"
os.environ["AZURE_OPENAI_API_VERSION"] = "2025-03-01-preview" 
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or ""

# API Keys are now loaded from .env file via load_dotenv()
# No need to set them here as they're already loaded

def get_api_config():
    """Get API configuration based on whether Azure endpoint is selected"""
    use_azure = st.session_state.get('use_azure', True)  # Default to Azure
    
    if use_azure:
        return {
            'use_azure': True,
            'api_key': os.getenv("AZURE_API_KEY"),
            'azure_endpoint': "https://haagahelia-poc-gaik.openai.azure.com/",
            'api_version': "2025-03-01-preview",
            'deployment_name': 'gpt-4.1',  # Azure deployment name
            'model': 'gpt-4.1',  # Model identifier
        }
    else:
        return {
            'use_azure': False,
            'api_key': os.getenv("OPENAI_API_KEY"),
            'model': 'gpt-4.1-2025-04-14',  # OpenAI model name
        }


# Resolve or suppress warnings
# Set global logging level to ERROR
logging.basicConfig(level=logging.ERROR, force=True)
# Suppress all SageMaker logs
logging.getLogger("sagemaker").setLevel(logging.CRITICAL)
logging.getLogger("sagemaker.config").setLevel(logging.CRITICAL)

# Ignore the specific FutureWarning from Hugging Face Transformers
warnings.filterwarnings(
    "ignore",
    message="`clean_up_tokenization_spaces` was not set.*",
    category=FutureWarning
)
# General suppression for other warnings (optional)
warnings.filterwarnings("ignore")
# Configure logging
logging.basicConfig(level=logging.INFO)
###################################################

# Legacy paths removed - knowledge base paths now managed by app.py
# persist_directory_openai = 'data/chroma_db_llamaparse-openai'
# persist_directory_huggingface = 'data/chroma_db_llamaparse-huggincface'
collection_name = 'rag'
# Legacy chunk parameters removed - now using Docling HierarchicalChunker

# Generic RAG prompt suitable for any knowledge base
generic_rag_prompt = PromptTemplate(
    template=r"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful, accurate assistant specialized in providing information from a knowledge base. You must follow the instructions strictly and never skip any of them. Follow these rules exactly:

1. **Language Matching**: Always respond in the same language as the question.

2. **Use ONLY Provided Context**: Do not add any external facts. Base your answer strictly and solely on the provided context. If the context includes 'Internet search results:', include them.

3. **Answer Style**:
   - "Concise": Short and direct answers only
   - "Moderate": Balanced and more explanatory than concise
   - "Explanatory": In-depth, with detailed breakdowns, examples, and explanations wherever necessary. 

4. **Citations (MANDATORY, NEVER OMIT)**:
   - You must place each citation **immediately after the statement it supports**, not grouped at the end.
   - If the context is from **Knowledge base results**, include citations like: **[document_name, page xx]** after each related sentence.
   - If the context is from **Internet search results**, include clickable hyperlinks with the name of the website right after the relevant sentence.
   - **Make sure** you do not omit any citations, both for **Knowledge base results** and **Internet search results** if they are present in the context. 
   - **Make sure** the citations in **Internet search results** sections include **clickable hyperlinks**, even in 'Concise' mode.
   - **DO NOT MAKE UP ANY CITATIONS**

5. **Important – Section Separation**:
   - If the context contains **both "Knowledge base results: "** and **"Internet search results: "**, you must always create these two distinct labeled sections:
     - **Knowledge base results**: Show only the knowledge base-based content here. Include citations next to each point as described.
     - **Internet search results**: Show only the Internet-based content here. Include hyperlinks next to each point.
   - **Do NOT combine these sources into a single answer**. Repeat: always split into two sections if both types are found in context.
   - **DO NOT CREATE TWO SECTIONS** if the 'page_content' does not contain **both "Knowledge base results: "** and **"Internet search results: "**

6. **If No Information Is Found**:
   - If the context says "I apologize, but I couldn't find relevant information in the knowledge base," repeat it exactly.
   - If it says "No information from the documents found.", repeat that exact sentence.

7. **Formatting Rules**:
   - Use bullet points for any list
   - Bold important terms
   - Use line breaks between paragraphs and sections
   - Use tables and other visual elements if helpful

8. **Tone**:
   - Be helpful and clear
   - If relevant, refer to earlier conversation
   - Explain in plain, easy-to-understand language

You must follow all the above rules strictly. Never skip citations. Never merge knowledge base and Internet results. Always format clearly.

<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question} 
Context: {context} 
Answer style: {answer_style}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context", "answer_style"]
)


def remove_tags(soup):
    # Remove unwanted tags
    for element in soup(["script", "style", "header", "footer", "nav", "aside", "noscript"]):
        element.decompose()

    # Extract text while preserving structure
    content = ""
    for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li']):
        text = element.get_text(strip=True)
        if element.name.startswith('h'):
            level = int(element.name[1])
            content += '#' * level + ' ' + text + '\n\n'  # Markdown-style headings
        elif element.name == 'p':
            content += text + '\n\n'
        elif element.name == 'li':
            content += '- ' + text + '\n'
    return content

# @st.cache_data
def get_info(URLs):
    """
    Fetch and return contact information from predefined URLs.
    """
    combined_info = ""
    for url in URLs:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                combined_info += "URL: " + url + \
                    ": " + remove_tags(soup) + "\n\n"
            else:
                combined_info += f"Failed to retrieve information from {url}\n\n"
        except Exception as e:
            combined_info += f"Error fetching URL {url}: {e}\n\n"
    return combined_info

# Legacy staticChunker function removed - now using Docling HierarchicalChunker in pdf_parser.py


def load_or_create_vs(persist_directory):
    # Check if the vector store directory exists
    if os.path.exists(persist_directory):
        print("Loading existing vector store...")
        # Load the existing vector store
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=st.session_state.embed_model,
            collection_name=collection_name
        )
    else:
        # Legacy function - this path should not be used anymore
        # New knowledge bases are created through app.py using DoclingParser
        raise ValueError("Knowledge base not found. Please create a new knowledge base through the UI.")

    return vectorstore


def initialize_generic_app(model_name, selected_embedding_model, selected_routing_model, selected_grading_model, kb_path, hybrid_search, internet_search, answer_style):
    """
    Initialize embeddings, vectorstore, retriever, and LLM for the generic RAG workflow.
    Reinitialize components only if the selection has changed.
    """
    # Track current state to prevent redundant initialization
    if "current_model_state" not in st.session_state:
        st.session_state.current_model_state = {
            "answering_model": None,
            "embedding_model": None,
            "routing_model": None,
            "grading_model": None,
        }

    # Check if models or settings have changed
    state_changed = (
        st.session_state.current_model_state["answering_model"] != model_name or
        st.session_state.current_model_state["embedding_model"] != selected_embedding_model or
        st.session_state.current_model_state["routing_model"] != selected_routing_model or
        st.session_state.current_model_state["grading_model"] != selected_grading_model
    )

    # Reinitialize components only if settings have changed
    if state_changed:
        try:
            # Use Azure or OpenAI embeddings based on configuration
            api_config = get_api_config()
            if api_config['use_azure']:
                st.session_state.embed_model = AzureOpenAIEmbeddings(
                    model="text-embedding-3-large",
                    azure_deployment="text-embedding-3-large",  # Azure deployment name for embeddings
                    azure_endpoint=api_config['azure_endpoint'],
                    api_key=api_config['api_key'],
                    api_version=api_config['api_version']
                )
            else:
                st.session_state.embed_model = OpenAIEmbeddings(
                    model="text-embedding-3-large",
                    openai_api_key=api_config['api_key']
                )

            # Initialize vectorstore and retriever using the provided kb_path
            st.session_state.vectorstore = Chroma(
                persist_directory=kb_path,
                embedding_function=st.session_state.embed_model,
                collection_name="rag"
            )
            st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                search_kwargs={"k": 5})

            st.session_state.llm = initialize_llm(model_name, answer_style)
            st.session_state.router_llm = initialize_router_llm(
                selected_routing_model)
            st.session_state.grader_llm = initialize_grading_llm(
                selected_grading_model)
            st.session_state.doc_grader = initialize_grader_chain()

            # Initialize OpenAI client for web search using API config
            api_config = get_api_config()
            if api_config['use_azure']:
                # For Azure, use the Azure OpenAI client
                from openai import AzureOpenAI
                st.session_state.openai_client = AzureOpenAI(
                    api_key=api_config['api_key'],
                    azure_endpoint=api_config['azure_endpoint'],
                    api_version=api_config['api_version']
                )
            else:
                st.session_state.openai_client = OpenAI(api_key=api_config['api_key'])

            # Set the generic RAG prompt
            st.session_state.rag_prompt = generic_rag_prompt

            # Save updated state
            st.session_state.current_model_state.update({
                "answering_model": model_name,
                "embedding_model": selected_embedding_model,
                "routing_model": selected_routing_model,
                "grading_model": selected_grading_model,
            })
        except Exception as e:
            st.error(f"Error during initialization: {e}")
            # Restore previous state if available
            if st.session_state.current_model_state["answering_model"]:
                st.warning(f"Continuing with previous configuration")
            else:
                # Fallback using API configuration
                api_config = get_api_config()
                if api_config['use_azure']:
                    st.session_state.llm = AzureChatOpenAI(
                        azure_deployment=api_config["deployment_name"],
                        api_version=api_config["api_version"],
                        temperature=0.0, streaming=True)
                    st.session_state.router_llm = AzureChatOpenAI(
                        azure_deployment=api_config["deployment_name"],
                        api_version=api_config["api_version"],
                        temperature=0.0)
                    st.session_state.grader_llm = AzureChatOpenAI(
                        azure_deployment=api_config["deployment_name"],
                        api_version=api_config["api_version"],
                        temperature=0.0)
                else:
                    st.session_state.llm = ChatOpenAI(
                        model=api_config['model'], temperature=0.0, streaming=True, api_key=api_config['api_key'])
                    st.session_state.router_llm = ChatOpenAI(
                        model=api_config['model'], temperature=0.0, api_key=api_config['api_key'])
                    st.session_state.grader_llm = ChatOpenAI(
                        model=api_config['model'], temperature=0.0, api_key=api_config['api_key'])
                api_config = get_api_config()
                if api_config['use_azure']:
                    from openai import AzureOpenAI
                    st.session_state.openai_client = AzureOpenAI(
                        api_key=api_config['api_key'],
                        azure_endpoint=api_config['azure_endpoint'],
                        api_version=api_config['api_version']
                    )
                else:
                    st.session_state.openai_client = OpenAI(api_key=api_config['api_key'])
                
                # Set the generic RAG prompt
                st.session_state.rag_prompt = generic_rag_prompt

    # print(f"Using LLM: {model_name}, Router LLM: {selected_routing_model}, Grader LLM:{selected_grading_model}, embedding model: {selected_embedding_model}")

    try:
        return workflow.compile()
    except Exception as e:
        st.error(f"Error compiling workflow: {e}")
        # Return a simple dummy workflow that just echoes the input
        return lambda x: {"generation": "Error in workflow. Please try a different model.", "question": x.get("question", "")}
# @st.cache_resource


def initialize_llm(model_name, answer_style):
    api_config = get_api_config()
    
    if "llm" not in st.session_state or st.session_state.llm.model_name != model_name:
        if answer_style == "Concise":
            temperature = 0.0
        elif answer_style == "Moderate":
            temperature = 0.0
        elif answer_style == "Explanatory":
            temperature = 0.0

        # Use API configuration to determine model and settings
        if api_config['use_azure']:
            # For Azure, use AzureChatOpenAI
            st.session_state.llm = AzureChatOpenAI(
                azure_deployment=api_config["deployment_name"],  # Azure deployment name
                api_version=api_config["api_version"],
                temperature=temperature,
                streaming=True
            )
        else:
            # For OpenAI API
            st.session_state.llm = ChatOpenAI(
                model=api_config['model'],  # gpt-4.1-2025-04-14 for OpenAI
                temperature=temperature,
                streaming=True,
                api_key=api_config['api_key']
            )
       
    return st.session_state.llm


# Legacy initialize_embedding_model function removed - embeddings now handled in initialize_generic_app

# @st.cache_resource

# FIX: mixtral model won't work with ChatGroq idk why. Maybe add gpt-4o-mini as fallback


def initialize_router_llm(selected_routing_model):
    api_config = get_api_config()
    
    if "router_llm" not in st.session_state or st.session_state.router_llm.model_name != selected_routing_model:
        # Use API configuration to determine model and settings
        if api_config['use_azure']:
            st.session_state.router_llm = AzureChatOpenAI(
                azure_deployment=api_config["deployment_name"],  # Azure deployment name
                api_version=api_config["api_version"],
                temperature=0.0
            )
        else:
            st.session_state.router_llm = ChatOpenAI(
                model=api_config['model'], 
                temperature=0.0, 
                api_key=api_config['api_key']
            )

    return st.session_state.router_llm

# @st.cache_resource


def initialize_grading_llm(selected_grading_model):
    api_config = get_api_config()
    
    if "grader_llm" not in st.session_state or st.session_state.grader_llm.model_name != selected_grading_model:
        # Use API configuration to determine model and settings
        if api_config['use_azure']:
            st.session_state.grader_llm = AzureChatOpenAI(
                azure_deployment=api_config["deployment_name"],  # Azure deployment name
                api_version=api_config["api_version"],
                temperature=0.0,
                max_tokens=4000
            )
        else:
            st.session_state.grader_llm = ChatOpenAI(
                model=api_config['model'], 
                temperature=0.0, 
                max_tokens=4000,
                api_key=api_config['api_key']
            )
            

    return st.session_state.grader_llm


model_list = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gpt-4o-mini",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "deepseek-r1-distill-llama-70b"
]

# Legacy embedding model list removed - now using only Azure OpenAI embeddings
embedding_model_list = [
    "text-embedding-3-small", 
    "text-embedding-3-large"
]


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def initialize_grader_chain():
    """
    Initialize the grader chain for document relevance assessment.
    """
    # LLM with function call
    structured_llm_grader = st.session_state.grader_llm.with_structured_output(
        GradeDocuments)

    # Prompt
    system = """You are a grader assessing relevance of a retrieved document to a user question. 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader


def grade_documents(state):
    question = state["question"]
    documents = state.get("documents", [])
    filtered_docs = []

    if not documents:
        print("No documents to grade")
        return {"documents": filtered_docs, "question": question, "web_search_needed": "No"}

    try:
        # Batch grade documents
        for doc in documents:
            score = st.session_state.doc_grader.invoke(
                {"question": question, "document": doc.page_content})
            if score.binary_score == "yes":
                filtered_docs.append(doc)

        if not filtered_docs:
            print("No relevant documents found after grading.")
            result = {"documents": [], "question": question, "web_search_needed": "No"}
            # print(f"Grade_documents returning: {result}")
            return result
        else:
            print(f"Filtered to {len(filtered_docs)} relevant documents.")
            return {"documents": filtered_docs, "question": question, "web_search_needed": "No"}

    except Exception as e:
        print(f"Error during document grading: {e}")
        # If grading fails, return all documents and let the system handle it
        return {"documents": documents, "question": question, "web_search_needed": "No"}


def route_after_grading(state):
    web_search_needed = state.get("web_search_needed", "No")
    # print(f"Routing decision based on web_search_needed={web_search_needed}")
    if web_search_needed == "Yes":
        return "websearch"
    else:
        return "generate"

# Define graph state class


class GraphState(TypedDict):
    question: str
    generation: str
    web_search_needed: str
    documents: List[Document]
    answer_style: str
    hybrid_search: bool
    internet_search: bool


def retrieve(state):
    print("Retrieving documents")
    question = state["question"]
    documents = st.session_state.retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    question = state["question"]
    documents = state.get("documents", [])
    answer_style = state.get("answer_style", "Concise")

    # print(f"Generate function called with {len(documents)} documents")
    # print(f"Documents content: {[doc.page_content[:50] + '...' if len(doc.page_content) > 50 else doc.page_content for doc in documents] if documents else 'Empty'}")

    if "llm" not in st.session_state:
        st.error("LLM not properly initialized")
        return {"generation": "LLM not properly initialized", "question": question}

    if not documents:
        # print("Generate: No documents found, creating context message for LLM")
        # Create a context that tells the LLM there's no relevant information
        context = "I apologize, but I couldn't find relevant information in the knowledge base for your question."
    else:
        # Build context with citation information
        context_parts = []
        for doc in documents:
            content = doc.page_content
            
            # Add metadata information for citation purposes if available
            if hasattr(doc, 'metadata') and doc.metadata:
                metadata = doc.metadata
                document_name = metadata.get('document_name', 'Unknown Document')
                page_number = metadata.get('page_number', 'Unknown')
                
                # Format content with inline citation info for knowledge base results
                if not content.startswith("Internet search results:"):
                    # Add citation metadata in a way the LLM can use
                    if page_number != 'Unknown':
                        content = f"{content} [SOURCE: {document_name}, page {page_number}]"
                    else:
                        content = f"{content} [SOURCE: {document_name}]"
            
            context_parts.append(content)
        
        context = "\n\n".join(context_parts)

    # Use generic RAG prompt
    if context != "No information from the documents found.":
        try:
            rag_chain = st.session_state.rag_prompt | st.session_state.llm | StrOutputParser()

            generation = rag_chain.invoke({
                "context": context,
                "question": question,
                "answer_style": answer_style
            })

            return {"documents": documents, "question": question, "generation": generation}

        except Exception as e:
            print(f"Error in RAG chain generation: {e}")
            # Fallback to a basic response
            return {"documents": documents, "question": question,
                    "generation": f"Error generating response: {e}"}
    else:
        return {"documents": documents, "question": question,
                "generation": "I apologize, but I couldn't find relevant information in the knowledge base."}


def handle_unrelated(state):
    question = state["question"]
    documents = state.get("documents", [])
    
    # Generic unrelated response
    response = "I apologize, but I couldn't find relevant information for your question in my knowledge base."
    
    return {"documents": documents, "question": question, "generation": response}


def grade_retriever_hybrid(vector_docs, question):
    """
    Grade only vector documents during hybrid search.
    
    Parameters:
    - vector_docs: Document list from retriever
    - question: User question
    
    Returns:
    - filtered_vector_docs: List of relevant documents (or error message document)
    """
    filtered_vector_docs = []
    
    if not vector_docs:
        print("No vector documents to grade")
        return [Document(page_content="No relevant information found in knowledge base.")]
    
    try:
        # Grade each vector document
        for doc in vector_docs:
            score = st.session_state.doc_grader.invoke(
                {"question": question, "document": doc.page_content})
            if score.binary_score == "yes":
                filtered_vector_docs.append(doc)
        
        if not filtered_vector_docs:
            print("No relevant vector documents found after grading.")
            return [Document(page_content="No relevant information found in knowledge base.")]
        else:
            print(f"Filtered to {len(filtered_vector_docs)} relevant vector documents.")
            return filtered_vector_docs
    
    except Exception as e:
        print(f"Error during vector document grading: {e}")
        # If grading fails, return all vector documents
        return vector_docs if vector_docs else [Document(page_content="Error during document grading.")]


def web_search(state):
    original_question = state["question"]
    documents = state.get("documents", [])
    
    try:
        print("Invoking web search...")
        
        # Create a generic search instruction
        search_instruction = f"""
        **INSTRUCTION**: The query could be in any language. First detect the language and translate it into English if needed. Your response should always be in English, regardless of the query language.
        
        Search for information related to: {original_question}
        """
        
        # Construct the query
        query = search_instruction
        
        # Call OpenAI's web search API using the old format
        # Use the correct model based on API configuration
        api_config = get_api_config()
        if api_config['use_azure']:
            search_model = api_config['deployment_name']  # Try gpt-4o-search-preview for Azure
            print(f"Azure mode: using deployment_name = {search_model}")
        else:
            search_model = api_config['model']  # Use OpenAI model name
            print(f"OpenAI mode: using model = {search_model}")
        
        print(f"Web search using model: {search_model}")
        
        response = st.session_state.openai_client.responses.create(
            model=search_model,
            tools=[{
                "type": "web_search_preview",
                "user_location": {
                    "type": "approximate",
                    "country": "FI"  
                }
            }],
            input=query
        )
        
        # Process the response using old format (don't add header here - let hybrid_search handle it)
        if hasattr(response, 'output_text') and response.output_text:
            web_results = response.output_text
            # print(f"Web search successful, got results: {len(response.output_text)} characters")
        else:
            web_results = "No web results found from OpenAI search."
            print("Web search returned empty results")
        
        # Add header for standalone internet search
        if len(documents) == 0:  # This is a standalone internet search
            web_results_with_header = "Internet search results: " + web_results
        else:
            web_results_with_header = web_results  # Hybrid search will add its own header
            
        web_results_doc = Document(page_content=web_results_with_header)
        documents.append(web_results_doc)
        # print(f"Added web results document with content prefix: {web_results_with_header[:100]}...")
        
    except Exception as e:
        print(f"Error during web search: {e}")
        # Ensure workflow can continue gracefully
        documents.append(Document(page_content=f"Web search failed: {e}"))
    
    return {"documents": documents, "question": original_question}


def hybrid_search(state):
    question = state["question"]
    print("Invoking hybrid search...")
    
    # Do hybrid search - both knowledge base and web search
    vector_docs = st.session_state.retriever.invoke(question)
    
    # Grade the vector documents
    filtered_vector_docs = grade_retriever_hybrid(vector_docs, question)
    
    # Add headings to distinguish between vector and web search results (preserve metadata)
    vector_results = [Document(
        page_content="Knowledge base results: " + doc.page_content,
        metadata=doc.metadata if hasattr(doc, 'metadata') else {}) for doc in filtered_vector_docs]

    # Proceed with web search
    web_docs = web_search({"question": question})["documents"]
    
    # Add "Internet search results:" header to web results
    web_results = [
        Document(page_content="Internet search results: " + doc.page_content) for doc in web_docs
    ]

    # Combine the filtered vector results with web results
    combined_docs = vector_results + web_results
    return {"documents": combined_docs, "question": question}


# Router function - simplified and generic
def route_question(state):
    question = state["question"]    
    hybrid_search_enabled = state.get("hybrid_search", False)
    internet_search_enabled = state.get("internet_search", False)
    
    # Simple routing logic based on search options
    if hybrid_search_enabled:
        return "hybrid_search_node"
    elif internet_search_enabled:
        return "websearch"
    else:
        return "retrieve"


workflow = StateGraph(GraphState)
# Add nodes
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("route_after_grading", route_after_grading)
workflow.add_node("websearch", web_search)
workflow.add_node("generate", generate)
workflow.add_node("hybrid_search_node", hybrid_search)
workflow.add_node("unrelated", handle_unrelated)

# Set conditional entry points
workflow.set_conditional_entry_point(
    route_question,
    {
        "retrieve": "retrieve",
        "websearch": "websearch",
        "hybrid_search_node": "hybrid_search_node",
        "unrelated": "unrelated"
    },
)

# Add edges
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges(
    "grade_documents",
    route_after_grading,
    {"websearch": "websearch", "generate": "generate"},
)
workflow.add_edge("websearch", "generate")
workflow.add_edge("hybrid_search_node", "generate")
workflow.add_edge("unrelated", "generate")
workflow.add_edge("generate", END)


# Compile app
app = workflow.compile()

