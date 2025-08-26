try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("Using pysqlite3 module instead of sqlite3 (Rahti compatible)")
except ImportError:
    print("pysqlite3 not found, using standard sqlite3 module (local development)")

import io
import os
import re
import sys
import time
import glob
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from agentic_rag import initialize_generic_app
from st_callback import get_streamlit_cb
from pdf_parser import DoclingParser

# -------------------- Initialization --------------------
st.set_option("client.showErrorDetails", False)

# Early session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "followup_key" not in st.session_state:
    st.session_state.followup_key = 0
if "pending_followup" not in st.session_state:
    st.session_state.pending_followup = None
if "last_assistant" not in st.session_state:
    st.session_state.last_assistant = None
if "followup_questions" not in st.session_state:
    st.session_state.followup_questions = []
if "selected_knowledge_base" not in st.session_state:
    st.session_state.selected_knowledge_base = None
if "available_knowledge_bases" not in st.session_state:
    st.session_state.available_knowledge_bases = []

# -------------------- Helper Functions --------------------
def get_available_knowledge_bases():
    """Get list of available Chroma databases in the data folder"""
    data_folder = Path("data")
    if not data_folder.exists():
        return []
    
    knowledge_bases = []
    for item in data_folder.iterdir():
        if item.is_dir() and item.name.startswith("chroma_db_"):
            kb_name = item.name.replace("chroma_db_", "").replace("-", " ").title()
            
            # Try to load description from description.txt
            description_file = item / "description.txt"
            description = "No description available"
            if description_file.exists():
                try:
                    with open(description_file, "r", encoding="utf-8") as f:
                        description = f.read().strip()
                except Exception as e:
                    print(f"Error reading description for {kb_name}: {e}")
            
            knowledge_bases.append({
                "name": kb_name, 
                "path": str(item),
                "description": description
            })
    
    return knowledge_bases

def update_knowledge_base_with_pdfs(kb_path, pdf_files):
    """Update an existing knowledge base with additional PDF files"""
    try:
        # Create data directory if it doesn't exist
        data_folder = Path("data")
        data_folder.mkdir(exist_ok=True)
        
        # Parse new PDFs using docling
        parser = DoclingParser()
        new_documents = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file.name} with metadata extraction...")
            
            # Save uploaded file temporarily
            temp_path = data_folder / pdf_file.name
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Parse PDF to chunks with metadata (document name, page numbers, etc.)
            chunks_with_metadata = parser.convert_pdf_to_chunks_with_metadata(str(temp_path))
            new_documents.extend(chunks_with_metadata)
            
            # Clean up temp file
            temp_path.unlink()
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        # Create embeddings using Azure or OpenAI based on configuration
        from agentic_rag import get_api_config
        api_config = get_api_config()
        
        if api_config['use_azure']:
            embeddings = AzureOpenAIEmbeddings(
                model="text-embedding-3-large",
                azure_deployment="text-embedding-3-large",  # Azure deployment name for embeddings
                azure_endpoint=api_config['azure_endpoint'],
                api_key=api_config['api_key'],
                api_version=api_config['api_version']
            )
        else:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=api_config['api_key']
            )
        
        # Load existing vector store
        existing_vectorstore = Chroma(
            persist_directory=kb_path,
            embedding_function=embeddings,
            collection_name="rag"
        )
        
        # Add new documents to existing vector store
        existing_vectorstore.add_documents(new_documents)
        
        status_text.text("Knowledge base updated successfully!")
        progress_bar.progress(1.0)
        
        return True
    
    except Exception as e:
        st.error(f"Error updating knowledge base: {str(e)}")
        return False

def create_knowledge_base_from_pdfs(kb_name, kb_description, pdf_files):
    """Create a new knowledge base from PDF files"""
    try:
        # Create data directory if it doesn't exist
        data_folder = Path("data")
        data_folder.mkdir(exist_ok=True)
        
        # Parse PDFs using docling
        parser = DoclingParser()
        all_documents = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file.name} with metadata extraction...")
            
            # Save uploaded file temporarily
            temp_path = data_folder / pdf_file.name
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Parse PDF to chunks with metadata (document name, page numbers, etc.)
            chunks_with_metadata = parser.convert_pdf_to_chunks_with_metadata(str(temp_path))
            all_documents.extend(chunks_with_metadata)
            
            # Clean up temp file
            temp_path.unlink()
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        # Use the chunks directly (already split with metadata preserved)
        splits = all_documents
        
        # Create embeddings using Azure or OpenAI based on configuration
        from agentic_rag import get_api_config
        api_config = get_api_config()
        
        if api_config['use_azure']:
            embeddings = AzureOpenAIEmbeddings(
                model="text-embedding-3-large",
                azure_deployment="text-embedding-3-large",  # Azure deployment name for embeddings
                azure_endpoint=api_config['azure_endpoint'],
                api_key=api_config['api_key'],
                api_version=api_config['api_version']
            )
        else:
            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=api_config['api_key']
            )
        
        # Create Chroma database
        persist_directory = data_folder / f"chroma_db_{kb_name.lower().replace(' ', '-')}"
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name="rag"
        )
        
        # Save description metadata
        description_file = persist_directory / "description.txt"
        with open(description_file, "w", encoding="utf-8") as f:
            f.write(kb_description.strip() if kb_description else "No description provided")
        
        status_text.text("Knowledge base created successfully!")
        progress_bar.progress(1.0)
        
        return True, str(persist_directory)
    
    except Exception as e:
        st.error(f"Error creating knowledge base: {str(e)}")
        return False, None

def get_followup_questions(last_user, last_assistant):
    """Generate three concise follow-up questions dynamically based on the latest conversation."""
    prompt = f"""Based on the conversation below:
User: {last_user}
Assistant: {last_assistant}
Generate three concise follow-up questions that a user might ask next.
Each question should be on a separate line. The generated questions should be independent and can be answered without knowing the last question. Focus on brevity.
Follow-up Questions:"""
    try:
        if any(model_type in st.session_state.selected_model.lower()
               for model_type in ["gemma2", "deepseek", "mixtral"]):
            # Use API configuration for fallback
            from agentic_rag import get_api_config
            api_config = get_api_config()
            if api_config['use_azure']:
                fallback_llm = AzureChatOpenAI(
                    azure_deployment=api_config["deployment_name"],
                    api_version=api_config["api_version"],
                    temperature=0.5)
            else:
                fallback_llm = ChatOpenAI(
                    model=api_config['model'], 
                    temperature=0.5, 
                    api_key=api_config['api_key'])
            response = fallback_llm.invoke(prompt)
        else:
            response = st.session_state.llm.invoke(prompt)

        text = response.content if hasattr(response, "content") else str(response)
        questions = [q.strip() for q in text.split('\n') if q.strip()]
        return questions[:3]
    except Exception as e:
        print(f"Failed to generate follow-up questions: {e}")
        return []

def process_question(question, answer_style):
    """Process a question through the RAG workflow"""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")

    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    assistant_response = ""

    st.session_state.messages.append({"role": "assistant", "content": ""})
    assistant_index = len(st.session_state.messages) - 1

    # Process the question without displaying response here (conversation history will handle display)
    debug_placeholder = st.empty()
    st_callback = get_streamlit_cb(st.empty())

    start_time = time.time()

    with st.spinner("Thinking..."):
        inputs = {
            "question": question,
            "hybrid_search": st.session_state.get("hybrid_search", False),
            "internet_search": st.session_state.get("internet_search", False),
            "answer_style": answer_style
        }
        try:
            for idx, chunk in enumerate(app.stream(inputs, config={"callbacks": [st_callback]})):
                debug_logs = output_buffer.getvalue()
                debug_placeholder.text_area(
                    "Debug Logs", debug_logs, height=100, key=f"debug_logs_{idx}"
                )
                if "generate" in chunk and "generation" in chunk["generate"]:
                    assistant_response += chunk["generate"]["generation"]
                    # Update the session state in real-time for conversation history display
                    st.session_state.messages[assistant_index]["content"] = assistant_response
        except Exception as e:
            error_str = str(e)
            if "Bad message format" not in error_str:
                error_msg = f"Error generating response: {error_str}"
                st.error(error_msg)
                st_callback.text = error_msg

        if not assistant_response.strip():
            try:
                result = app.invoke(inputs)
                if "generation" in result:
                    assistant_response = result["generation"]
                    # Update session state for conversation history display
                    st.session_state.messages[assistant_index]["content"] = assistant_response
                else:
                    raise ValueError("No generation found in result")
            except Exception as fallback_error:
                fallback_str = str(fallback_error)
                if "Bad message format" not in fallback_str:
                    print(f"Fallback also failed: {fallback_str}")
                    if not assistant_response.strip():
                        error_msg = ("Sorry, I encountered an error while generating a response. "
                                     "Please try again or select a different model.")
                        st.error(error_msg)
                        assistant_response = error_msg

    end_time = time.time()
    generation_time = end_time - start_time
    st.session_state["last_generation_time"] = generation_time

    sys.stdout = sys.__stdout__

    st.session_state.messages[assistant_index]["content"] = assistant_response
    st.session_state.followup_key += 1

# -------------------- Knowledge Base Selection Screen --------------------
if st.session_state.selected_knowledge_base is None:
    st.set_page_config(page_title="Knowledge Base Selection", layout="centered", page_icon="🧠")
    
    st.markdown("""
    <style>
    .kb-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: #f9f9f9;
    }
    .kb-title {
        font-size: 18px;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .create-kb-section {
        border: 2px dashed #007bff;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 5px;
        text-align: center;
        background-color: #f8f9fa;
        height: fit-content;
    }
    /* Style for disabled radio options */
    div[data-testid="stRadio"] > div > div > label:has(span:contains("disabled")) {
        opacity: 0.6;
        pointer-events: none;
        color: #888888;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center;'>🧠 Knowledge Base Manager</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Select a knowledge base or create a new one from your documents</p>", unsafe_allow_html=True)
    
    # API Configuration Section
    st.markdown("## ⚙️ API Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        # Show both options but handle the disabled state properly
        selected_option = st.radio(
            "Select API Provider",
            ["Microsoft Azure Endpoint", "🔒 OpenAI API Key (disabled)"],
            index=0,  # Default to Azure
            key="api_provider_choice"
        )
        
        # Show warning if user selects the disabled option
        if selected_option == "🔒 OpenAI API Key (disabled)":
            st.warning("⚠️ OpenAI API option is currently disabled. Using Azure instead.")
        
        # Always use Azure regardless of selection
        api_choice = "Microsoft Azure Endpoint"
    
    with col2:
        if api_choice == "Microsoft Azure Endpoint":
            st.session_state.use_azure = True
            st.info("🏢 **Using Azure Endpoint**: haagahelia-poc-gaik.openai.azure.com")
        else:
            st.session_state.use_azure = False
            st.info("🔑 **Using OpenAI API**: Standard OpenAI endpoint")
    
    # Check API key availability
    from agentic_rag import get_api_config
    api_config = get_api_config()
    if not api_config['api_key']:
        api_type = "Azure" if api_config['use_azure'] else "OpenAI"
        st.error(f"❌ {api_type} API key not found in .env file. Please check your configuration.")
    else:
        api_type = "Azure" if api_config['use_azure'] else "OpenAI" 
        st.success(f"✅ {api_type} API key configured")
    
    st.divider()
    
    # Get available knowledge bases
    available_kbs = get_available_knowledge_bases()
    st.session_state.available_knowledge_bases = available_kbs
    
    if available_kbs:
        st.markdown("## 📚 Available Knowledge Bases")
        
        # Create compact list with selectbox
        kb_names = [kb["name"] for kb in available_kbs]
        selected_kb_name = st.selectbox(
            "Select a knowledge base:",
            kb_names,
            key="kb_selector",
            help="Choose an existing knowledge base to work with"
        )
        
        if st.button("🚀 Load Selected Knowledge Base", key="load_kb", use_container_width=True):
            selected_kb = next(kb for kb in available_kbs if kb["name"] == selected_kb_name)
            st.session_state.selected_knowledge_base = {
                "name": selected_kb["name"],
                "path": selected_kb["path"], 
                "description": selected_kb.get("description", "No description available")
            }
            st.rerun()
    
    # Create two columns for Create and Update sections
    col1, col2 = st.columns(2)
    
    with col1:
        # Create new knowledge base section
        st.markdown('<div class="create-kb-section">', unsafe_allow_html=True)
        st.markdown("## ➕ Create New Knowledge Base")
        st.markdown("Upload PDF files to create a new knowledge base")
        
        with st.form("create_kb_form"):
            kb_name = st.text_input("Knowledge Base Name", placeholder="e.g., My Documents")
            kb_description = st.text_area(
                "Knowledge Base Description",
                placeholder="Brief description of what this knowledge base contains (max 200 characters)",
                max_chars=200,
                help="Provide a brief description of the content and purpose of this knowledge base"
            )
            uploaded_files = st.file_uploader(
                "Upload PDF Files",
                type=['pdf'],
                accept_multiple_files=True,
                key="create_files"
            )
            st.info("📋 **A new knowledge base will be created from scratch.**")
            
            if st.form_submit_button("Create Knowledge Base"):
                if not kb_name.strip():
                    st.error("Please enter a knowledge base name")
                elif not uploaded_files:
                    st.error("Please upload at least one PDF file")
                else:
                    success, persist_dir = create_knowledge_base_from_pdfs(kb_name, kb_description, uploaded_files)
                    if success:
                        st.success("Knowledge base created successfully!")
                        st.session_state.selected_knowledge_base = {
                            "name": kb_name,
                            "path": persist_dir,
                            "description": kb_description.strip() if kb_description else "No description provided"
                        }
                        time.sleep(2)
                        st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Update existing knowledge base section
        if available_kbs:
            st.markdown('<div class="create-kb-section">', unsafe_allow_html=True)
            st.markdown("## 🔄 Update Knowledge Base")
            st.markdown("Add more PDF files to an existing knowledge base")
            
            with st.form("update_kb_form"):
                update_kb_names = [kb["name"] for kb in available_kbs]
                selected_update_kb = st.selectbox(
                    "Select knowledge base to update:",
                    update_kb_names,
                    key="update_kb_selector"
                )
                
                additional_files = st.file_uploader(
                    "Upload Additional PDF Files",
                    type=['pdf'],
                    accept_multiple_files=True,
                    key="update_files"
                )
                
                st.info("📋 **New files will be added to the existing knowledge base**")
                
                if st.form_submit_button("Update Knowledge Base"):
                    if not additional_files:
                        st.error("Please upload at least one PDF file")
                    else:
                        # Find the selected knowledge base
                        kb_to_update = next(kb for kb in available_kbs if kb["name"] == selected_update_kb)
                        success = update_knowledge_base_with_pdfs(kb_to_update["path"], additional_files)
                        if success:
                            st.success(f"Knowledge base '{selected_update_kb}' updated successfully!")
                            time.sleep(2)
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="create-kb-section">', unsafe_allow_html=True)
            st.markdown("## 🔄 Update Knowledge Base")
            st.info("💡 Create a knowledge base first to enable updates")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# -------------------- Main Application --------------------
st.set_page_config(
    page_title=st.session_state.selected_knowledge_base['name'],
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

st.markdown("""
    <style>
    .reference {
        color: blue;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Get API configuration to set default models
from agentic_rag import get_api_config
api_config = get_api_config()

model_default = api_config['model']  # Will be gpt-4.1 for Azure or gpt-4.1-2025-04-14 for OpenAI
other_model = api_config['model']

# -------------------- Sidebar --------------------
with st.sidebar:
    st.title("🧠 Knowledge Assistant")
    st.markdown(f"**Knowledge Base:** {st.session_state.selected_knowledge_base['name']}")
    
    st.markdown("**⚙️ Settings:**")
    
    # Model settings
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_default
    if "selected_routing_model" not in st.session_state:
        st.session_state.selected_routing_model = other_model
    if "selected_grading_model" not in st.session_state:
        st.session_state.selected_grading_model = other_model
    if "selected_embedding_model" not in st.session_state:
        st.session_state.selected_embedding_model = "text-embedding-3-large"
    
    answer_style = st.select_slider(
        "💬 Answer Style",
        options=["Concise", "Moderate", "Explanatory"],
        value="Concise",
        key="answer_style_slider"
    )
    st.session_state.answer_style = answer_style
    
    # Search options - show all options but disable internet search for Azure
    if st.session_state.get('use_azure', True):
        # Show all options but mark internet search as disabled for Azure
        st.info("ℹ️ **Note**: Internet search in this application is currently only available with OpenAI API. Select OpenAI to enable internet search options.")
        search_option = st.radio(
            "🔍 Search Options",
            ["Knowledge base only", "🔒 Internet search only (disabled)", "🔒 Knowledge base + Internet search (disabled)"],
            index=0,
            help="Internet search disabled when using Azure endpoint"
        )
        
        # Show warning if user selects disabled options
        if "disabled" in search_option:
            st.warning("⚠️ Internet search options are currently disabled with Azure API. Using Knowledge base only instead.")
        
        # Always use knowledge base only for Azure
        search_option = "Knowledge base only"
    else:
        # Full options for OpenAI
        search_option = st.radio(
            "🔍 Search Options",
            ["Knowledge base only", "Internet search only", "Knowledge base + Internet search"],
            index=0,  # Default to knowledge base only
            help="Choose your search strategy"
        )
    
    # Set search flags based on selection
    st.session_state.hybrid_search = (search_option == "Knowledge base + Internet search")
    st.session_state.internet_search = (search_option == "Internet search only")
    
    # Show explanation based on selected option
    if search_option == "Knowledge base only":
        st.info("📚 **Knowledge base only**: Search only in your uploaded documents")
    elif search_option == "Internet search only":
        st.info("🌐 **Internet search only**: Get current information from the web")
    else:
        st.info("🔄 **Hybrid search**: Combines both knowledge base and internet results with separate sections")
    
    # Action buttons
    if st.button("🔄 Reset Chat", key="reset_button", use_container_width=True):
        st.session_state.messages = []
    
    if st.button("📚 Change Knowledge Base", key="change_kb", use_container_width=True):
        st.session_state.selected_knowledge_base = None
        for key in list(st.session_state.keys()):
            if key not in ["_is_running", "_script_run_ctx"]:
                del st.session_state[key]
        st.rerun()
    
    
    # Initialize RAG workflow
    try:
        app = initialize_generic_app(
            st.session_state.selected_model,
            st.session_state.selected_embedding_model,
            st.session_state.selected_routing_model,
            st.session_state.selected_grading_model,
            st.session_state.selected_knowledge_base['path'],
            st.session_state.get("hybrid_search", False),
            st.session_state.get("internet_search", False),
            st.session_state.answer_style
        )
    except Exception as e:
        st.error("Error initializing model: " + str(e))
        # Use API configuration for fallback
        api_config = get_api_config()
        if api_config['use_azure']:
            st.session_state.llm = AzureChatOpenAI(
                azure_deployment=api_config["deployment_name"],
                api_version=api_config["api_version"],
                temperature=0.5)
        else:
            st.session_state.llm = ChatOpenAI(
                model=api_config['model'], 
                temperature=0.5, 
                api_key=api_config['api_key'])

# -------------------- Main Interface --------------------
st.title(f"🧠 {st.session_state.selected_knowledge_base['name']}")

# Display knowledge base description
kb_description = st.session_state.selected_knowledge_base.get('description', 'No description available')
if kb_description and kb_description != "No description available":
    st.info(f"📝 **About this knowledge base**: {kb_description}")

st.markdown("## 📝 What I can help you with:")
st.markdown("""
- 📚 **Knowledge base search**: Answer questions from your uploaded documents
- 🌐 **Internet search**: Get current information from the web
- 🔄 **Hybrid search**: Combine both sources with separate sections for comprehensive answers
- 💡 Generate insights and analyze information with customizable detail levels
""")

st.markdown("💡 **Pro tip:** Ask specific questions for the most accurate answers!")
st.markdown("**Start by typing your question in the chat below!**")

# Sample questions
st.subheader("Try asking:")
sample_questions = [
    "What are the main topics covered in my documents?",
    "Can you summarize the key points from my knowledge base?",
    "What information do you have about [specific topic]?"
]

cols = st.columns(3)
for i, question in enumerate(sample_questions):
    if cols[i].button(f"💬 {question}", key=f"sample_q_{i}", use_container_width=True):
        st.session_state.pending_followup = question
        st.rerun()

# -------------------- Display Conversation History --------------------
# Display conversation history, excluding the last two messages if they are currently being processed
# This prevents duplicate display during real-time streaming
if len(st.session_state.messages) > 0:
    # Skip the last two messages (current user question and assistant response) if assistant response is incomplete
    messages_to_show = st.session_state.messages
    if (len(messages_to_show) >= 2 and 
        messages_to_show[-1]["role"] == "assistant" and 
        messages_to_show[-2]["role"] == "user" and
        not messages_to_show[-1]["content"].strip()):
        # Currently processing - show all messages except the last incomplete pair
        messages_to_show = messages_to_show[:-2]
    
    for message in messages_to_show:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(f"**You:** {message['content']}")
        elif message["role"] == "assistant" and message["content"].strip():
            with st.chat_message("assistant"):
                styled_response = re.sub(
                    r'\[(.*?)\]',
                    r'<span class="reference">[\1]</span>',
                    message['content']
                )
                st.markdown(
                    f"**Assistant:** {styled_response}",
                    unsafe_allow_html=True
                )


# -------------------- Process Pending Follow-Up --------------------
if st.session_state.pending_followup is not None:
    question = st.session_state.pending_followup
    st.session_state.pending_followup = None
    process_question(question, st.session_state.answer_style)

# -------------------- Process New User Input --------------------
user_input = st.chat_input("Type your question:")
if user_input:
    process_question(user_input, st.session_state.answer_style)

# -------------------- Generate Follow-Up Questions --------------------
def handle_followup(question: str):
    st.session_state.pending_followup = question

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    try:
        last_assistant_message = st.session_state.messages[-1]["content"]

        if (last_assistant_message.strip() and 
            "Sorry, I encountered an error" not in last_assistant_message):
            
            last_user_message = next(
                (msg["content"] for msg in reversed(st.session_state.messages)
                 if msg["role"] == "user"),
                ""
            )

            if st.session_state.last_assistant != last_assistant_message:
                print("Generating new followup questions")
                st.session_state.last_assistant = last_assistant_message
                try:
                    st.session_state.followup_questions = get_followup_questions(
                        last_user_message,
                        last_assistant_message
                    )
                except Exception as e:
                    print(f"Failed to generate followup questions: {e}")
                    st.session_state.followup_questions = []

        if st.session_state.followup_questions and len(st.session_state.followup_questions) > 0:
            st.markdown("#### Related Questions:")
            cols = st.columns(len(st.session_state.followup_questions))
            
            for i, question in enumerate(st.session_state.followup_questions):
                clean_question = re.sub(r'^\d+\.\s*', '', question)
                with cols[i]:
                    if st.button(
                        f"💬 {clean_question}",
                        key=f"followup_{i}_{st.session_state.followup_key}",
                        use_container_width=True
                    ):
                        handle_followup(clean_question)
                        st.rerun()
    except Exception as e:
        print(f"Error in followup section: {e}")
        st.session_state.followup_questions = []

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px;">
        <p>Knowledge Assistant | Powered by AI | &copy; 2025</p>
    </div>
    """,
    unsafe_allow_html=True
)