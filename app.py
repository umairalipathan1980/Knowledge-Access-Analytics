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
from datetime import datetime
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
from user_management import UserManager, APIUsageTracker
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# -------------------- Initialization --------------------
st.set_option("client.showErrorDetails", False)

# Initialize user management
user_manager = UserManager()
usage_tracker = APIUsageTracker()

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

# Authentication session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# -------------------- Authentication Functions --------------------
def show_login_screen():
    """Display login screen for admin and user authentication"""
    st.set_page_config(page_title="Knowledge Assistant - Login", layout="centered", page_icon="🏥")
    
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        border: 1px solid #ddd;
        border-radius: 10px;
        background-color: #f8f9fa;
    }
    .login-title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="login-title">🏥 Knowledge Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="login-title">Please Login to Continue</h3>', unsafe_allow_html=True)
    
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("🔐 Login as User", use_container_width=True):
                if username and password:
                    user = user_manager.authenticate_user(username, password)
                    if user and not user['is_admin']:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user
                        st.success(f"Welcome, {user['name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid credentials or not authorized as user")
                else:
                    st.error("Please enter both username and password")
        
        with col2:
            if st.form_submit_button("⚙️ Login as Admin", use_container_width=True):
                if username and password:
                    user = user_manager.authenticate_user(username, password)
                    if user and user['is_admin']:
                        st.session_state.authenticated = True
                        st.session_state.current_user = user
                        st.success(f"Welcome, Administrator {user['name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Invalid admin credentials")
                else:
                    st.error("Please enter both username and password")
    
    st.markdown("</div>", unsafe_allow_html=True)

def show_admin_interface():
    """Display administrator interface with user management"""
    st.set_page_config(
        page_title="Admin Dashboard - Knowledge Assistant",
        layout="wide",
        page_icon="⚙️"
    )
    
    # Header with logout
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("⚙️ Administrator Dashboard")
        st.markdown(f"**Logged in as:** {st.session_state.current_user['name']}")
    with col2:
        if st.button("🚪 Logout", key="admin_logout"):
            logout_user()
    
    # Admin tabs
    tab1, tab2, tab3, tab4 = st.tabs(["👥 User Management", "📊 Usage Analytics", "💬 Knowledge Base", "⚙️ System Settings"])
    
    with tab1:
        show_user_management()
    
    with tab2:
        show_usage_analytics()
    
    with tab3:
        show_knowledge_base_interface()
    
    with tab4:
        show_system_settings()

def show_user_interface():
    """Display regular user interface with usage tracking"""
    # Show knowledge base selection if none selected
    if st.session_state.selected_knowledge_base is None:
        show_knowledge_base_selection()
        return
    
    # Main user interface (existing RAG functionality)
    show_main_rag_interface()

def show_user_management():
    """User management interface for admins"""
    st.subheader("👥 User Account Management")
    
    # Create new user form
    with st.expander("➕ Create New User Account", expanded=False):
        with st.form("create_user_form"):
            st.subheader("Create New User")
            
            col1, col2 = st.columns(2)
            with col1:
                employee_id = st.text_input("Employee ID*", placeholder="EMP001", key="create_employee_id")
                name = st.text_input("Full Name*", placeholder="John Doe", key="create_name")
                department = st.text_input("Department*", placeholder="e.g., Elderly Care Unit A", key="create_department")
                role = st.text_input("Role*", placeholder="e.g., Nurse", key="create_role")
            
            with col2:
                email = st.text_input("Email*", placeholder="john.doe@healthcare.com", key="create_email")
                username = st.text_input("Username*", placeholder="jdoe", key="create_username")
                password = st.text_input("Password*", type="password", placeholder="Enter secure password", key="create_password")
                confirm_password = st.text_input("Confirm Password*", type="password", key="create_confirm_password")
            
            st.markdown("**Token Limits Configuration:**")
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("*Daily Limits (Reset Daily)*")
                daily_gpt_limit = st.number_input("Daily GPT Token Limit", min_value=1000, value=30000, help="Daily limit for GPT model usage (resets daily)", key="create_daily_gpt_limit")
                daily_embedding_limit = st.number_input("Daily Embedding Token Limit", min_value=500, value=20000, help="Daily limit for embedding model usage (resets daily)", key="create_daily_embedding_limit")
            with col4:
                st.markdown("*Maximum Limits (Total Pool)*")
                max_gpt_limit = st.number_input("Maximum GPT Token Limit", min_value=1000, value=150000, help="Total maximum GPT tokens available", key="create_max_gpt_limit")
                max_embedding_limit = st.number_input("Maximum Embedding Token Limit", min_value=500, value=50000, help="Total maximum embedding tokens available", key="create_max_embedding_limit")
            
            # Calculate totals for backward compatibility
            daily_limit = daily_gpt_limit + daily_embedding_limit
            max_limit = max_gpt_limit + max_embedding_limit
            
            if st.form_submit_button("➕ Create User Account", use_container_width=True):
                # Validation
                if not all([employee_id, name, department, role, email, username, password]):
                    st.error("All fields marked with * are required")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters long")
                else:
                    # Create user
                    success = user_manager.create_user(
                        employee_id=employee_id,
                        name=name,
                        department=department,
                        role=role,
                        email=email,
                        username=username,
                        password=password,
                        daily_token_limit=daily_limit,
                        maximum_token_limit=max_limit,
                        daily_gpt_limit=daily_gpt_limit,
                        daily_embedding_limit=daily_embedding_limit,
                        max_gpt_limit=max_gpt_limit,
                        max_embedding_limit=max_embedding_limit,
                        created_by=st.session_state.current_user['username']
                    )
                    
                    if success:
                        st.success(f"User account created successfully for {name}")
                        time.sleep(1)
                        st.rerun()
    
    # Display existing users
    st.subheader("📋 Existing Users")
    users = user_manager.get_all_users()
    
    if users:
        user_df = pd.DataFrame([
            {
                'Employee ID': user['employee_id'],
                'Name': user['name'],
                'Department': user['department'],
                'Role': user['role'],
                'Username': user['username'],
                'Email': user['email'],
                'Daily Limit': user['daily_token_limit'],
                'Max Limit': user['maximum_token_limit'],
                'Created': user['created_at'][:10] if user['created_at'] else 'N/A',
                'Status': 'Active' if user['is_active'] else 'Inactive'
            }
            for user in users
        ])
        
        st.dataframe(user_df, use_container_width=True)
        st.info(f"Total Users: {len(users)}")
        
        # User management actions
        st.subheader("🛠️ User Management Actions")
        with st.expander("⚠️ Delete User Account", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # Regular users only (exclude admins)
                regular_users = [user for user in users if not user.get('is_admin', False)]
                if regular_users:
                    user_options = [f"{user['name']} ({user['username']})" for user in regular_users]
                    selected_user_display = st.selectbox(
                        "Select user to delete:",
                        user_options,
                        key="delete_user_select"
                    )
                    
                    # Find the selected user
                    selected_user = next(
                        user for user in regular_users 
                        if f"{user['name']} ({user['username']})" == selected_user_display
                    )
                    
                    st.warning("⚠️ **Deactivate User**: This will disable the user's access but preserve all data.")
                    
                    if st.button("🚫 Deactivate User", key="deactivate_user"):
                        success, message = user_manager.delete_user(
                            selected_user['id'], 
                            st.session_state.current_user['username']
                        )
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.info("No regular users available to delete (admin accounts cannot be deleted).")
            
            with col2:
                if regular_users:
                    st.error("⚠️ **Permanently Delete**: This will remove all user data and cannot be undone!")
                    
                    # Confirmation checkbox
                    confirm_permanent = st.checkbox(
                        "I understand this action cannot be undone",
                        key="confirm_permanent_delete"
                    )
                    
                    if st.button(
                        "🗑️ Permanently Delete User", 
                        key="permanent_delete_user",
                        disabled=not confirm_permanent
                    ):
                        success, message = user_manager.permanently_delete_user(
                            selected_user['id'], 
                            st.session_state.current_user['username']
                        )
                        if success:
                            st.success(message)
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(message)
        
        # User modification section
        st.subheader("✏️ Modify User Account")
        with st.expander("✏️ Edit User Account Fields", expanded=False):
            if users:
                # Select user to modify
                regular_users = [user for user in users if not user.get('is_admin', False)]
                if regular_users:
                    user_options = [f"{user['name']} ({user['username']})" for user in regular_users]
                    selected_user_display = st.selectbox(
                        "Select user to modify:",
                        user_options,
                        key="modify_user_select"
                    )
                    
                    # Find the selected user
                    selected_user = next(
                        user for user in regular_users 
                        if f"{user['name']} ({user['username']})" == selected_user_display
                    )
                    
                    with st.form("modify_user_form"):
                        st.markdown(f"**Editing: {selected_user['name']} ({selected_user['username']})**")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            new_employee_id = st.text_input("Employee ID", value=selected_user['employee_id'], key="modify_employee_id")
                            new_name = st.text_input("Full Name", value=selected_user['name'], key="modify_name")
                            new_department = st.text_input("Department", value=selected_user['department'], key="modify_department")
                            new_role = st.text_input("Role", value=selected_user['role'], key="modify_role")
                        
                        with col2:
                            new_email = st.text_input("Email", value=selected_user['email'], key="modify_email")
                        
                        st.markdown("**Token Limits:**")
                        col_daily, col_max = st.columns(2)
                        with col_daily:
                            st.markdown("*Daily Limits*")
                            # Get current limits with fallback
                            current_daily_gpt = selected_user.get('daily_gpt_limit', 30000)
                            current_daily_embedding = selected_user.get('daily_embedding_limit', 20000)
                            
                            new_daily_gpt_limit = st.number_input(
                                "Daily GPT Tokens", 
                                min_value=1000, 
                                value=current_daily_gpt,
                                key="modify_daily_gpt"
                            )
                            new_daily_embedding_limit = st.number_input(
                                "Daily Embedding Tokens", 
                                min_value=500, 
                                value=current_daily_embedding,
                                key="modify_daily_embedding"
                            )
                        
                        with col_max:
                            st.markdown("*Maximum Limits*")
                            current_max_gpt = selected_user.get('max_gpt_limit', 150000)
                            current_max_embedding = selected_user.get('max_embedding_limit', 50000)
                            
                            new_max_gpt_limit = st.number_input(
                                "Maximum GPT Tokens", 
                                min_value=1000,  # Changed from new_daily_gpt_limit to avoid circular dependency
                                value=current_max_gpt,
                                key="modify_max_gpt"
                            )
                            new_max_embedding_limit = st.number_input(
                                "Maximum Embedding Tokens", 
                                min_value=500,  # Changed from new_daily_embedding_limit to avoid circular dependency
                                value=current_max_embedding,
                                key="modify_max_embedding"
                            )
                        
                        if st.form_submit_button("💾 Update User Account", use_container_width=True):
                            # Prepare update data
                            update_data = {
                                'employee_id': new_employee_id,
                                'name': new_name,
                                'department': new_department,
                                'role': new_role,
                                'email': new_email,
                                'daily_gpt_limit': new_daily_gpt_limit,
                                'daily_embedding_limit': new_daily_embedding_limit,
                                'daily_token_limit': new_daily_gpt_limit + new_daily_embedding_limit,
                                'maximum_token_limit': new_max_gpt_limit + new_max_embedding_limit,
                                'max_gpt_limit': new_max_gpt_limit,
                                'max_embedding_limit': new_max_embedding_limit
                            }
                            
                            success, message = user_manager.update_user(selected_user['id'], **update_data)
                            if success:
                                st.success(message)
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error(message)
                else:
                    st.info("No regular users available to modify (admin accounts cannot be modified).")
            else:
                st.info("No users found to modify.")
    else:
        st.info("No users found. Create the first user account above.")

def show_usage_analytics():
    """Usage analytics dashboard for admins"""
    st.subheader("📊 API Usage Analytics")
    
    # Date selector
    date_col1, date_col2 = st.columns(2)
    with date_col1:
        selected_date = st.date_input("Select Date", value=datetime.now().date())
    
    # Get usage data
    usage_data = usage_tracker.get_all_users_usage(selected_date.isoformat())
    
    if usage_data:
        # Summary metrics
        total_cost = sum(user['total_cost'] for user in usage_data)
        total_tokens = sum(user['total_tokens'] for user in usage_data)
        total_calls = sum(user['api_calls'] for user in usage_data)
        active_users = len([user for user in usage_data if user['api_calls'] > 0])
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cost", f"${total_cost:.3f}")
        with col2:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col3:
            st.metric("API Calls", total_calls)
        with col4:
            st.metric("Active Users", active_users)
        
        # Usage by department
        df = pd.DataFrame(usage_data)
        if not df.empty and 'department' in df.columns:
            dept_usage = df.groupby('department').agg({
                'total_cost': 'sum',
                'total_tokens': 'sum',
                'api_calls': 'sum'
            }).reset_index()
            
            col1, col2 = st.columns(2)
            with col1:
                fig_cost = px.bar(dept_usage, x='department', y='total_cost', 
                                title='Cost by Department')
                st.plotly_chart(fig_cost, use_container_width=True)
            
            with col2:
                fig_tokens = px.bar(dept_usage, x='department', y='total_tokens',
                                  title='Tokens by Department')
                st.plotly_chart(fig_tokens, use_container_width=True)
        
        # Detailed user usage table
        st.subheader("📋 Detailed User Usage")
        user_usage_df = pd.DataFrame([
            {
                'Name': user['name'],
                'Department': user['department'],
                'Role': user['role'],
                'Tokens Used': user['total_tokens'],
                'Cost': f"${user['total_cost']:.3f}",
                'API Calls': user['api_calls'],
                'Usage %': f"{user['token_usage_percent']:.1f}%",
                'Daily Limit': user['daily_token_limit']
            }
            for user in usage_data
        ])
        
        st.dataframe(user_usage_df, use_container_width=True)
    else:
        st.info(f"No usage data found for {selected_date}")

def show_user_usage_panel():
    """Show usage panel for regular users in sidebar"""
    if st.session_state.current_user:
        st.markdown("📊 **Token Usage:**")
        
        user_id = st.session_state.current_user['id']
        daily_usage = usage_tracker.get_user_daily_usage(user_id)
        total_usage = usage_tracker.get_user_total_usage(user_id)
        
        # Get user limits (with fallback for backward compatibility)
        user = st.session_state.current_user
        daily_gpt_limit = user.get('daily_gpt_limit', 30000)
        daily_embedding_limit = user.get('daily_embedding_limit', 20000)
        max_gpt_limit = user.get('max_gpt_limit', 150000)
        max_embedding_limit = user.get('max_embedding_limit', 50000)
        
        # GPT Model Usage
        st.markdown("🤖 **GPT Model:**")
        gpt_daily_used = daily_usage.get('gpt_tokens', 0)
        gpt_total_used = total_usage.get('total_gpt_tokens', 0)
        gpt_daily_remaining = max(0, daily_gpt_limit - gpt_daily_used)
        gpt_total_remaining = max(0, max_gpt_limit - gpt_total_used)
        gpt_total_percent = (gpt_total_used / max_gpt_limit * 100) if max_gpt_limit > 0 else 0
        
        st.text(f"Today: {gpt_daily_used:,} / {daily_gpt_limit:,}")
        st.text(f"Total: {gpt_total_used:,} / {max_gpt_limit:,}")
        st.text(f"Remaining: {gpt_total_remaining:,}")
        if gpt_total_percent >= 90:
            st.text(f"⚠️ {gpt_total_percent:.1f}% of max used")
        else:
            st.text(f"✅ {gpt_total_percent:.1f}% of max used")
        st.progress(min(gpt_total_percent / 100, 1.0))
        
        st.markdown("")  # Small space
        
        # Embedding Model Usage
        st.markdown("🔍 **Embedding Model:**")
        embedding_daily_used = daily_usage.get('embedding_tokens', 0)
        embedding_total_used = total_usage.get('total_embedding_tokens', 0)
        embedding_daily_remaining = max(0, daily_embedding_limit - embedding_daily_used)
        embedding_total_remaining = max(0, max_embedding_limit - embedding_total_used)
        embedding_total_percent = (embedding_total_used / max_embedding_limit * 100) if max_embedding_limit > 0 else 0
        
        st.text(f"Today: {embedding_daily_used:,} / {daily_embedding_limit:,}")
        st.text(f"Total: {embedding_total_used:,} / {max_embedding_limit:,}")
        st.text(f"Remaining: {embedding_total_remaining:,}")
        if embedding_total_percent >= 90:
            st.text(f"⚠️ {embedding_total_percent:.1f}% of max used")
        else:
            st.text(f"✅ {embedding_total_percent:.1f}% of max used")
        st.progress(min(embedding_total_percent / 100, 1.0))
        
        st.markdown("")  # Small space
        
        # Total cost
        st.text(f"💰 Total Cost: ${total_usage['total_cost']:.4f}")
        st.text(f"💰 Cost Today: ${daily_usage['total_cost']:.4f}")

def logout_user():
    """Logout current user"""
    for key in ['authenticated', 'current_user', 'selected_knowledge_base']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

def load_chat_history_for_user():
    """Load chat history from database for current user and knowledge base"""
    if (st.session_state.get('current_user') and 
        st.session_state.get('selected_knowledge_base')):
        
        user_id = st.session_state.current_user['id']
        kb_name = st.session_state.selected_knowledge_base['name']
        
        # Load chat history from database
        chat_history = user_manager.load_chat_history(
            user_id=user_id,
            knowledge_base=kb_name,
            limit=100  # Load last 100 messages
        )
        
        # Convert to session state format
        st.session_state.messages = []
        for msg in chat_history:
            st.session_state.messages.append({
                'role': msg['role'],
                'content': msg['content']
            })

def save_message_to_history(role: str, content: str):
    """Save a message to the database chat history"""
    if (st.session_state.get('current_user') and 
        st.session_state.get('selected_knowledge_base')):
        
        user_id = st.session_state.current_user['id']
        kb_name = st.session_state.selected_knowledge_base['name']
        
        user_manager.save_chat_message(
            user_id=user_id,
            message_role=role,
            message_content=content,
            knowledge_base=kb_name
        )

def show_knowledge_base_interface():
    """Knowledge base interface for admin"""
    st.subheader("💬 Knowledge Base Management")
    st.info("This tab provides the same knowledge base functionality as regular users, with administrative privileges.")
    
    # Show knowledge base selection if none selected
    if st.session_state.selected_knowledge_base is None:
        show_kb_selection_content()
        return
    
    # Show RAG interface
    show_rag_content()

def show_system_settings():
    """System settings for admin"""
    st.subheader("⚙️ System Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔧 API Configuration**")
        from agentic_rag import get_api_config
        api_config = get_api_config()
        
        st.text(f"Current API: {'Azure' if api_config['use_azure'] else 'OpenAI'}")
        st.text(f"Model: {api_config['model']}")
        st.text(f"API Key Status: {'✅ Configured' if api_config['api_key'] else '❌ Missing'}")
        
        st.markdown("**💾 Database Status**")
        try:
            users = user_manager.get_all_users()
            st.text(f"Total Users: {len(users)}")
            st.text(f"Database: ✅ Connected")
        except Exception as e:
            st.text(f"Database: ❌ Error - {str(e)[:50]}")
    
    with col2:
        st.markdown("**📊 Usage Limits**")
        st.text("Default Daily Token Limit: 50,000")
        st.text("Default Maximum Token Limit: 200,000")
        st.text("Rate Limiting: Enabled")
        
        st.markdown("**🛡️ Security**")
        st.text("Password Hashing: PBKDF2 (100k iterations)")
        st.text("Session Management: Streamlit Session State")
        st.text("Admin Account: ✅ Active")
    
    # Token Calculation Method Setting
    st.markdown("---")
    st.markdown("**🔢 Token Calculation Settings**")
    
    # Get current setting
    current_method = usage_tracker.get_system_setting('token_calculation_method')
    method_index = 0 if current_method == 'rough_estimate' else 1
    
    # Create form for token calculation method
    with st.form("token_calculation_form"):
        st.markdown("Select the token calculation method used system-wide:")
        
        col3, col4 = st.columns(2)
        with col3:
            token_method = st.radio(
                "Token Calculation Method",
                ["Rough Estimation (Fast)", "OpenAI tiktoken (Precise)"],
                index=method_index,
                key="token_method_radio",
                help="Rough estimation is faster but less accurate. tiktoken provides precise token counts but is slower."
            )
        
        with col4:
            st.markdown("**Method Details:**")
            if token_method == "Rough Estimation (Fast)":
                st.markdown("- Uses word count × 1.33 approximation")
                st.markdown("- Very fast computation")
                st.markdown("- ~10-15% margin of error")
            else:
                st.markdown("- Uses OpenAI's official tiktoken library")
                st.markdown("- Slower computation")
                st.markdown("- Exact token count accuracy")
        
        if st.form_submit_button("💾 Update Token Calculation Method", use_container_width=True):
            try:
                method_value = 'rough_estimate' if token_method == "Rough Estimation (Fast)" else 'tiktoken'
                usage_tracker.update_system_setting(
                    'token_calculation_method', 
                    method_value
                )
                st.success(f"✅ Token calculation method updated to: {token_method}")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error updating token calculation method: {str(e)}")

def show_kb_selection_content():
    """Content for knowledge base selection"""
    st.markdown("<h1 style='text-align: center;'>🧠 Knowledge Base Manager</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Select a knowledge base or create a new one from your documents</p>", unsafe_allow_html=True)
    
    # API Configuration Section
    st.markdown("## ⚙️ API Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        selected_option = st.radio(
            "Select API Provider",
            ["Microsoft Azure Endpoint", "🔒 OpenAI API Key (disabled)"],
            index=0,
            key="api_provider_choice"
        )
        
        if selected_option == "🔒 OpenAI API Key (disabled)":
            st.warning("⚠️ OpenAI API option is currently disabled. Using Azure instead.")
        
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
            # Load chat history for this user and knowledge base
            load_chat_history_for_user()
            st.rerun()
    
    # Create and update sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## ➕ Create New Knowledge Base")
        with st.form("create_kb_form"):
            kb_name = st.text_input("Knowledge Base Name", placeholder="e.g., Healthcare Guidelines")
            kb_description = st.text_area(
                "Knowledge Base Description",
                placeholder="Brief description (max 200 characters)",
                max_chars=200
            )
            uploaded_files = st.file_uploader(
                "Upload PDF Files",
                type=['pdf'],
                accept_multiple_files=True,
                key="create_files"
            )
            
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
                        # Load chat history (will be empty for new KB)
                        load_chat_history_for_user()
                        st.rerun()
    
    with col2:
        if available_kbs:
            st.markdown("## 🔄 Update Knowledge Base")
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
                
                if st.form_submit_button("Update Knowledge Base"):
                    if not additional_files:
                        st.error("Please upload at least one PDF file")
                    else:
                        kb_to_update = next(kb for kb in available_kbs if kb["name"] == selected_update_kb)
                        success = update_knowledge_base_with_pdfs(kb_to_update["path"], additional_files)
                        if success:
                            st.success(f"Knowledge base '{selected_update_kb}' updated successfully!")
                            # Automatically select the updated knowledge base
                            st.session_state.selected_knowledge_base = {
                                "name": kb_to_update["name"],
                                "path": kb_to_update["path"],
                                "description": kb_to_update.get("description", "No description available")
                            }
                            # Load chat history for this knowledge base
                            load_chat_history_for_user()
                            st.rerun()
        else:
            st.markdown("## 🔄 Update Knowledge Base")
            st.info("💡 Create a knowledge base first to enable updates")

def show_main_rag_interface():
    """Main RAG interface for authenticated users"""
    # Header
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(f"🧠 {st.session_state.selected_knowledge_base['name']}")
        st.markdown(f"**Welcome, {st.session_state.current_user['name']}** | {st.session_state.current_user['role']}")
    with col2:
        if st.button("🚪 Logout", key="user_logout"):
            logout_user()
    
    show_rag_content()

def show_rag_content():
    """Shared RAG content for both admin and users"""
    
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

    # Display conversation history
    if len(st.session_state.messages) > 0:
        messages_to_show = st.session_state.messages
        if (len(messages_to_show) >= 2 and 
            messages_to_show[-1]["role"] == "assistant" and 
            messages_to_show[-2]["role"] == "user" and
            not messages_to_show[-1]["content"].strip()):
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

    # Process pending follow-up
    if st.session_state.pending_followup is not None:
        question = st.session_state.pending_followup
        st.session_state.pending_followup = None
        process_question_with_tracking(question, st.session_state.answer_style)

    # Process new user input
    user_input = st.chat_input("Type your question:")
    if user_input:
        process_question_with_tracking(user_input, st.session_state.answer_style)

    # Generate follow-up questions
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
                            st.session_state.pending_followup = clean_question
                            st.rerun()
        except Exception as e:
            print(f"Error in followup section: {e}")
            st.session_state.followup_questions = []

def process_question_with_tracking(question, answer_style):
    """Process question with API usage tracking"""
    user_id = st.session_state.current_user['id']
    
    # Check limits before processing
    user = st.session_state.current_user
    limits = usage_tracker.check_user_limits(
        user_id,
        user.get('daily_gpt_limit', 1000000),
        user.get('daily_embedding_limit', 1000000),
        user.get('max_gpt_limit', 5000000),
        user.get('max_embedding_limit', 5000000)
    )
    
    if not limits['gpt_within_daily_limit'] or not limits['gpt_within_max_limit']:
        st.error("⚠️ GPT token limit exceeded. Please contact your administrator.")
        return
    
    if not limits['embedding_within_daily_limit'] or not limits['embedding_within_max_limit']:
        st.error("⚠️ Embedding token limit exceeded. Please contact your administrator.")
        return
    
    # Process the question (existing logic)
    st.session_state.messages.append({"role": "user", "content": question})
    # Save user message to database
    save_message_to_history("user", question)
    
    with st.chat_message("user"):
        st.markdown(f"**You:** {question}")

    output_buffer = io.StringIO()
    sys.stdout = output_buffer
    assistant_response = ""

    st.session_state.messages.append({"role": "assistant", "content": ""})
    assistant_index = len(st.session_state.messages) - 1

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
    sys.stdout = sys.__stdout__

    st.session_state.messages[assistant_index]["content"] = assistant_response
    # Save assistant response to database
    save_message_to_history("assistant", assistant_response)
    st.session_state.followup_key += 1

    # Track API usage
    if assistant_response and assistant_response.strip():
        # Get actual model names being used
        from agentic_rag import get_api_config
        api_config = get_api_config()
        main_model = st.session_state.get('selected_model', api_config['model'])
        embedding_model = st.session_state.get('selected_embedding_model', 'text-embedding-3-large')
        
        # Log the usage
        usage_tracker.log_api_usage(
            user_id=user_id,
            model=main_model,  # Use actual model from session state
            operation_type="chat",
            input_text=question,
            output_text=assistant_response,
            query_type="rag_query",
            knowledge_base=st.session_state.selected_knowledge_base['name']
        )
        
        # Also track embedding usage (estimated)
        usage_tracker.log_api_usage(
            user_id=user_id,
            model=embedding_model,  # Use actual embedding model
            operation_type="embedding",
            input_text=question,
            output_text="",
            query_type="document_search",
            knowledge_base=st.session_state.selected_knowledge_base['name']
        )
        
        # Refresh UI to immediately show updated token usage
        st.rerun()

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

        print("Updating knowledge base. This can take a while...\n")
        for i, pdf_file in enumerate(pdf_files):
            status_text.text("Creating knowledge base. This can take a while...")
            status_text.text(f"Processing {pdf_file.name} with metadata extraction...")
            
            # Save uploaded file temporarily
            temp_path = data_folder / pdf_file.name
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Parse PDF to chunks with metadata (document name, page numbers, etc.)
            chunks_with_metadata = parser.convert_pdf_to_chunks_with_metadata(str(temp_path))
            new_documents.extend(chunks_with_metadata)
            
            # # Save parsed document as markdown (optional - can be disabled for performance)
            # if st.session_state.get('save_markdown', False):  # Only save if explicitly enabled
            #     kb_base_path = Path(kb_path)
            #     md_output_path = kb_base_path / f"{pdf_file.name.replace('.pdf', '')}.md"
            #     try:
            #         parser.convert_pdf_to_markdown(str(temp_path), str(md_output_path))
            #         print(f"Saved markdown: {md_output_path}")
            #     except Exception as e:
            #         print(f"Warning: Could not save markdown for {pdf_file.name}: {e}")
            
            # Clean up temp file
            temp_path.unlink()
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        print("Creating embeddings...")
        status_text.text("Creating embeddings...")
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
        
        print("\nUpdating vector database...")
        status_text.text("Creating vector database...")
        # Load existing vector store
        existing_vectorstore = Chroma(
            persist_directory=kb_path,
            embedding_function=embeddings,
            collection_name="rag"
        )
        
        # Add new documents to existing vector store
        existing_vectorstore.add_documents(new_documents)
        
        # Track API usage for embedding creation (accurate token counting)
        if 'current_user' in st.session_state and st.session_state.current_user:
            # Get embedding model name
            embedding_model = st.session_state.get('selected_embedding_model', 'text-embedding-3-large')
            
            # Calculate actual tokens from document content
            total_text = ""
            for doc in new_documents:
                total_text += doc.page_content + " "
            
            # Pre-calculate tokens for performance (avoid storing massive text)
            input_tokens = usage_tracker.estimate_tokens(total_text, embedding_model)
            
            # Log usage with pre-calculated tokens
            usage_tracker.log_api_usage(
                user_id=st.session_state.current_user['id'],
                model=embedding_model,
                operation_type="embedding",
                input_text="",  # Don't store massive text
                input_tokens=input_tokens,  # Use pre-calculated tokens
                output_tokens=0,
                query_type="knowledge_base_update",
                knowledge_base=kb_path
            )
        
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
        
        # Store PDF files for markdown conversion after persistence directory is created
        temp_files_info = []
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text("Creating knowledge base...")
            status_text.text(f"Processing {pdf_file.name} with metadata extraction. This may take a while.")
            
            # Save uploaded file temporarily
            temp_path = data_folder / pdf_file.name
            with open(temp_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            
            # Store temp path and filename for later markdown saving
            temp_files_info.append((str(temp_path), pdf_file.name))
            
            # Parse PDF to chunks with metadata (document name, page numbers, etc.)
            chunks_with_metadata = parser.convert_pdf_to_chunks_with_metadata(str(temp_path))
            all_documents.extend(chunks_with_metadata)
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        # Use the chunks directly (already split with metadata preserved)
        splits = all_documents
        
        status_text.text("Creating knowledge base...")
        status_text.text("Creating embeddings...")
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
        
        status_text.text("Creating knowledge base...")
        status_text.text("Creating vector database...")
        # Create Chroma database
        persist_directory = data_folder / f"chroma_db_{kb_name.lower().replace(' ', '-')}"
        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(persist_directory),
            collection_name="rag"
        )
        
        # Track API usage for embedding creation (accurate token counting)
        if 'current_user' in st.session_state and st.session_state.current_user:
            # Get embedding model name
            embedding_model = st.session_state.get('selected_embedding_model', 'text-embedding-3-large')
            
            # Calculate actual tokens from document content
            total_text = ""
            for doc in splits:
                total_text += doc.page_content + " "
            
            # Pre-calculate tokens for performance (avoid storing massive text)
            input_tokens = usage_tracker.estimate_tokens(total_text, embedding_model)
            
            # Log usage with pre-calculated tokens
            usage_tracker.log_api_usage(
                user_id=st.session_state.current_user['id'],
                model=embedding_model,
                operation_type="embedding",
                input_text="",  # Don't store massive text
                input_tokens=input_tokens,  # Use pre-calculated tokens
                output_tokens=0,
                query_type="knowledge_base_creation",
                knowledge_base=kb_name
            )
        
        # # Save parsed documents as markdown files (optional - can be disabled for performance)  
        # if st.session_state.get('save_markdown', False):  # Only save if explicitly enabled
        #     for temp_path, pdf_filename in temp_files_info:
        #         try:
        #             md_output_path = persist_directory / f"{pdf_filename.replace('.pdf', '')}.md"
        #             parser.convert_pdf_to_markdown(temp_path, str(md_output_path))
        #             print(f"Saved markdown: {md_output_path}")
        #         except Exception as e:
        #             print(f"Warning: Could not save markdown for {pdf_filename}: {e}")
        
        # Clean up temp files
        for temp_path, _ in temp_files_info:
            try:
                Path(temp_path).unlink()
            except Exception as e:
                print(f"Warning: Could not clean up temp file {temp_path}: {e}")
        
        # Save description metadata
        description_file = persist_directory / "description.txt"
        with open(description_file, "w", encoding="utf-8") as f:
            f.write(kb_description.strip() if kb_description else "No description provided")
        
        status_text.text("Knowledge base crated successfully!")
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

# Add remaining interface functions
def show_knowledge_base_selection():
    """Knowledge base selection interface"""
    
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
    
    st.markdown("<h1 style='text-align: center;'>🧠 Intelligent Knowledge Access</h1>", unsafe_allow_html=True)
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
            # Load chat history for this user and knowledge base
            load_chat_history_for_user()
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
                            # Automatically select the updated knowledge base
                            st.session_state.selected_knowledge_base = {
                                "name": kb_to_update["name"],
                                "path": kb_to_update["path"],
                                "description": kb_to_update.get("description", "No description available")
                            }
                            # Load chat history for this knowledge base
                            load_chat_history_for_user()
                            time.sleep(2)
                            st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="create-kb-section">', unsafe_allow_html=True)
            st.markdown("## 🔄 Update Knowledge Base")
            st.info("💡 Create a knowledge base first to enable updates")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# -------------------- Main Application Flow --------------------

# Check authentication first
if not st.session_state.authenticated:
    show_login_screen()
else:
    # User is authenticated, check if admin or regular user
    current_user = st.session_state.current_user
    
    if current_user['is_admin']:
        # Admin interface
        show_admin_interface()
    else:
        # Regular user interface
        if st.session_state.selected_knowledge_base is None:
            # Show knowledge base selection with page config
            st.set_page_config(page_title="Knowledge Base Selection", layout="centered", page_icon="🧠")
            show_kb_selection_content()
        else:
            # Show main RAG interface with sidebar
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
            
            # Sidebar with settings and usage tracking
            with st.sidebar:
                st.title("🧠 Knowledge Assistant")
                st.markdown(f"**Knowledge Base:** {st.session_state.selected_knowledge_base['name']}")
                
                # User info
                st.markdown(f"**User:** {current_user['name']}")
                st.markdown(f"**Department:** {current_user['department']}")
                st.markdown(f"**Role:** {current_user['role']}")
                
                st.divider()
                
                # Usage tracking panel
                show_user_usage_panel()
                
                st.divider()
                
                # Settings
                st.markdown("**⚙️ Settings:**")
                
                # Model settings
                if "selected_model" not in st.session_state:
                    from agentic_rag import get_api_config
                    api_config = get_api_config()
                    st.session_state.selected_model = api_config['model']
                    st.session_state.selected_routing_model = api_config['model']
                    st.session_state.selected_grading_model = api_config['model']
                    st.session_state.selected_embedding_model = "text-embedding-3-large"
                
                answer_style = st.select_slider(
                    "💬 Answer Style",
                    options=["Concise", "Moderate", "Explanatory"],
                    value="Concise",
                    key="answer_style_slider"
                )
                st.session_state.answer_style = answer_style
                
                # Search options
                if st.session_state.get('use_azure', True):
                    # st.info("ℹ️ **Note**: Internet search in this application is currently only available with OpenAI API. Select OpenAI to enable internet search options.")
                    search_option = st.radio(
                        "🔍 Search Options",
                        ["Knowledge base only", "🔒 Internet search only (disabled)", "🔒 Knowledge base + Internet search (disabled)"],
                        index=0,
                        help="Internet search disabled when using Azure endpoint"
                    )
                    
                    if "disabled" in search_option:
                        st.warning("⚠️ Internet search options are currently disabled with Azure API. Using Knowledge base only instead.")
                    
                    search_option = "Knowledge base only"
                else:
                    search_option = st.radio(
                        "🔍 Search Options",
                        ["Knowledge base only", "Internet search only", "Knowledge base + Internet search"],
                        index=0,
                        help="Choose your search strategy"
                    )
                
                # Set search flags
                st.session_state.hybrid_search = (search_option == "Knowledge base + Internet search")
                st.session_state.internet_search = (search_option == "Internet search only")
                
                # Show explanation
                if search_option == "Knowledge base only":
                    st.info("📚 **Knowledge base only**: Search only in your uploaded documents")
                elif search_option == "Internet search only":
                    st.info("🌐 **Internet search only**: Get current information from the web")
                else:
                    st.info("🔄 **Hybrid search**: Combines both knowledge base and internet results with separate sections")
                
                st.divider()
                
                # Action buttons
                if st.button("🔄 Reset Chat", key="reset_button", use_container_width=True):
                    # Clear session state messages
                    st.session_state.messages = []
                    # Clear database chat history for current user and knowledge base
                    user_manager.clear_chat_history(
                        user_id=current_user['id'],
                        knowledge_base=st.session_state.selected_knowledge_base['name']
                    )
                    st.rerun()
                
                if st.button("📚 Change Knowledge Base", key="change_kb", use_container_width=True):
                    st.session_state.selected_knowledge_base = None
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
                    from agentic_rag import get_api_config
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
            
            # Main content area with RAG interface
            show_main_rag_interface()

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
