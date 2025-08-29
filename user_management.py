"""
User Management System for Healthcare RAG Application
Handles authentication, user creation, and API usage tracking
"""

import sqlite3
import hashlib
import secrets
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, List
import streamlit as st


class UserManager:
    def __init__(self, db_path: str = "data/users.db"):
        self.db_path = db_path
        # Create data directory if it doesn't exist
        Path("data").mkdir(exist_ok=True)
        self.init_database()
        self.init_admin_user()
    
    def init_database(self):
        """Initialize the user management database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            department TEXT NOT NULL,
            role TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            daily_token_limit INTEGER DEFAULT 50000,
            maximum_token_limit INTEGER DEFAULT 200000,
            daily_gpt_limit INTEGER DEFAULT 30000,
            daily_embedding_limit INTEGER DEFAULT 20000,
            max_gpt_limit INTEGER DEFAULT 150000,
            max_embedding_limit INTEGER DEFAULT 50000,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            created_by TEXT,
            is_active BOOLEAN DEFAULT 1,
            is_admin BOOLEAN DEFAULT 0
        )
        ''')
        
        # Add new columns to existing users table if they don't exist
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN daily_gpt_limit INTEGER DEFAULT 30000')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN daily_embedding_limit INTEGER DEFAULT 20000')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN max_gpt_limit INTEGER DEFAULT 150000')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN max_embedding_limit INTEGER DEFAULT 50000')
        except sqlite3.OperationalError:
            pass
        
        # API usage tracking table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            session_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model TEXT NOT NULL,
            operation_type TEXT NOT NULL,  -- 'chat', 'embedding', 'search'
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            estimated_cost REAL DEFAULT 0.0,
            query_type TEXT,
            knowledge_base TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Daily usage summary table for faster queries
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_usage_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            usage_date DATE NOT NULL,
            total_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            api_calls INTEGER DEFAULT 0,
            gpt_tokens INTEGER DEFAULT 0,
            embedding_tokens INTEGER DEFAULT 0,
            UNIQUE(user_id, usage_date),
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # Total (lifetime) usage summary table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS total_usage_summary (
            user_id INTEGER PRIMARY KEY,
            total_gpt_tokens INTEGER DEFAULT 0,
            total_embedding_tokens INTEGER DEFAULT 0,
            total_cost REAL DEFAULT 0.0,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        
        # System settings table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_settings (
            setting_key TEXT PRIMARY KEY,
            setting_value TEXT NOT NULL,
            setting_description TEXT,
            last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Chat history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            knowledge_base TEXT,
            message_role TEXT NOT NULL CHECK (message_role IN ('user', 'assistant')),
            message_content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            session_id TEXT,
            FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        ''')
        
        # Initialize default token calculation method
        cursor.execute('''
        INSERT OR IGNORE INTO system_settings (setting_key, setting_value, setting_description)
        VALUES ('token_calculation_method', 'rough_estimate', 'Method for calculating tokens: rough_estimate or tiktoken')
        ''')
        
        conn.commit()
        conn.close()
    
    def init_admin_user(self):
        """Initialize the default admin user"""
        admin_exists = self.get_user_by_username("admin")
        if not admin_exists:
            self.create_admin_user(
                username="admin",
                password="SagittariusA*3.14",
                employee_id="ADMIN001",
                name="System Administrator",
                department="IT",
                role="Administrator",
                email="admin@healthcare.com"
            )
    
    def hash_password(self, password: str, salt: str = None) -> tuple:
        """Hash password with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        # Use PBKDF2 for secure hashing
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        return password_hash.hex(), salt
    
    def create_user(self, employee_id: str, name: str, department: str, role: str,
                   email: str, username: str, password: str, daily_token_limit: int,
                   maximum_token_limit: int, daily_gpt_limit: int = None, 
                   daily_embedding_limit: int = None, max_gpt_limit: int = None,
                   max_embedding_limit: int = None, created_by: str = None) -> bool:
        """Create a new user account with separate model limits"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Hash password
            password_hash, salt = self.hash_password(password)
            
            # Set default values if not provided
            if daily_gpt_limit is None:
                daily_gpt_limit = int(daily_token_limit * 0.6)  # 60% for GPT
            if daily_embedding_limit is None:
                daily_embedding_limit = int(daily_token_limit * 0.4)  # 40% for embeddings
            if max_gpt_limit is None:
                max_gpt_limit = int(maximum_token_limit * 0.75)  # 75% for GPT
            if max_embedding_limit is None:
                max_embedding_limit = int(maximum_token_limit * 0.25)  # 25% for embeddings
            
            cursor.execute('''
            INSERT INTO users (employee_id, name, department, role, email, username, 
                             password_hash, salt, daily_token_limit, maximum_token_limit,
                             daily_gpt_limit, daily_embedding_limit, max_gpt_limit, max_embedding_limit, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (employee_id, name, department, role, email, username, 
                  password_hash, salt, daily_token_limit, maximum_token_limit,
                  daily_gpt_limit, daily_embedding_limit, max_gpt_limit, max_embedding_limit, created_by))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.IntegrityError as e:
            st.error(f"User creation failed: {str(e)}")
            return False
        except Exception as e:
            st.error(f"Unexpected error creating user: {str(e)}")
            return False
    
    def create_admin_user(self, username: str, password: str, employee_id: str,
                         name: str, department: str, role: str, email: str) -> bool:
        """Create admin user with elevated privileges"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash, salt = self.hash_password(password)
            
            cursor.execute('''
            INSERT INTO users (employee_id, name, department, role, email, username, 
                             password_hash, salt, daily_token_limit, maximum_token_limit, 
                             is_admin, created_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (employee_id, name, department, role, email, username, 
                  password_hash, salt, 999999, 999999, 1, "SYSTEM"))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error creating admin user: {str(e)}")
            return False
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict]:
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, employee_id, name, department, role, email, username, 
               password_hash, salt, daily_token_limit, maximum_token_limit, is_admin, is_active,
               daily_gpt_limit, daily_embedding_limit, max_gpt_limit, max_embedding_limit
        FROM users 
        WHERE username = ? AND is_active = 1
        ''', (username,))
        
        user_data = cursor.fetchone()
        conn.close()
        
        if not user_data:
            return None
        
        # Verify password
        stored_hash = user_data[7]
        salt = user_data[8]
        password_hash, _ = self.hash_password(password, salt)
        
        if password_hash == stored_hash:
            return {
                'id': user_data[0],
                'employee_id': user_data[1],
                'name': user_data[2],
                'department': user_data[3],
                'role': user_data[4],
                'email': user_data[5],
                'username': user_data[6],
                'daily_token_limit': user_data[9],
                'maximum_token_limit': user_data[10],
                'is_admin': bool(user_data[11]),
                'is_active': bool(user_data[12]),
                'daily_gpt_limit': user_data[13] if len(user_data) > 13 else user_data[9] * 0.6,
                'daily_embedding_limit': user_data[14] if len(user_data) > 14 else user_data[9] * 0.4,
                'max_gpt_limit': user_data[15] if len(user_data) > 15 else user_data[10] * 0.75,
                'max_embedding_limit': user_data[16] if len(user_data) > 16 else user_data[10] * 0.25
            }
        
        return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict]:
        """Get user information by username"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, employee_id, name, department, role, email, username, 
               daily_token_limit, maximum_token_limit, is_admin, is_active
        FROM users 
        WHERE username = ?
        ''', (username,))
        
        user_data = cursor.fetchone()
        conn.close()
        
        if user_data:
            return {
                'id': user_data[0],
                'employee_id': user_data[1],
                'name': user_data[2],
                'department': user_data[3],
                'role': user_data[4],
                'email': user_data[5],
                'username': user_data[6],
                'daily_token_limit': user_data[7],
                'maximum_token_limit': user_data[8],
                'is_admin': bool(user_data[9]),
                'is_active': bool(user_data[10])
            }
        return None
    
    def get_all_users(self) -> List[Dict]:
        """Get all users for admin dashboard"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, employee_id, name, department, role, email, username, 
               daily_token_limit, maximum_token_limit, created_at, is_active,
               daily_gpt_limit, daily_embedding_limit, max_gpt_limit, max_embedding_limit
        FROM users 
        ORDER BY created_at DESC
        ''')
        
        users = []
        for row in cursor.fetchall():
            users.append({
                'id': row[0],
                'employee_id': row[1],
                'name': row[2],
                'department': row[3],
                'role': row[4],
                'email': row[5],
                'username': row[6],
                'daily_token_limit': row[7],
                'maximum_token_limit': row[8],
                'created_at': row[9],
                'is_active': bool(row[10]),
                'daily_gpt_limit': row[11] if len(row) > 11 and row[11] is not None else row[7] * 0.6,
                'daily_embedding_limit': row[12] if len(row) > 12 and row[12] is not None else row[7] * 0.4,
                'max_gpt_limit': row[13] if len(row) > 13 and row[13] is not None else row[8] * 0.75,
                'max_embedding_limit': row[14] if len(row) > 14 and row[14] is not None else row[8] * 0.25
            })
        
        conn.close()
        return users
    
    def delete_user(self, user_id: int, deleted_by: str = None) -> bool:
        """Delete a user account (soft delete by setting is_active to False)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists and is not admin
            cursor.execute('SELECT is_admin, username FROM users WHERE id = ?', (user_id,))
            user_info = cursor.fetchone()
            
            if not user_info:
                return False, "User not found"
            
            if user_info[0]:  # is_admin is True
                return False, "Cannot delete admin users"
            
            # Soft delete by setting is_active to False
            cursor.execute('''
            UPDATE users 
            SET is_active = 0 
            WHERE id = ?
            ''', (user_id,))
            
            conn.commit()
            conn.close()
            return True, f"User '{user_info[1]}' has been deactivated successfully"
            
        except Exception as e:
            return False, f"Error deleting user: {str(e)}"
    
    def permanently_delete_user(self, user_id: int, deleted_by: str = None) -> bool:
        """Permanently delete a user account and all associated data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists and is not admin
            cursor.execute('SELECT is_admin, username FROM users WHERE id = ?', (user_id,))
            user_info = cursor.fetchone()
            
            if not user_info:
                return False, "User not found"
            
            if user_info[0]:  # is_admin is True
                return False, "Cannot delete admin users"
            
            # Delete user's API usage history
            cursor.execute('DELETE FROM api_usage WHERE user_id = ?', (user_id,))
            cursor.execute('DELETE FROM daily_usage_summary WHERE user_id = ?', (user_id,))
            
            # Delete the user
            cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
            
            conn.commit()
            conn.close()
            return True, f"User '{user_info[1]}' and all associated data has been permanently deleted"
            
        except Exception as e:
            return False, f"Error permanently deleting user: {str(e)}"
    
    def update_user(self, user_id: int, **kwargs) -> tuple:
        """Update user account fields"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user exists and get current info
            cursor.execute('SELECT username, is_admin FROM users WHERE id = ?', (user_id,))
            user_info = cursor.fetchone()
            
            if not user_info:
                return False, "User not found"
            
            # Build update query dynamically
            valid_fields = ['employee_id', 'name', 'department', 'role', 'email', 
                          'daily_token_limit', 'maximum_token_limit', 'daily_gpt_limit',
                          'daily_embedding_limit', 'max_gpt_limit', 'max_embedding_limit']
            
            updates = []
            values = []
            
            for field, value in kwargs.items():
                if field in valid_fields and value is not None:
                    updates.append(f"{field} = ?")
                    values.append(value)
            
            if not updates:
                return False, "No valid fields to update"
            
            values.append(user_id)
            query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
            
            cursor.execute(query, values)
            conn.commit()
            conn.close()
            
            return True, f"User '{user_info[0]}' updated successfully"
            
        except sqlite3.IntegrityError as e:
            return False, f"Update failed: {str(e)}"
        except Exception as e:
            return False, f"Error updating user: {str(e)}"
    
    def save_chat_message(self, user_id: int, message_role: str, message_content: str, 
                         knowledge_base: str = None, session_id: str = None) -> bool:
        """Save a chat message to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT INTO chat_history (user_id, knowledge_base, message_role, message_content, session_id)
            VALUES (?, ?, ?, ?, ?)
            ''', (user_id, knowledge_base, message_role, message_content, session_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error saving chat message: {str(e)}")
            return False
    
    def load_chat_history(self, user_id: int, knowledge_base: str = None, limit: int = 50) -> List[Dict]:
        """Load chat history for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if knowledge_base:
                cursor.execute('''
                SELECT message_role, message_content, timestamp
                FROM chat_history
                WHERE user_id = ? AND knowledge_base = ?
                ORDER BY timestamp ASC
                LIMIT ?
                ''', (user_id, knowledge_base, limit))
            else:
                cursor.execute('''
                SELECT message_role, message_content, timestamp
                FROM chat_history
                WHERE user_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                ''', (user_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                messages.append({
                    'role': row[0],
                    'content': row[1],
                    'timestamp': row[2]
                })
            
            conn.close()
            return messages
            
        except Exception as e:
            print(f"Error loading chat history: {str(e)}")
            return []
    
    def clear_chat_history(self, user_id: int, knowledge_base: str = None) -> bool:
        """Clear chat history for a user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if knowledge_base:
                cursor.execute('''
                DELETE FROM chat_history
                WHERE user_id = ? AND knowledge_base = ?
                ''', (user_id, knowledge_base))
            else:
                cursor.execute('''
                DELETE FROM chat_history
                WHERE user_id = ?
                ''', (user_id,))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error clearing chat history: {str(e)}")
            return False


class APIUsageTracker:
    def __init__(self, db_path: str = "data/users.db"):
        self.db_path = db_path
        
        # Azure OpenAI pricing (as of 2025)
        self.pricing = {
            'gpt-4.1': {  # Azure gpt-4.1 model
                'input': 0.002,   # $0.002 per 1K input tokens
                'output': 0.008   # $0.008 per 1K output tokens
            },
            'gpt-4.1-2025-04-14': {  # OpenAI gpt-4.1 model
                'input': 0.002,   # $0.002 per 1K input tokens
                'output': 0.008   # $0.008 per 1K output tokens
            },
            'text-embedding-3-large': {
                'input': 0.00013,  # $0.00013 per 1K tokens
                'output': 0.0      # No output cost for embeddings
            }
        }
        
        # Initialize tiktoken encoders cache (lazy loading, per model)
        self._tiktoken_encoders = {}
    
    def get_system_setting(self, setting_key: str) -> str:
        """Get system setting value"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT setting_value FROM system_settings WHERE setting_key = ?', (setting_key,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else None
    
    def update_system_setting(self, setting_key: str, setting_value: str) -> bool:
        """Update system setting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            INSERT OR REPLACE INTO system_settings (setting_key, setting_value, last_updated)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (setting_key, setting_value))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating system setting: {e}")
            return False
    
    def _get_tiktoken_encoder(self, model: str = "gpt-4"):
        """Get tiktoken encoder (lazy loading with per-model caching)"""
        if model not in self._tiktoken_encoders:
            try:
                import tiktoken
                self._tiktoken_encoders[model] = tiktoken.encoding_for_model(model)
            except ImportError:
                print("tiktoken not available, falling back to rough estimation")
                return None
            except Exception as e:
                print(f"Error initializing tiktoken for model {model}: {e}")
                return None
        return self._tiktoken_encoders[model]
    
    def estimate_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Estimate token count using selected method"""
        if not text:
            return 0
            
        # Get system setting for token calculation method
        method = self.get_system_setting('token_calculation_method')
        
        if method == 'tiktoken':
            encoder = self._get_tiktoken_encoder(model)
            if encoder:
                try:
                    return len(encoder.encode(text))
                except Exception as e:
                    print(f"Error with tiktoken encoding: {e}, falling back to rough estimate")
        
        # Fallback to rough estimation
        word_count = len(text.split())
        return int(word_count * 1.33)
    
    def log_api_usage(self, user_id: int, model: str, operation_type: str,
                     input_text: str = "", output_text: str = "",
                     input_tokens: int = None, output_tokens: int = None,
                     query_type: str = None, knowledge_base: str = None) -> float:
        """Log API usage and calculate cost"""
        
        # Calculate tokens if not provided
        if input_tokens is None and input_text:
            input_tokens = self.estimate_tokens(input_text, model)
        if output_tokens is None and output_text:
            output_tokens = self.estimate_tokens(output_text, model)
        
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0
        total_tokens = input_tokens + output_tokens
        
        # Use the model name directly for pricing lookup
        model_key = model
        if model_key in self.pricing:
            cost = (input_tokens / 1000 * self.pricing[model_key]['input'] + 
                   output_tokens / 1000 * self.pricing[model_key]['output'])
        else:
            cost = 0.0
        
        # Log to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO api_usage (user_id, model, operation_type, input_tokens, 
                              output_tokens, total_tokens, estimated_cost, query_type, knowledge_base)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, model, operation_type, input_tokens, output_tokens, 
              total_tokens, cost, query_type, knowledge_base))
        
        # Update daily summary with simpler approach
        today = date.today().isoformat()
        
        # First, check if record exists
        cursor.execute('''
        SELECT total_tokens, total_cost, api_calls, gpt_tokens, embedding_tokens
        FROM daily_usage_summary WHERE user_id = ? AND usage_date = ?
        ''', (user_id, today))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            new_total_tokens = existing[0] + total_tokens
            new_total_cost = existing[1] + cost
            new_api_calls = existing[2] + 1
            new_gpt_tokens = existing[3] + (input_tokens + output_tokens if model_key in ['gpt-4.1', 'gpt-4.1-2025-04-14'] else 0)
            new_embedding_tokens = existing[4] + (input_tokens + output_tokens if model_key == 'text-embedding-3-large' else 0)
            
            cursor.execute('''
            UPDATE daily_usage_summary 
            SET total_tokens = ?, total_cost = ?, api_calls = ?, gpt_tokens = ?, embedding_tokens = ?
            WHERE user_id = ? AND usage_date = ?
            ''', (new_total_tokens, new_total_cost, new_api_calls, new_gpt_tokens, new_embedding_tokens, user_id, today))
        else:
            # Insert new record
            gpt_tokens = input_tokens + output_tokens if model_key in ['gpt-4.1', 'gpt-4.1-2025-04-14'] else 0
            embedding_tokens = input_tokens + output_tokens if model_key == 'text-embedding-3-large' else 0
            
            cursor.execute('''
            INSERT INTO daily_usage_summary (user_id, usage_date, total_tokens, total_cost, api_calls, gpt_tokens, embedding_tokens)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, today, total_tokens, cost, 1, gpt_tokens, embedding_tokens))
        
        # Update total (lifetime) usage summary
        cursor.execute('''
        SELECT total_gpt_tokens, total_embedding_tokens, total_cost
        FROM total_usage_summary WHERE user_id = ?
        ''', (user_id,))
        
        total_existing = cursor.fetchone()
        
        gpt_tokens_to_add = input_tokens + output_tokens if model_key in ['gpt-4.1', 'gpt-4.1-2025-04-14'] else 0
        embedding_tokens_to_add = input_tokens + output_tokens if model_key == 'text-embedding-3-large' else 0
        
        if total_existing:
            # Update existing total record
            new_total_gpt = total_existing[0] + gpt_tokens_to_add
            new_total_embedding = total_existing[1] + embedding_tokens_to_add
            new_total_cost = total_existing[2] + cost
            
            cursor.execute('''
            UPDATE total_usage_summary 
            SET total_gpt_tokens = ?, total_embedding_tokens = ?, total_cost = ?, last_updated = CURRENT_TIMESTAMP
            WHERE user_id = ?
            ''', (new_total_gpt, new_total_embedding, new_total_cost, user_id))
        else:
            # Insert new total record
            cursor.execute('''
            INSERT INTO total_usage_summary (user_id, total_gpt_tokens, total_embedding_tokens, total_cost)
            VALUES (?, ?, ?, ?)
            ''', (user_id, gpt_tokens_to_add, embedding_tokens_to_add, cost))
        
        conn.commit()
        conn.close()
        
        return cost
    
    def get_user_daily_usage(self, user_id: int, date_str: str = None) -> Dict:
        """Get daily usage statistics for a user"""
        if date_str is None:
            date_str = date.today().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT total_tokens, total_cost, api_calls, gpt_tokens, embedding_tokens
        FROM daily_usage_summary
        WHERE user_id = ? AND usage_date = ?
        ''', (user_id, date_str))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_tokens': result[0],
                'total_cost': result[1],
                'api_calls': result[2],
                'gpt_tokens': result[3],
                'embedding_tokens': result[4],
                'date': date_str
            }
        else:
            return {
                'total_tokens': 0,
                'total_cost': 0.0,
                'api_calls': 0,
                'gpt_tokens': 0,
                'embedding_tokens': 0,
                'date': date_str
            }
    
    def get_user_total_usage(self, user_id: int) -> Dict:
        """Get total (lifetime) usage statistics for a user"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT total_gpt_tokens, total_embedding_tokens, total_cost
        FROM total_usage_summary
        WHERE user_id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_gpt_tokens': result[0],
                'total_embedding_tokens': result[1],
                'total_cost': result[2]
            }
        else:
            return {
                'total_gpt_tokens': 0,
                'total_embedding_tokens': 0,
                'total_cost': 0.0
            }
    
    def get_all_users_usage(self, date_str: str = None) -> List[Dict]:
        """Get usage statistics for all users (admin view)"""
        if date_str is None:
            date_str = date.today().isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT u.id, u.name, u.department, u.role, 
               COALESCE(dus.total_tokens, 0) as total_tokens,
               COALESCE(dus.total_cost, 0) as total_cost,
               COALESCE(dus.api_calls, 0) as api_calls,
               u.daily_token_limit, u.maximum_token_limit
        FROM users u
        LEFT JOIN daily_usage_summary dus ON u.id = dus.user_id AND dus.usage_date = ?
        WHERE u.is_active = 1
        ORDER BY dus.total_cost DESC NULLS LAST
        ''', (date_str,))
        
        usage_data = []
        for row in cursor.fetchall():
            usage_data.append({
                'user_id': row[0],
                'name': row[1],
                'department': row[2],
                'role': row[3],
                'total_tokens': row[4],
                'total_cost': row[5],
                'api_calls': row[6],
                'daily_token_limit': row[7],
                'maximum_token_limit': row[8],
                'token_usage_percent': (row[4] / row[7] * 100) if row[7] > 0 else 0
            })
        
        conn.close()
        return usage_data
    
    def check_user_limits(self, user_id: int, daily_gpt_limit: int, daily_embedding_limit: int, 
                          max_gpt_limit: int, max_embedding_limit: int) -> Dict:
        """Check if user has exceeded their limits for both models"""
        daily_usage = self.get_user_daily_usage(user_id)
        total_usage = self.get_user_total_usage(user_id)
        
        gpt_daily_used = daily_usage.get('gpt_tokens', 0)
        embedding_daily_used = daily_usage.get('embedding_tokens', 0)
        gpt_total_used = total_usage.get('total_gpt_tokens', 0)
        embedding_total_used = total_usage.get('total_embedding_tokens', 0)
        
        return {
            'gpt_within_daily_limit': gpt_daily_used < daily_gpt_limit,
            'gpt_within_max_limit': gpt_total_used < max_gpt_limit,
            'embedding_within_daily_limit': embedding_daily_used < daily_embedding_limit,
            'embedding_within_max_limit': embedding_total_used < max_embedding_limit,
            'gpt_daily_remaining': max(0, daily_gpt_limit - gpt_daily_used),
            'gpt_total_remaining': max(0, max_gpt_limit - gpt_total_used),
            'embedding_daily_remaining': max(0, daily_embedding_limit - embedding_daily_used),
            'embedding_total_remaining': max(0, max_embedding_limit - embedding_total_used),
            'daily_usage': daily_usage,
            'total_usage': total_usage
        }