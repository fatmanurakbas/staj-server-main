"""
Database configuration and utilities for PostgreSQL
"""
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
import os
from typing import Dict, Any, List, Optional
import bcrypt

class DatabaseManager:
    def __init__(self):
        self.host = os.environ.get('DB_HOST', 'localhost')
        self.port = os.environ.get('DB_PORT', '5432')
        self.database = os.environ.get('DB_NAME', 'liver_assessment')
        self.user = os.environ.get('DB_USER', 'postgres')
        self.password = os.environ.get('DB_PASSWORD', 'postgres')
        
        # Create database if it doesn't exist
        self.create_database_if_not_exists()
        # Initialize tables
        self.init_tables()
    
    def create_database_if_not_exists(self):
        """Create the database if it doesn't exist"""
        try:
            # Connect to default postgres database to create our database
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database='postgres',
                user=self.user,
                password=self.password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            
            # Check if database exists
            cursor.execute("SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (self.database,))
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(f'CREATE DATABASE "{self.database}"')
                print(f"✅ Database '{self.database}' created successfully")
            else:
                print(f"✅ Database '{self.database}' already exists")
            
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"❌ Error creating database: {e}")
    
    @contextmanager
    def get_connection(self):
        """Get database connection with context manager"""
        conn = None
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                cursor_factory=psycopg2.extras.RealDictCursor
            )
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def init_tables(self):
        """Initialize database tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create users table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        name_surname VARCHAR(255) NOT NULL,
                        first_name VARCHAR(255) NOT NULL,
                        last_name VARCHAR(255) NOT NULL,
                        email VARCHAR(255) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        medical_field VARCHAR(255) NOT NULL,
                        organization VARCHAR(255) NOT NULL,
                        diploma_number VARCHAR(255) NOT NULL,
                        years_experience INTEGER DEFAULT 0,
                        phone VARCHAR(50) DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        is_active BOOLEAN DEFAULT TRUE
                    )
                """)
                
                # Add missing columns if they don't exist (for existing databases)
                self._add_missing_columns(cursor)
                
                # Create user_sessions table for storing assessment data
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
                        session_data JSONB,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create index on email for faster lookups
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
                """)
                
                conn.commit()
                print("✅ Database tables initialized successfully")
                
        except Exception as e:
            print(f"❌ Error initializing tables: {e}")
    
    def _add_missing_columns(self, cursor):
        """Add missing columns to existing users table"""
        try:
            # Check if doctor_title column exists (most recent addition)
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name='users' AND column_name='doctor_title'
            """)
            
            if not cursor.fetchone():
                print("Adding doctor_title column to users table...")
                cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS doctor_title VARCHAR(100) DEFAULT 'Dr.'")
                print("✅ doctor_title column added successfully")
            
            # Check if first_name column exists (older migration)
            cursor.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name='users' AND column_name='first_name'
            """)
            
            if not cursor.fetchone():
                print("Adding missing columns to users table...")
                
                # Add first_name and last_name columns
                cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS first_name VARCHAR(255) DEFAULT ''")
                cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_name VARCHAR(255) DEFAULT ''")
                cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS years_experience INTEGER DEFAULT 0")
                cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(50) DEFAULT ''")
                
                # Update existing records to populate first_name and last_name from name_surname
                cursor.execute("""
                    UPDATE users 
                    SET first_name = SPLIT_PART(name_surname, ' ', 1),
                        last_name = CASE 
                            WHEN ARRAY_LENGTH(STRING_TO_ARRAY(name_surname, ' '), 1) > 1 
                            THEN SUBSTRING(name_surname FROM POSITION(' ' IN name_surname) + 1)
                            ELSE ''
                        END
                    WHERE (first_name IS NULL OR first_name = '') AND name_surname IS NOT NULL
                """)
                
                # Now make first_name and last_name NOT NULL
                cursor.execute("ALTER TABLE users ALTER COLUMN first_name SET NOT NULL")
                cursor.execute("ALTER TABLE users ALTER COLUMN last_name SET NOT NULL")
                
                print("✅ Missing columns added successfully")
                
        except Exception as e:
            print(f"❌ Error adding missing columns: {e}")
    
    def create_user(self, email: str, password: str, first_name: str, last_name: str,
                   medical_field: str, organization: str, diploma_number: str, 
                   years_experience: int = 0, phone: str = "", doctor_title: str = "Dr.") -> Optional[int]:
        """Create a new user"""
        try:
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Combine first and last name for compatibility
            name_surname = f"{first_name} {last_name}"
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO users (name_surname, email, password_hash, medical_field, organization, diploma_number, first_name, last_name, years_experience, phone, doctor_title)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (name_surname, email, password_hash, medical_field, organization, diploma_number, first_name, last_name, years_experience, phone, doctor_title))
                
                user_id = cursor.fetchone()['id']
                conn.commit()
                return user_id
                
        except psycopg2.IntegrityError:
            # Email already exists
            return None
        except Exception as e:
            print(f"❌ Error creating user: {e}")
            return None
    
    def verify_user_credentials(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials and return user data"""
        return self.verify_user(email, password)
    
    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email address"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, first_name, last_name, email, medical_field, organization, diploma_number, doctor_title
                    FROM users 
                    WHERE email = %s AND is_active = TRUE
                """, (email,))
                
                user = cursor.fetchone()
                return dict(user) if user else None
        except Exception as e:
            print(f"❌ Error getting user by email: {e}")
            return None
    
    def verify_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials and return user data"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, first_name, last_name, email, password_hash, medical_field, organization, diploma_number, doctor_title
                    FROM users 
                    WHERE email = %s AND is_active = TRUE
                """, (email,))
                
                user = cursor.fetchone()
                if user and bcrypt.checkpw(password.encode('utf-8'), user['password_hash'].encode('utf-8')):
                    # Remove password_hash from returned data
                    user_data = dict(user)
                    del user_data['password_hash']
                    return user_data
                return None
                
        except Exception as e:
            print(f"❌ Error verifying user: {e}")
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, first_name, last_name, email, medical_field, organization, diploma_number, doctor_title
                    FROM users 
                    WHERE id = %s AND is_active = TRUE
                """, (user_id,))
                
                user = cursor.fetchone()
                return dict(user) if user else None
                
        except Exception as e:
            print(f"❌ Error getting user: {e}")
            return None
    
    def email_exists(self, email: str) -> bool:
        """Check if email already exists"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM users WHERE email = %s", (email,))
                return cursor.fetchone() is not None
        except Exception as e:
            print(f"❌ Error checking email: {e}")
            return False
    
    def get_doctor_titles(self) -> List[str]:
        """Get available doctor titles"""
        return [
            "Dr.",
            "Prof. Dr.",
            "Doç. Dr.",
            "Öğr. Gör. Dr.",
            "Uzm. Dr.",
            "Op. Dr.",
            "Dt.",
            "Vet.",
            "Ebe",
            "Hemşire"
        ]

# Global database instance
db = DatabaseManager()
