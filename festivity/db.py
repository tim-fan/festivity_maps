"""
Database operations for festivity workspace.
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime


def create_database(db_path: Path) -> None:
    """
    Create SQLite database with required schema.
    
    Args:
        db_path: Path to database file
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create gps_data table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gps_data (
            filename TEXT PRIMARY KEY,
            is_left BOOLEAN NOT NULL,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            timestamp TIMESTAMP,
            heading REAL,
            offset_lat REAL,
            offset_lon REAL,
            address TEXT,
            street_number TEXT,
            street_name TEXT,
            address_lat REAL,
            address_lon REAL
        )
    """)
    
    # Create festivity_scores table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS festivity_scores (
            filename TEXT PRIMARY KEY,
            mean_probability REAL NOT NULL
        )
    """)
    
    # Create indices for common queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gps_is_left ON gps_data(is_left)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_gps_timestamp ON gps_data(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scores_probability ON festivity_scores(mean_probability)")
    
    conn.commit()
    conn.close()


def get_db_connection(workspace_path: Path) -> sqlite3.Connection:
    """
    Get SQLite database connection with foreign keys enabled.
    
    Args:
        workspace_path: Path to workspace directory
        
    Returns:
        SQLite connection object
    """
    db_path = workspace_path / 'database.db'
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    # Return rows as dictionaries
    conn.row_factory = sqlite3.Row
    return conn


def get_unprocessed_images(conn: sqlite3.Connection, target_table: str) -> List[Dict[str, Any]]:
    """
    Query images that don't have data in the specified target table.
    
    Args:
        conn: Database connection
        target_table: Target table name ('festivity_scores' only, gps_data is the base table)
        
    Returns:
        List of image records (as dictionaries)
    """
    if target_table == 'festivity_scores':
        query = """
            SELECT g.filename, g.is_left
            FROM gps_data g
            LEFT JOIN festivity_scores s ON g.filename = s.filename
            WHERE s.filename IS NULL
            ORDER BY g.filename
        """
    else:
        # For gps_data, we'll handle this differently in the extract_gps command
        # It will scan the images directory
        raise ValueError(f"Invalid target_table: {target_table}")
    
    cursor = conn.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def insert_gps_data(conn: sqlite3.Connection, gps_records: List[Dict[str, Any]]) -> int:
    """
    Bulk insert or update GPS data.
    
    Args:
        conn: Database connection
        gps_records: List of dicts with GPS data fields (including is_left)
        
    Returns:
        Number of records inserted/updated
    """
    cursor = conn.cursor()
    
    # Use INSERT OR REPLACE to update existing records
    cursor.executemany(
        """
        INSERT OR REPLACE INTO gps_data (
            filename, is_left, lat, lon, timestamp, heading,
            offset_lat, offset_lon, address, street_number, street_name,
            address_lat, address_lon
        ) VALUES (
            :filename, :is_left, :lat, :lon, :timestamp, :heading,
            :offset_lat, :offset_lon, :address, :street_number, :street_name,
            :address_lat, :address_lon
        )
        """,
        gps_records
    )
    
    conn.commit()
    return cursor.rowcount


def insert_festivity_scores(conn: sqlite3.Connection, score_records: List[Dict[str, Any]]) -> int:
    """
    Bulk insert or update festivity scores.
    
    Args:
        conn: Database connection
        score_records: List of dicts with keys: filename, mean_probability
        
    Returns:
        Number of records inserted/updated
    """
    cursor = conn.cursor()
    
    cursor.executemany(
        """
        INSERT OR REPLACE INTO festivity_scores (
            filename, mean_probability
        ) VALUES (
            :filename, :mean_probability
        )
        """,
        score_records
    )
    
    conn.commit()
    return cursor.rowcount


def get_all_data_for_visualization(
    conn: sqlite3.Connection,
    min_score: Optional[float] = None,
    max_score: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Join all tables to get complete data for map visualization.
    
    Args:
        conn: Database connection
        min_score: Minimum festivity score filter (optional)
        max_score: Maximum festivity score filter (optional)
        
    Returns:
        List of complete records with all fields
    """
    query = """
        SELECT 
            g.filename,
            g.is_left,
            g.lat,
            g.lon,
            g.timestamp,
            g.heading,
            g.offset_lat,
            g.offset_lon,
            g.address,
            g.street_number,
            g.street_name,
            g.address_lat,
            g.address_lon,
            s.mean_probability
        FROM gps_data g
        INNER JOIN festivity_scores s ON g.filename = s.filename
        WHERE 1=1
    """
    
    params = []
    if min_score is not None:
        query += " AND s.mean_probability >= ?"
        params.append(min_score)
    if max_score is not None:
        query += " AND s.mean_probability <= ?"
        params.append(max_score)
    
    query += " ORDER BY g.timestamp"
    
    cursor = conn.execute(query, params)
    return [dict(row) for row in cursor.fetchall()]


def get_images_with_gps(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """
    Get all images that have GPS data, with is_left flag.
    
    Args:
        conn: Database connection
        
    Returns:
        List of records with image and GPS data
    """
    query = """
        SELECT 
            g.filename,
            g.is_left,
            g.lat,
            g.lon,
            g.timestamp,
            g.heading
        FROM gps_data g
        ORDER BY g.is_left, g.timestamp
    """
    
    cursor = conn.execute(query)
    return [dict(row) for row in cursor.fetchall()]


def get_filenames_in_gps_data(conn: sqlite3.Connection) -> set:
    """
    Get set of filenames that already have GPS data.
    
    Args:
        conn: Database connection
        
    Returns:
        Set of filenames
    """
    cursor = conn.execute("SELECT filename FROM gps_data")
    return {row[0] for row in cursor.fetchall()}

