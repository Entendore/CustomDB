import numpy as np
import pandas as pd
import numexpr as ne
import operator
import re
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pickle
from dataclasses import dataclass
import struct

# -------------------------
# Constants and Types
# -------------------------
OPS = {
    '=': operator.eq,
    '!=': operator.ne,
    '>': operator.gt,
    '<': operator.lt,
    '>=': operator.ge,
    '<=': operator.le
}

DTYPE_MAP = {
    'int': np.int64,
    'integer': np.int64,
    'float': np.float64,
    'double': np.float64,
    'str': np.dtype('U256'),
    'string': np.dtype('U256'),
    'bool': np.bool_,
    'boolean': np.bool_
}

@dataclass
class ColumnMetadata:
    name: str
    dtype: type
    min_val: Any = None
    max_val: Any = None
    distinct_count: int = 0

# -------------------------
# Column Vector
# -------------------------
class ColumnVector:
    def __init__(self, name: str, dtype: type, capacity: int = 1024):
        self.name = name
        self.dtype = dtype
        self.capacity = capacity
        self.size = 0
        self.data = np.empty(capacity, dtype=dtype)
        self.null_mask = np.zeros(capacity, dtype=bool)
        self.metadata = ColumnMetadata(name, dtype)
    
    def append(self, value):
        if self.size >= self.capacity:
            self._resize(self.capacity * 2)
        
        if value is None:
            self.null_mask[self.size] = True
            # For numeric types, use NaN, for others use appropriate null value
            if np.issubdtype(self.dtype, np.number):
                self.data[self.size] = np.nan
            else:
                self.data[self.size] = self.dtype().item() if hasattr(self.dtype(), 'item') else None
        else:
            self.data[self.size] = value
            # Update metadata
            if self.metadata.min_val is None or value < self.metadata.min_val:
                self.metadata.min_val = value
            if self.metadata.max_val is None or value > self.metadata.max_val:
                self.metadata.max_val = value
        
        self.size += 1
    
    def _resize(self, new_capacity):
        new_data = np.empty(new_capacity, dtype=self.dtype)
        new_null_mask = np.zeros(new_capacity, dtype=bool)
        
        new_data[:self.size] = self.data[:self.size]
        new_null_mask[:self.size] = self.null_mask[:self.size]
        
        self.data = new_data
        self.null_mask = new_null_mask
        self.capacity = new_capacity
    
    def filter(self, mask: np.ndarray) -> 'ColumnVector':
        """Return a new column with filtered data"""
        # Only use the actual data, not the entire capacity
        actual_data = self.data[:self.size]
        actual_null_mask = self.null_mask[:self.size]
        
        result_size = np.sum(mask)
        result = ColumnVector(self.name, self.dtype, result_size)
        result.data[:result_size] = actual_data[mask]
        result.null_mask[:result_size] = actual_null_mask[mask]
        result.size = result_size
        return result
    
    def to_arrow(self) -> pa.Array:
        """Convert to PyArrow array for efficient I/O"""
        if np.issubdtype(self.dtype, np.number):
            return pa.array(self.data[:self.size], mask=self.null_mask[:self.size])
        else:
            return pa.array(self.data[:self.size].astype(str), mask=self.null_mask[:self.size])
    
    @classmethod
    def from_arrow(cls, arrow_array: pa.Array, name: str):
        """Create from PyArrow array"""
        dtype = arrow_array.type.to_pandas_dtype()
        col = cls(name, dtype, len(arrow_array))
        col.data[:len(arrow_array)] = arrow_array.to_numpy()
        col.null_mask[:len(arrow_array)] = arrow_array.is_null().to_numpy()
        col.size = len(arrow_array)
        return col

# -------------------------
# Data Chunk (Vectorized Processing Unit)
# -------------------------
class DataChunk:
    def __init__(self, columns: Dict[str, ColumnVector]):
        self.columns = columns
        self.size = next(iter(columns.values())).size if columns else 0
    
    def filter(self, mask: np.ndarray) -> 'DataChunk':
        """Filter the chunk based on a boolean mask"""
        # Make sure mask size matches actual data size
        if len(mask) != self.size:
            raise ValueError(f"Mask size {len(mask)} doesn't match chunk size {self.size}")
        
        filtered_columns = {name: col.filter(mask) for name, col in self.columns.items()}
        return DataChunk(filtered_columns)
    
    def project(self, expressions: Dict[str, str]) -> 'DataChunk':
        """Apply projections/expressions to create new columns"""
        # Create a dictionary of all available data for expression evaluation
        data_dict = {name: col.data[:col.size] for name, col in self.columns.items()}
        
        new_columns = {}
        for alias, expr in expressions.items():
            # Evaluate expression using numexpr for performance
            try:
                result = ne.evaluate(expr, data_dict)
                new_col = ColumnVector(alias, result.dtype, len(result))
                new_col.data[:len(result)] = result
                new_col.size = len(result)
                new_columns[alias] = new_col
            except Exception as e:
                raise ValueError(f"Error evaluating expression '{expr}': {e}")
        
        return DataChunk({**self.columns, **new_columns})
    
    def materialize(self) -> 'DataChunk':
        """Create a materialized copy of the chunk"""
        new_columns = {}
        for name, col in self.columns.items():
            new_col = ColumnVector(name, col.dtype, col.size)
            new_col.data[:col.size] = col.data[:col.size]
            new_col.null_mask[:col.size] = col.null_mask[:col.size]
            new_col.size = col.size
            new_columns[name] = new_col
        return DataChunk(new_columns)

# -------------------------
# Table (Collection of Data Chunks)
# -------------------------
class Table:
    def __init__(self, name: str, schema: Dict[str, type]):
        self.name = name
        self.schema = schema
        self.chunks: List[DataChunk] = []
        self.current_chunk = self._create_empty_chunk()
        self.row_count = 0
        self.indices: Dict[str, Any] = {}  # Column indices for faster lookups
    
    def _create_empty_chunk(self) -> DataChunk:
        columns = {name: ColumnVector(name, dtype) for name, dtype in self.schema.items()}
        return DataChunk(columns)
    
    def insert(self, rows: List[Dict[str, Any]]):
        """Batch insert for better performance"""
        for row in rows:
            for col_name, value in row.items():
                if col_name in self.current_chunk.columns:
                    self.current_chunk.columns[col_name].append(value)
            
            self.row_count += 1
            self.current_chunk.size += 1
            
            # If current chunk is full, add it to chunks and create a new one
            if self.current_chunk.size >= 1024:  # Chunk size
                self.chunks.append(self.current_chunk)
                self.current_chunk = self._create_empty_chunk()
    
    def finalize(self):
        """Finalize the table after inserts"""
        if self.current_chunk.size > 0:
            self.chunks.append(self.current_chunk)
            self.current_chunk = self._create_empty_chunk()
    
    def create_index(self, column_name: str):
        """Create an index on a column for faster lookups"""
        if column_name not in self.schema:
            raise ValueError(f"Column {column_name} does not exist")
        
        # For simplicity, we'll create a dictionary index
        index = {}
        for chunk in self.chunks:
            col = chunk.columns[column_name]
            for i in range(col.size):
                value = col.data[i]
                if value not in index:
                    index[value] = []
                index[value].append((chunk, i))
        
        self.indices[column_name] = index
    
    def scan(self) -> 'TableScanner':
        """Create a scanner for this table"""
        return TableScanner(self)
    
    def aggregate(self, group_by: List[str], aggregates: Dict[str, Tuple[str, str]]):
        """Perform aggregation operations"""
        # Implementation for GROUP BY and aggregate functions
        result_schema = {col: self.schema[col] for col in group_by}
        for alias, (func, col) in aggregates.items():
            if func in ['SUM', 'AVG', 'MIN', 'MAX']:
                result_schema[alias] = self.schema[col]
            else:  # COUNT
                result_schema[alias] = np.int64
        
        result_table = Table(f"{self.name}_agg", result_schema)
        
        # Group by logic
        groups = {}
        for chunk in self.chunks:
            # Create group key arrays
            group_keys = []
            for col in group_by:
                group_keys.append(chunk.columns[col].data[:chunk.size])
            
            # For each row, create a group key tuple
            for i in range(chunk.size):
                key = tuple(key[i] for key in group_keys)
                if key not in groups:
                    groups[key] = {agg_col: [] for agg_col in aggregates.keys()}
                
                # Collect values for aggregates
                for alias, (func, col) in aggregates.items():
                    if col in chunk.columns:
                        groups[key][alias].append(chunk.columns[col].data[i])
        
        # Compute aggregates and insert into result table
        for key, agg_values in groups.items():
            row = {group_by[i]: key[i] for i in range(len(group_by))}
            for alias, values in agg_values.items():
                func, col = aggregates[alias]
                if func == 'SUM':
                    row[alias] = np.sum(values)
                elif func == 'AVG':
                    row[alias] = np.mean(values)
                elif func == 'MIN':
                    row[alias] = np.min(values)
                elif func == 'MAX':
                    row[alias] = np.max(values)
                elif func == 'COUNT':
                    row[alias] = len(values)
            
            result_table.insert([row])
        
        result_table.finalize()
        return result_table
    
    def join(self, other: 'Table', left_on: List[str], right_on: List[str], how: str = 'inner'):
        """Join this table with another table"""
        # Validate join keys
        if len(left_on) != len(right_on):
            raise ValueError("Number of join keys must match")
        
        # Create result schema
        result_schema = {}
        for col, dtype in self.schema.items():
            result_schema[col] = dtype
        for col, dtype in other.schema.items():
            result_schema[f"{other.name}_{col}"] = dtype
        
        result_table = Table(f"{self.name}_{how}_join_{other.name}", result_schema)
        
        # Build hash table from right table for hash join
        hash_table = {}
        for chunk in other.chunks:
            # Create join key arrays
            join_keys = []
            for col in right_on:
                join_keys.append(chunk.columns[col].data[:chunk.size])
            
            # Add to hash table
            for i in range(chunk.size):
                key = tuple(key[i] for key in join_keys)
                if key not in hash_table:
                    hash_table[key] = []
                hash_table[key].append({col: chunk.columns[col].data[i] for col in other.schema.keys()})
        
        # Probe hash table with left table
        for chunk in self.chunks:
            # Create join key arrays
            join_keys = []
            for col in left_on:
                join_keys.append(chunk.columns[col].data[:chunk.size])
            
            # For each row, probe the hash table
            for i in range(chunk.size):
                key = tuple(key[i] for key in join_keys)
                if key in hash_table:
                    for right_row in hash_table[key]:
                        left_row = {col: chunk.columns[col].data[i] for col in self.schema.keys()}
                        merged_row = {**left_row, **{f"{other.name}_{col}": val for col, val in right_row.items()}}
                        result_table.insert([merged_row])
                elif how in ['left', 'outer']:
                    left_row = {col: chunk.columns[col].data[i] for col in self.schema.keys()}
                    right_row = {col: None for col in other.schema.keys()}
                    merged_row = {**left_row, **{f"{other.name}_{col}": val for col, val in right_row.items()}}
                    result_table.insert([merged_row])
        
        result_table.finalize()
        return result_table
    
    def to_parquet(self, file_path: str):
        """Save table to Parquet format"""
        # Convert all chunks to Arrow tables and write
        arrow_tables = []
        for chunk in self.chunks:
            arrow_arrays = [col.to_arrow() for col in chunk.columns.values()]
            arrow_schema = pa.schema([(name, array.type) for name, array in zip(chunk.columns.keys(), arrow_arrays)])
            arrow_tables.append(pa.Table.from_arrays(arrow_arrays, schema=arrow_schema))
        
        # Combine all chunks and write to Parquet
        combined_table = pa.concat_tables(arrow_tables)
        pq.write_table(combined_table, file_path)
    
    def from_parquet(self, file_path: str):
        """Load table from Parquet format"""
        table = pq.read_table(file_path)
        self.schema = {field.name: field.type.to_pandas_dtype() for field in table.schema}
        
        # Convert to chunks
        num_rows = table.num_rows
        chunk_size = 1024
        self.chunks = []
        
        for i in range(0, num_rows, chunk_size):
            end = min(i + chunk_size, num_rows)
            chunk_table = table.slice(i, end - i)
            
            columns = {}
            for field in chunk_table.schema:
                col_name = field.name
                arrow_array = chunk_table[col_name]
                columns[col_name] = ColumnVector.from_arrow(arrow_array, col_name)
            
            chunk = DataChunk(columns)
            chunk.size = end - i
            self.chunks.append(chunk)
        
        self.row_count = num_rows

# -------------------------
# Table Scanner (For Query Execution)
# -------------------------
class TableScanner:
    def __init__(self, table: Table):
        self.table = table
        self.filters = []
        self.projections = {}
        self._limit = None
        self._offset = 0
    
    def filter(self, condition: str):
        """Add a filter condition"""
        # Convert condition to lowercase to match column names
        condition_lower = condition.lower()
        self.filters.append(condition_lower)
        return self
    
    def project(self, expressions: Dict[str, str]):
        """Add projection expressions"""
        # Convert expressions to lowercase
        lower_expressions = {}
        for alias, expr in expressions.items():
            lower_expressions[alias] = expr.lower()
        self.projections.update(lower_expressions)
        return self
    
    def limit(self, n: int, offset: int = 0):
        """Set limit and offset"""
        self._limit = n
        self._offset = offset
        return self
    
    def execute(self) -> DataChunk:
        """Execute the scan and return results"""
        result_chunks = []
        rows_processed = 0
        rows_to_skip = self._offset
        
        for chunk in self.table.chunks:
            if self._limit and rows_processed >= self._limit:
                break
            
            # Skip empty chunks
            if chunk.size == 0:
                continue
                
            # Apply filters
            filtered_chunk = chunk
            for condition in self.filters:
                # Evaluate condition using numexpr - only use actual data, not entire capacity
                data_dict = {name: col.data[:col.size] for name, col in filtered_chunk.columns.items()}
                try:
                    mask = ne.evaluate(condition, data_dict)
                    filtered_chunk = filtered_chunk.filter(mask)
                except Exception as e:
                    raise ValueError(f"Error evaluating condition '{condition}': {e}")
            
            # Skip rows for offset
            if rows_to_skip > 0:
                if filtered_chunk.size <= rows_to_skip:
                    rows_to_skip -= filtered_chunk.size
                    continue
                else:
                    # Create a mask to skip first rows_to_skip rows
                    mask = np.ones(filtered_chunk.size, dtype=bool)
                    mask[:rows_to_skip] = False
                    filtered_chunk = filtered_chunk.filter(mask)
                    rows_to_skip = 0
            
            # Apply projections
            if self.projections:
                filtered_chunk = filtered_chunk.project(self.projections)
            
            # Apply limit
            if self._limit:
                remaining = self._limit - rows_processed
                if filtered_chunk.size > remaining:
                    # Create a mask to take only remaining rows
                    mask = np.ones(filtered_chunk.size, dtype=bool)
                    mask[remaining:] = False
                    filtered_chunk = filtered_chunk.filter(mask)
            
            if filtered_chunk.size > 0:
                result_chunks.append(filtered_chunk)
                rows_processed += filtered_chunk.size
        
        # Combine all result chunks into one
        if not result_chunks:
            return DataChunk({})
        
        # For simplicity, we'll materialize the first chunk
        # In a real implementation, we'd combine all chunks
        return result_chunks[0].materialize()

# -------------------------
# Database
# -------------------------
class Database:
    def __init__(self):
        self.tables: Dict[str, Table] = {}
    
    def execute(self, query: str):
        """Execute a SQL query"""
        # Normalize query by removing extra whitespace and newlines
        query = ' '.join(query.strip().split())
        
        if query.upper().startswith("CREATE TABLE"):
            return self._execute_create_table(query)
        elif query.upper().startswith("INSERT INTO"):
            return self._execute_insert(query)
        elif query.upper().startswith("SELECT"):
            return self._execute_select(query)
        else:
            raise ValueError(f"Unsupported query: {query}")
    
    def _execute_create_table(self, query: str):
        """Execute CREATE TABLE statement"""
        # Parse table name and schema
        pattern = r"CREATE TABLE (\w+) \((.*)\)"
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            raise ValueError("Invalid CREATE TABLE syntax")
        
        table_name = match.group(1).lower()
        columns_str = match.group(2)
        
        # Parse column definitions
        schema = {}
        for col_def in columns_str.split(','):
            col_def = col_def.strip()
            parts = re.split(r'\s+', col_def)
            if len(parts) < 2:
                raise ValueError(f"Invalid column definition: {col_def}")
            
            col_name = parts[0].lower()
            col_type = parts[1].lower()
            
            if col_type not in DTYPE_MAP:
                raise ValueError(f"Unsupported data type: {col_type}")
            
            schema[col_name] = DTYPE_MAP[col_type]
        
        self.tables[table_name] = Table(table_name, schema)
        return f"Table {table_name} created successfully"
    
    def _execute_insert(self, query: str):
        """Execute INSERT INTO statement"""
        # Parse INSERT statement
        pattern = r"INSERT INTO (\w+) \((.*?)\) VALUES \((.*?)\)"
        match = re.search(pattern, query, re.IGNORECASE)
        if not match:
            raise ValueError("Invalid INSERT syntax")
        
        table_name = match.group(1).lower()
        columns = [col.strip().lower() for col in match.group(2).split(',')]
        values = [val.strip() for val in match.group(3).split(',')]
        
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
        
        # Convert values to appropriate types
        row = {}
        table = self.tables[table_name]
        for col, val in zip(columns, values):
            if col not in table.schema:
                raise ValueError(f"Column {col} does not exist in table {table_name}")
            
            # Convert value to appropriate type
            dtype = table.schema[col]
            if val.upper() == 'NULL':
                row[col] = None
            elif np.issubdtype(dtype, np.integer):
                row[col] = int(val)
            elif np.issubdtype(dtype, np.floating):
                row[col] = float(val)
            elif np.issubdtype(dtype, np.bool_):
                row[col] = val.upper() == 'TRUE'
            else:
                # Remove quotes from string values
                if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                    val = val[1:-1]
                row[col] = val
        
        table.insert([row])
        table.finalize()
        return "1 row inserted"
    
    def _execute_select(self, query: str):
        """Execute SELECT statement"""
        # Parse SELECT statement
        select_match = re.search(r"SELECT (.+?) FROM", query, re.IGNORECASE)
        from_match = re.search(r"FROM (\w+)", query, re.IGNORECASE)
        where_match = re.search(r"WHERE (.+?)(?:GROUP BY|ORDER BY|LIMIT|$)", query, re.IGNORECASE)
        limit_match = re.search(r"LIMIT (\d+)", query, re.IGNORECASE)
        
        if not select_match or not from_match:
            raise ValueError("Invalid SELECT syntax")
        
        table_name = from_match.group(1).lower()
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
        
        table = self.tables[table_name]
        scanner = table.scan()
        
        # Parse selected columns/expressions
        select_exprs = [expr.strip() for expr in select_match.group(1).split(',')]
        projections = {}
        for expr in select_exprs:
            if expr == "*":
                # Select all columns
                for col in table.schema.keys():
                    projections[col] = col
            elif " AS " in expr.upper():
                # Handle aliases
                parts = re.split(r"\s+AS\s+", expr, flags=re.IGNORECASE)
                if len(parts) == 2:
                    projections[parts[1].strip()] = parts[0].strip()
            else:
                projections[expr] = expr
        
        scanner.project(projections)
        
        # Apply WHERE clause
        if where_match:
            where_condition = where_match.group(1).strip()
            scanner.filter(where_condition)
        
        # Apply LIMIT
        if limit_match:
            scanner.limit(int(limit_match.group(1)))
        
        # Execute the query
        result = scanner.execute()
        
        # Convert to pandas DataFrame for display
        data = {name: col.data[:col.size] for name, col in result.columns.items()}
        return pd.DataFrame(data)

# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    # Create database
    db = Database()
    
    # Create a table with a simpler query format
    db.execute("CREATE TABLE users (id INT, name STR, age INT, salary FLOAT)")
    
    # Insert some data
    db.execute("INSERT INTO users (id, name, age, salary) VALUES (1, 'Alice', 30, 50000.0)")
    db.execute("INSERT INTO users (id, name, age, salary) VALUES (2, 'Bob', 25, 45000.0)")
    db.execute("INSERT INTO users (id, name, age, salary) VALUES (3, 'Charlie', 35, 60000.0)")
    
    # Finalize the table to ensure all data is properly stored
    db.tables['users'].finalize()
    
    # Query the data
    print("Users with age > 25:")
    result = db.execute("SELECT id, name, age FROM users WHERE age > 25")
    print(result)
    print()
    
    # Create another table
    db.execute("CREATE TABLE departments (id INT, user_id INT, department STR)")
    
    db.execute("INSERT INTO departments (id, user_id, department) VALUES (1, 1, 'Engineering')")
    db.execute("INSERT INTO departments (id, user_id, department) VALUES (2, 2, 'Marketing')")
    db.execute("INSERT INTO departments (id, user_id, department) VALUES (3, 3, 'Sales')")
    
    # Finalize the second table
    db.tables['departments'].finalize()
    
    # Perform a join
    users = db.tables['users']
    depts = db.tables['departments']
    joined = users.join(depts, ['id'], ['user_id'], 'inner')
    
    # Query the joined table
    print("Joined users and departments:")
    scanner = joined.scan()
    result = scanner.execute()
    data = {name: col.data[:col.size] for name, col in result.columns.items()}
    print(pd.DataFrame(data))
    print()
    
    # Save to Parquet
    users.to_parquet("users.parquet")
    print("Saved users table to users.parquet")
    
    # Load from Parquet
    new_table = Table("users_copy", {})
    new_table.from_parquet("users.parquet")
    
    print("Loaded from Parquet:")
    scanner = new_table.scan()
    result = scanner.execute()
    data = {name: col.data[:col.size] for name, col in result.columns.items()}
    print(pd.DataFrame(data))