import re
from collections import defaultdict

class TinyDB:
    def __init__(self):
        self.tables = {}

    def execute(self, query):
        query = query.strip()
        cmd = query.split()[0].upper()
        if cmd == "CREATE":
            self.create_table(query)
        elif cmd == "INSERT":
            self.insert_into(query)
        elif cmd == "SELECT":
            return self.select(query)
        elif cmd == "UPDATE":
            self.update(query)
        elif cmd == "DELETE":
            self.delete(query)
        else:
            raise Exception("Unsupported command")

    def create_table(self, query):
        pattern = r"CREATE TABLE (\w+) \((.+)\)"
        match = re.match(pattern, query, re.IGNORECASE)
        if not match:
            raise Exception("Invalid CREATE TABLE syntax")
        table_name = match.group(1)
        columns = [col.strip() for col in match.group(2).split(",")]
        self.tables[table_name] = {"columns": columns, "rows": []}
        print(f"Table '{table_name}' created with columns {columns}")

    def insert_into(self, query):
        # Supports two forms:
        # 1. INSERT INTO table VALUES (val1, val2, ...)
        # 2. INSERT INTO table (col1, col2) VALUES (val1, val2)
        pattern1 = r"INSERT INTO (\w+) VALUES \((.+)\)"
        pattern2 = r"INSERT INTO (\w+) \((.+)\) VALUES \((.+)\)"
        match = re.match(pattern2, query, re.IGNORECASE)
        if match:
            table_name = match.group(1)
            cols = [c.strip() for c in match.group(2).split(",")]
            values = [v.strip().strip("'\"") for v in re.split(r",(?![^(]*\))", match.group(3))]
            if table_name not in self.tables:
                raise Exception(f"Table '{table_name}' does not exist")
            if len(cols) != len(values):
                raise Exception("Column count does not match value count")
            row = [None] * len(self.tables[table_name]['columns'])
            for col, val in zip(cols, values):
                if col not in self.tables[table_name]['columns']:
                    raise Exception(f"Column '{col}' does not exist in table '{table_name}'")
                idx = self.tables[table_name]['columns'].index(col)
                row[idx] = val
            # Fill missing columns with None or empty string
            for i in range(len(row)):
                if row[i] is None:
                    row[i] = ""
            self.tables[table_name]['rows'].append(row)
            print(f"Inserted {row} into '{table_name}'")
            return

        match = re.match(pattern1, query, re.IGNORECASE)
        if not match:
            raise Exception("Invalid INSERT INTO syntax")
        table_name = match.group(1)
        values = [v.strip().strip("'\"") for v in re.split(r",(?![^(]*\))", match.group(2))]
        if table_name not in self.tables:
            raise Exception(f"Table '{table_name}' does not exist")
        if len(values) != len(self.tables[table_name]['columns']):
            raise Exception("Value count does not match column count")
        self.tables[table_name]['rows'].append(values)
        print(f"Inserted {values} into '{table_name}'")

    def select(self, query):
        # Regex to parse SELECT with optional INNER JOIN, WHERE, GROUP BY, ORDER BY
        select_pattern = (
              r"SELECT\s+(.+?)\s+FROM\s+(\w+)(?:\s+(\w+))?"
              r"(?:\s+INNER\s+JOIN\s+(\w+)\s+(\w+)\s+ON\s+(.+?))?"
              r"(?:\s+WHERE\s+(.+?))?"
              r"(?:\s+GROUP\s+BY\s+(.+?))?"
              r"(?:\s+ORDER\s+BY\s+(.+?))?\s*$"
        )
        match = re.match(select_pattern, query, re.IGNORECASE | re.DOTALL)
        if not match:
            raise Exception("Invalid SELECT syntax")

        select_columns = [col.strip() for col in match.group(1).split(",")]
        table1 = match.group(2)
        alias1 = match.group(3)
        table2 = match.group(4)
        alias2 = match.group(5)
        join_condition = match.group(6)
        where_clause = match.group(7)
        group_by_clause = match.group(8)
        order_by_clause = match.group(9)

        if table1 not in self.tables:
            raise Exception(f"Table '{table1}' does not exist")
        if table2 and table2 not in self.tables:
            raise Exception(f"Table '{table2}' does not exist")

        alias_map = {}
        alias_map[alias1 if alias1 else table1] = table1
        if table2:
            alias_map[alias2 if alias2 else table2] = table2

        def resolve_column(col):
            func_match = re.match(r"(\w+)\((.+)\)", col)
            if func_match:
                func = func_match.group(1).upper()
                col_inner = func_match.group(2).strip()
                tbl, colname = self._resolve_col_alias(col_inner, alias_map, table1)
                return ("FUNC", func, tbl, colname)
            else:
                tbl, colname = self._resolve_col_alias(col, alias_map, table1)
                return ("COL", tbl, colname)

        output_cols = [resolve_column(col) for col in select_columns]

        combined_rows = []
        if not table2:
            for row in self.tables[table1]['rows']:
                if self._where_matches_complex(row, self.tables[table1], where_clause, alias_map, single_table=table1):
                    combined_rows.append({table1: row})
        else:
            join_pattern = r"(\w+)\.(\w+) *= *(\w+)\.(\w+)"
            jm = re.match(join_pattern, join_condition.strip())
            if not jm:
                raise Exception("Invalid JOIN condition")
            left_alias, left_col, right_alias, right_col = jm.groups()
            if left_alias not in alias_map or right_alias not in alias_map:
                raise Exception("Unknown table alias in JOIN condition")

            left_table = self.tables[alias_map[left_alias]]
            right_table = self.tables[alias_map[right_alias]]
            left_idx = left_table['columns'].index(left_col)
            right_idx = right_table['columns'].index(right_col)

            for left_row in left_table['rows']:
                for right_row in right_table['rows']:
                    if left_row[left_idx] == right_row[right_idx]:
                        combined = {
                            alias_map[left_alias]: left_row,
                            alias_map[right_alias]: right_row,
                        }
                        if self._where_matches_complex(combined, None, where_clause, alias_map):
                            combined_rows.append(combined)

        # If GROUP BY exists, do grouping and aggregation
        if group_by_clause:
            group_cols = [c.strip() for c in group_by_clause.split(",")]
            group_keys = []
            group_map = defaultdict(list)

            # Resolve group columns to (table, col) tuples
            group_cols_resolved = [self._resolve_col_alias(c, alias_map, table1) for c in group_cols]

            for row_set in combined_rows:
                key = tuple(
                    row_set[tbl][self.tables[tbl]['columns'].index(col)]
                    for tbl, col in group_cols_resolved
                )
                group_map[key].append(row_set)

            results = []
            for key, rows in group_map.items():
                result_row = []
                for col in output_cols:
                    if col[0] == "COL":
                        tbl, colname = col[1], col[2]
                        if (tbl, colname) in group_cols_resolved:
                            # output group key value
                            idx = group_cols_resolved.index((tbl, colname))
                            result_row.append(key[idx])
                        else:
                            # If non-aggregated column not in group by => error
                            raise Exception(f"Column {colname} must be in GROUP BY or used in aggregation")
                    elif col[0] == "FUNC":
                        func, tbl, colname = col[1], col[2], col[3]
                        vals = []
                        for row_set in rows:
                            v = row_set[tbl][self.tables[tbl]['columns'].index(colname)]
                            try:
                                v = float(v)
                            except ValueError:
                                v = 0
                            vals.append(v)
                        if func == "COUNT":
                            result_row.append(len(vals))
                        elif func == "SUM":
                            result_row.append(sum(vals))
                        elif func == "AVG":
                            result_row.append(sum(vals) / len(vals) if vals else 0)
                        elif func == "MIN":
                            result_row.append(min(vals) if vals else None)
                        elif func == "MAX":
                            result_row.append(max(vals) if vals else None)
                        else:
                            raise Exception(f"Unsupported function '{func}'")
                results.append(result_row)
            # ORDER BY support for grouped results
            if order_by_clause:
                results = self._order_by_results(results, select_columns, order_by_clause)
            return results

        else:
            # No GROUP BY
            agg_present = any(col[0] == "FUNC" for col in output_cols)
            if agg_present:
                # Aggregate without group by: operate over all rows
                results = []
                for col in output_cols:
                    if col[0] == "FUNC":
                        func, tbl, colname = col[1], col[2], col[3]
                        values = []
                        for row_set in combined_rows:
                            val = row_set[tbl][self.tables[tbl]['columns'].index(colname)]
                            try:
                                val = float(val)
                            except ValueError:
                                val = 0
                            values.append(val)
                        if func == "COUNT":
                            results.append(len(values))
                        elif func == "SUM":
                            results.append(sum(values))
                        elif func == "AVG":
                            results.append(sum(values) / len(values) if values else 0)
                        elif func == "MIN":
                            results.append(min(values) if values else None)
                        elif func == "MAX":
                            results.append(max(values) if values else None)
                        else:
                            raise Exception(f"Unsupported function '{func}'")
                    else:
                        raise Exception("Mixing aggregates and normal columns without GROUP BY not supported")
                return [results]
            else:
                # Just normal columns
                results = []
                for row_set in combined_rows:
                    row_out = []
                    for col in output_cols:
                        if col[0] == "COL":
                            tbl, colname = col[1], col[2]
                            idx = self.tables[tbl]['columns'].index(colname)
                            row_out.append(row_set[tbl][idx])
                        else:
                            raise Exception("Unexpected function column without GROUP BY")
                    results.append(row_out)
                if order_by_clause:
                    results = self._order_by_results(results, select_columns, order_by_clause)
                return results

    def _order_by_results(self, results, select_columns, order_by_clause):
        order_fields = [f.strip() for f in order_by_clause.split(",")]

        def sort_key(row):
            keys = []
            for f in order_fields:
                parts = f.split()
                col = parts[0]
                direction = parts[1].upper() if len(parts) > 1 else "ASC"
                try:
                    idx = select_columns.index(col)
                except ValueError:
                    idx = None
                    for i, sc in enumerate(select_columns):
                        if sc.endswith(col):
                            idx = i
                            break
                    if idx is None:
                        raise Exception(f"ORDER BY column '{col}' not in SELECT list")
                val = row[idx]
                keys.append(val if direction == "ASC" else self._reverse_val(val))
            return keys

        results.sort(key=sort_key)
        return results

    def _reverse_val(self, v):
        try:
            return -float(v)
        except:
            return ''.join(chr(255 - ord(c)) for c in v)

    def _resolve_col_alias(self, col, alias_map, default_table):
        if '.' in col:
            a, c = col.split('.', 1)
            if a not in alias_map:
                raise Exception(f"Unknown table alias '{a}' in column '{col}'")
            tbl = alias_map[a]
            if c not in self.tables[tbl]['columns']:
                raise Exception(f"Column '{c}' does not exist in table '{tbl}'")
            return tbl, c
        else:
            tbl = default_table
            if col not in self.tables[tbl]['columns']:
                raise Exception(f"Column '{col}' does not exist in table '{tbl}'")
            return tbl, col

    def update(self, query):
        pattern = r"UPDATE (\w+) SET (.+?)( WHERE (.+))?$"
        match = re.match(pattern, query, re.IGNORECASE)
        if not match:
            raise Exception("Invalid UPDATE syntax")
        table_name = match.group(1)
        set_clause = match.group(2)
        where_clause = match.group(4)

        if table_name not in self.tables:
            raise Exception(f"Table '{table_name}' does not exist")

        assignments = {}
        for assign in set_clause.split(","):
            col_val = assign.strip().split("=", 1)
            if len(col_val) != 2:
                raise Exception("Invalid SET clause")
            col, val = col_val[0].strip(), col_val[1].strip().strip("'\"")
            if col not in self.tables[table_name]['columns']:
                raise Exception(f"Column '{col}' does not exist in table '{table_name}'")
            assignments[col] = val

        updated_count = 0
        for row in self.tables[table_name]['rows']:
            if self._where_matches_complex(row, self.tables[table_name], where_clause, {table_name: table_name}, single_table=table_name):
                for col, val in assignments.items():
                    idx = self.tables[table_name]['columns'].index(col)
                    row[idx] = val
                updated_count += 1

        print(f"Updated {updated_count} rows in '{table_name}'")

    def delete(self, query):
        pattern = r"DELETE FROM (\w+)( WHERE (.+))?$"
        match = re.match(pattern, query, re.IGNORECASE)
        if not match:
            raise Exception("Invalid DELETE syntax")
        table_name = match.group(1)
        where_clause = match.group(3)

        if table_name not in self.tables:
            raise Exception(f"Table '{table_name}' does not exist")

        initial_len = len(self.tables[table_name]['rows'])
        self.tables[table_name]['rows'] = [
            row for row in self.tables[table_name]['rows']
            if not self._where_matches_complex(row, self.tables[table_name], where_clause, {table_name: table_name}, single_table=table_name)
        ]
        deleted_count = initial_len - len(self.tables[table_name]['rows'])
        print(f"Deleted {deleted_count} rows from '{table_name}'")

    def _where_matches_complex(self, row_or_combined, table, where_clause, alias_map, single_table=None):
        if not where_clause:
            return True
        or_parts = [p.strip() for p in re.split(r"\s+OR\s+", where_clause, flags=re.IGNORECASE)]
        for or_part in or_parts:
            and_parts = [p.strip() for p in re.split(r"\s+AND\s+", or_part, flags=re.IGNORECASE)]
            if all(self._eval_condition(cond, row_or_combined, table, alias_map, single_table) for cond in and_parts):
                return True
        return False

    def _eval_condition(self, cond, row_or_combined, table, alias_map, single_table):
        cond = cond.strip()
        m_val = re.match(r"(\w+)\.(\w+) *= *'?(.*?)'?$", cond)
        m_col = re.match(r"(\w+)\.(\w+) *= *(\w+)\.(\w+)", cond)
        m_val_single = re.match(r"(\w+) *= *'?(.*?)'?$", cond)

        if m_val:
            alias, col, val = m_val.groups()
            if alias not in alias_map:
                raise Exception(f"Unknown alias '{alias}' in WHERE clause")
            tbl = self.tables[alias_map[alias]]
            if col not in tbl['columns']:
                raise Exception(f"Column '{col}' does not exist in table '{alias_map[alias]}'")
            idx = tbl['columns'].index(col)
            if isinstance(row_or_combined, dict):
                return row_or_combined[alias_map[alias]][idx] == val
            else:
                if single_table != alias_map[alias]:
                    raise Exception(f"Alias {alias} does not match table {single_table}")
                return row_or_combined[idx] == val
        elif m_col:
            alias1, col1, alias2, col2 = m_col.groups()
            if alias1 not in alias_map or alias2 not in alias_map:
                raise Exception("Unknown alias in WHERE clause")
            tbl1 = self.tables[alias_map[alias1]]
            tbl2 = self.tables[alias_map[alias2]]
            idx1 = tbl1['columns'].index(col1)
            idx2 = tbl2['columns'].index(col2)
            return row_or_combined[alias_map[alias1]][idx1] == row_or_combined[alias_map[alias2]][idx2]
        elif m_val_single and single_table:
            col, val = m_val_single.groups()
            tbl = self.tables[single_table]
            if col not in tbl['columns']:
                raise Exception(f"Column '{col}' does not exist in table '{single_table}'")
            idx = tbl['columns'].index(col)
            return row_or_combined[idx] == val
        else:
            raise Exception(f"Unsupported WHERE condition: {cond}")


# Example usage:
db = TinyDB()

db.execute("CREATE TABLE users (id, name, age)")
db.execute("CREATE TABLE orders (order_id, user_id, amount)")

# Insert with column names
db.execute("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)")
db.execute("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)")
db.execute("INSERT INTO orders VALUES (101, 1, 100)")
db.execute("INSERT INTO orders VALUES (102, 2, 150)")
db.execute("INSERT INTO orders VALUES (103, 1, 200)")

print("\n-- GROUP BY with aggregates --")
query = ("SELECT u.id, COUNT(o.order_id), SUM(o.amount), MIN(o.amount), MAX(o.amount) "
         "FROM users u INNER JOIN orders o ON u.id = o.user_id "
         "GROUP BY u.id "
         "ORDER BY u.id")
result = db.execute(query)
print(result)

print("\n-- SELECT with INSERT column names --")
result = db.execute("SELECT id, name, age FROM users ORDER BY age DESC")
print(result)
