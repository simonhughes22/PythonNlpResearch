from IdGenerator import IdGenerator
from collections import defaultdict
import numpy as np
from gensim import matutils
from itertools import imap

class CoOccurenceMatrix(object):
    """ 
          Computes a co-occurence matrix between the row items (1 to M) and the column items (1 to M).
          Pass column_items = None to compute a matrix of items with themselves.
          Stored as a dictionary, can be convered into a gensim sparse or numpy densse matrix
    """
    def __init__(self, row_items, column_items = None, default_val = 1, cnt_fn = None):
      
        self.row_id_gen = IdGenerator()
        if cnt_fn == None:
            cnt_fn = lambda i:i #Identity function
        
        self.cnt_fn = cnt_fn
        if column_items == None:
            column_items = row_items
            self.ignore_matches = True
            self.col_id_gen = self.row_id_gen
        else:
            self.ignore_matches = False
            assert len(row_items) == len(column_items), "The number of row_items and column_items must be equal"
            self.col_id_gen = IdGenerator()
        
        self.default_val = default_val
        """ Add one to all counts for smoothing """
        self.row_tallies = self.compute_row_col_tallies(row_items, column_items)
        self.col_tallies = self.transpose_tallies(self.row_tallies)
        
        """ Take total as np array to add the smoothed counts """
        self.total_counts = np.sum(self.to_np_array(rows =True))
          
    def ensure_items_are_collections(self, items):
        if type(items[0]) == list or type(items[0]) == np.ndarray:
            return items
        else:
            return map(lambda i:[i], items)

    def transpose_tallies(self, tallies):
        tally = self.create_tally_dict()
        for a in tallies.keys():
            d = tallies[a]
            for b in d.keys():
                val = d[b]
                tally[b][a] = val
        return tally

    def create_tally_dict(self):
        return defaultdict(lambda:defaultdict(lambda:self.default_val))

    def compute_row_col_tallies(self, rows, columns):
        
        rows = self.ensure_items_are_collections(rows)
        columns = self.ensure_items_are_collections(columns)

        row_tally = self.create_tally_dict()
        for i in range(len(rows)):
            row = rows[i]
            col = columns[i]
            
            for r in row:
                r_id = self.row_id_gen.get_id(r)
                for c in col:
                    if self.ignore_matches and r == c:
                        continue
                    c_id = self.col_id_gen.get_id(c)
                    row_tally[r_id][c_id] += 1
        return row_tally

    def row_count(self):
        return self.row_id_gen.max_id() + 1

    def col_count(self):
        return self.col_id_gen.max_id() + 1

    def dict_to_gensim(self, tallies):
        if self.default_val != 0:
            raise Exception("Can only use a sparse matrix if zero's are the default value")
        
        return map(lambda (k,v) : (k, self.cnt_fn(v)), imap(lambda dct: sorted(dct.items(), key=lambda (k, v):k), tallies))

    def to_gensim_format(self, rows = True):
        """ Note that there is an issue with this. It does not currently add the default value 
            when a value is missing
        """
        if rows:
            return self.dict_to_gensim(self.row_tallies)
        else:
            return self.dict_to_gensim(self.col_tallies)

    def to_np_array(self, rows = True):
        num_rows = self.row_count()
        num_cols = self.col_count()
        
        mat = np.zeros((num_rows, num_cols)) + self.default_val
        for row_id in self.row_tallies.keys():
            row = self.row_tallies[row_id]
            for col_id, val in row.items():
                mat[row_id, col_id] = self.cnt_fn(val)

        """ If cols, simply transpose the matrix """
        if not rows:
            return mat.T
        return mat

    def __get_item__(self, val, rows = True, sparse = True, use_ids = None):
        
        if not sparse and use_ids != True:
            raise Exception("Cannot use original values if format is not sparse")
        
        size = -1
        if rows:
            dct = self.row_tallies[val]
            lookup = lambda id: self.row_id_gen.get_key(id)
            size = self.row_count()
        else:
            dct = self.col_tallies[val]
            lookup = lambda id: self.col_id_gen.get_key(id)
            size = self.col_count()
        
        sitems = sorted(dct.items(), key = lambda (k,v): k)
        if not sparse:
            return matutils.sparse2full(sitems, size)
            
        if use_ids:
            return sitems
        else:
            return [(lookup(k), v) for k,v in sitems]

    def get_gensim_row(self, val):
        return self.__get_item__(val, rows = True, sparse = True, use_ids = True)


    def get_row_with_values_from_values(self, rowval):
        rowid = self.row_id_gen.get_id(rowval)
        return self.__get_item__(rowid, rows=True, sparse=True, use_ids=False)

    def get_row_with_values(self, val):
        return self.__get_item__(val, rows = True, sparse = True, use_ids = False)
    
    def get_gensim_col(self, val):
        return self.__get_item__(val, rows = False, sparse = True, use_ids = True)
    
    def get_col_with_values(self, val):
        return self.__get_item__(val, rows = False, sparse = True, use_ids = False)

    def get_col_with_values_from_values(self, colval):
        colid = self.col_id_gen.get_id(colval)
        return self.__get_item__(colid, rows = False, sparse = True, use_ids = False)
    
    def get_row(self, val):
        return self.__get_item__(val, rows = True, sparse = False, use_ids = True)
    
    def get_column(self, val):
        return self.__get_item__(val, rows = False, sparse = False, use_ids = True)
    
    def __vals_to_index__(self, vals, rows = True):
        if rows:
            gen = self.row_id_gen
        else:
            gen = self.col_id_gen
        
        return np.array([gen.get_id(val) for val in vals])
    
    def get_sub_matrix(self, row_vals, col_vals = None, rows = True):
        
        row_ids = self.__vals_to_index__(row_vals, rows = True)
        if col_vals == None:
            col_ids = row_ids
        else:
            col_ids = self.__vals_to_index__(col_vals, rows = False)
        
        mat = np.ones((len(row_ids), len(col_ids))) + self.default_val
        
        """ For efficiency, don't build full matrix and then index """
        for ix_row, row_id in enumerate(row_ids):
            row = self.row_tallies[row_id]
            for ix_col, col_id in enumerate(col_ids):
                if col_id in row:
                    val = row[col_id]
                    mat[ix_row, ix_col] = val                    
        if not rows:
            return mat.T
        return mat

if __name__ == "__main__":
    mat = CoOccurenceMatrix(row_items=[
        ["a", "b", "c"],
        ["a", "d", "e"]
    ], column_items=[
        ["KK"],
        ["LL"]
    ], default_val=1)

    rows = mat.get_row_with_values_from_values("a")
    print rows
    cols = mat.get_col_with_values_from_values("KK")
    print cols
    pass
