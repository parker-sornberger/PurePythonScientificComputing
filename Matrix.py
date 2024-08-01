import math
import operator
import random
import itertools as it
from copy import deepcopy
from collections import deque
from functools import cached_property

# for inverses https://integratedmlai.com/matrixinverse/
class Array(list):
    def __init__(self, *args, **kwargs):
        super(Array, self).__init__(*args, **kwargs)
        self._row = 0
        self._col = 0
        
        self.shape = Array._calc_shape(self)
        
        self.oneD = len(self.shape) == 1
        self.twoD = len(self.shape) == 2
        self.nD = len(self.shape) > 3
        
        self._get_rowcol()
        self._inner_shape = (self._row, self._col)
        
        self._setup_generators()
        self.square_type = self._row == self._col
    
    
    def prettystr(self, full = False):
        def print_row(row):
            maxwidth = 20
            
        def print_mat(mat):
            pass
        
        
    def __getitem__(self, index):
        def _slice(mat, i, colslice=False):
            start = 0 if i.start is None else i.start
            stop = len(mat) if i.stop is None else i.stop
            step = 1 if i.step is None else i.step
            ret = Array()

            for I in range(start, stop, step):
                ret.append(mat[I])
            return Array.from_container(ret)
        if isinstance(index, slice):
            return _slice(self, index)
            
        if isinstance(index, tuple):
            rval = None
            for k, i in enumerate(index):
                if isinstance(i, slice):
                    rval = _slice(self, i, k) if rval is None else _slice(rval, i, k)
                    continue
                rval = self[i] if rval is None else rval[i]
                
            return rval
        
        return super().__getitem__(index)
    def __setitem__(self, i, o):
        if isinstance(i, tuple):

            item = self[i[:-1]]
            item[i[-1]] = o
            return
        return super().__setitem__(i, o)
        

    def __add__(self, other):
        copy = self.copy()
        
        if not isinstance(other, Array):
            for index in range(len(self)):
                
                copy[index]  = copy[index]+ other
        else:
            assert self.shape == other.shape, "you suck"
            for index, (self_row, other_row) in enumerate(zip(self, other)):
                
                copy[index] = self_row + other_row
        return copy
    
    def __sub__(self, other):
        copy = self.copy()

        
        if not isinstance(other, Array):
            for index in range(len(self)):
                
                copy[index]  = copy[index] - other
        else:
            assert self.shape == other.shape, "you suck"
            for index, (self_row, other_row) in enumerate(zip(self, other)):

                copy[index] = self_row - other_row

        return copy
        
    def __mul__(self, other):
        copy = self.copy()
        
        if not isinstance(other, Array):
            for index in range(len(self)):
                
                copy[index]  = copy[index] * other
        else:
            
            for index, (self_row, other_row) in enumerate(zip(self, other)):
                
                copy[index] = self_row * other_row
        return copy
    def _matmul(self, other):
        temp = Array()

        for mat, mat2 in zip(self.matiter(), other.matiter()):
            temp.append(mat @ mat2)
        temp.shape = Array._calc_shape(temp)
        bigger_shape = self.shape if len(self.shape) > len(other.shape) else other.shape
        newshape = list(bigger_shape)[:-2] + [self._row, self._row]

        temp = temp.reshape(*newshape)
        return temp
        
    def _mul_mat_and_vec(self, other):
        temp = Array()
        mat, vec= (self, other) if other.oneD  else (other, self)

        for row in mat.rowiter():

            temp.append(sum(a * b for a, b in zip(row, other) ))
        temp.shape = Array._calc_shape(temp)
        temp = temp.reshape(*mat.shape[:-1])
        return temp
    def __matmul__(self, other):

        assert self._row == other._col
        
        if other.oneD or self.oneD:
            return self._mul_mat_and_vec(other)
        if other.nD or self.nD:

            return self._matmul(other)

        size = len(other[0]) if len(other.shape)>1 else len(other)

        temp = Array(self._batch(it.starmap(self._sumprod, it.product(self, other.T)), size))
        return temp

    def __truediv__(self, other):
        copy = self.copy()
        
        if not isinstance(other, Array):
            for index in range(len(self)):
                
                copy[index]  = copy[index] / other
        else:
            assert self.shape == other.shape, "you suck"
            for index, (self_row, other_row) in enumerate(zip(self, other)):
                
                copy[index] = self_row / other_row
        return copy
    def __floordiv__(self, other):
        copy = self.copy()
        
        if not isinstance(other, Array):
            for index in range(len(self)):
                
                copy[index]  = copy[index] // other
        else:
            assert self.shape == other.shape, "you suck"
            for index, (self_row, other_row) in enumerate(zip(self, other)):
                
                copy[index] = self_row // other_row
        return copy
    def __mod__(self, other):
        copy = self.copy()
        
        if not isinstance(other, Array):
            for index in range(len(self)):
                
                copy[index]  = copy[index] % other
        else:
            assert self.shape == other.shape, "you suck"
            for index, (self_row, other_row) in enumerate(zip(self, other)):
                
                copy[index] = self_row % other_row
        return copy
    
        
        
    def __rmatmul__(self, other):
        return self.__matmul__(self, other)
    def __rmul__(self, other):
        return self.__mul__(other)
    def __rsub__(self, other):
        return self.__add__(other)
    def __radd__(self, other):
        return self.__add__(other)
    def __rtruediv__(self, other):
        return self.__truediv__(other)
    def  __rfloordiv__(self, other):
        return self. __floordiv__(other)
    def __rmod__(self, other):
        return self.__mod__(other)
    def __lt__(self, other):
        res = Array()
        if not isinstance(other, Array):
            
            for item in self.flatiter():
                res.append(item < other)
            
        else:
            for myitem, otheritem in zip(self.flatiter(), other.flatiter()):
                res.append(myitem < otheritem)
        res = res.reshape(*self.shape)
        return res
            
    def __le__(self, other):
        res = Array()
        if not isinstance(other, Array):
            
            for item in self.flatiter():
                res.append(item <= other)
            
        else:
            for myitem, otheritem in zip(self.flatiter(), other.flatiter()):
                res.append(myitem <= otheritem)
        res = res.reshape(*self.shape)
        return res

    def __gt__(self, other):
        res = Array()
        if not isinstance(other, Array):
            
            for item in self.flatiter():
                res.append(item > other)
            
        else:
            for myitem, otheritem in zip(self.flatiter(), other.flatiter()):
                res.append(myitem > otheritem)
        res = res.reshape(*self.shape)
        return res
    def __ge__(self, other):
        res = Array()
        if not isinstance(other, Array):
            
            for item in self.flatiter():
                res.append(item >= other)
            
        else:
            for myitem, otheritem in zip(self.flatiter(), other.flatiter()):
                res.append(myitem >= otheritem)
        res = res.reshape(*self.shape)
        return res
    def __neg__(self):
        n = Array()
        for i in self.flatiter():
            n.append(-i)
        return n.reshape(*self.shape)



    def _sumprod(self, mat1, mat2):
        return Array.dot(mat1, mat2)
    def _batch(self,chunk, size):
        chunk = iter(chunk)
        while batch:= Array(it.islice(chunk, size)):
            yield batch
    def _get_rowcol(self):
        
        if len(self.shape) <2:
            self._col = self.shape[-1]
            
        else:
            self._row, self._col = self.shape[-2], self.shape[-1]
    def _setup_chain(self, times):
       
        if times == 0:
            return iter(self)

        y_obj = it.chain.from_iterable(self)
        for _ in range(times-1):
            y_obj = it.chain.from_iterable(y_obj)
        return y_obj
    def _setup_generators(self):
        self._flat = self._setup_chain(times = len(self.shape)-1)
        
        self._rowiter = self._setup_chain(times = len(self.shape) -2)
        self._matiter = self._setup_chain(times = len(self.shape)-3)
    def flatiter(self):
        self._flat = self._setup_chain(times = len(self.shape)-1)
        yield from self._flat
        self._flat = self._setup_chain(times = len(self.shape)-1)
    def rowiter(self):
        if len(self.shape) < 2:
            yield self
        else:
            self._rowiter = self._setup_chain(times = len(self.shape) -2)
            yield from self._rowiter
            self._rowiter = self._setup_chain(times = len(self.shape) -2)
    def matiter(self):
        if len(self.shape) < 3:
            yield self
        else:
            self._matiter = self._setup_chain(times = len(self.shape)-3)
            yield from self._matiter
            self._matiter = self._setup_chain(times = len(self.shape)-3)
            
    def numel(self):
        return math.prod(self.shape)
    def copy(self):

        
        
        return self.reshape(*self.shape)

    def flatten(self):

        return Array(self.flatiter())

    def flat_index(self, index):
        if not index:
            indices = tuple(0 for _ in self.shape)
            return self[indices]
        def index_math(index, shape):
            pass
        indices = index_math(index, self.shape)
        return self[indices]

    def reshape(self, *new_shape):

        def check_shape(shape):
            self_flat = math.prod(self.shape)
            new_flat = math.prod(new_shape)
            if new_flat != self_flat:
                raise ValueError(F"Cannot reshape array with {self_flat} elements into array with {new_flat} elements")
        def batch_factory(chunk, col):
            cbatch = self._batch(chunk, col)
            return Array(cbatch)
        if not isinstance(new_shape[0], int):
            new_shape=new_shape[0]
        if not self.shape[0] and len(self):
            self.shape = Array._calc_shape(self)
        check_shape(new_shape)

        flat = self.flatiter()
        if len(new_shape) == 1:
            return Array(flat)
        elif len(new_shape) > 1:
            _, *new_shape = new_shape
            chunk = flat
            
            for size in reversed(new_shape):
                chunk = batch_factory(chunk, size)
            return chunk
    def abs(self):
        res = Array()
        for item in self.flatiter():
            res.append(abs(item))
        return res.reshape(*self.shape)

    def unravel_index(self, index,):
        indices = []
        for dim in reversed(self.shape):
            div, mod = divmod(index, dim)
            indices.append(mod)
            index //= dim
        return tuple(reversed(indices))
    def flat_index(self, index):
        multi_index = self.unravel_index(index)
        return self[multi_index]

    def sum(self):
        return sum(self.flatiter())
    def astype(self, totype):
        if totype not in {int, float}:
            print(F"{totype} not good :( 👯👽🏫")
            return self
        return Array(map(totype, self.flatiter())).reshape(*self.shape)
    @cached_property
    def T(self):
        if len(self.shape) <2 :

            t = self.copy()
        elif len(self.shape) == 2:

            t = Array(zip(*self))
        else:

            t = self.reshape(tuple(reversed(self.shape)))

        return t
    
        

    @classmethod
    def _calc_shape(cls, arr):
        if not arr:
            return (0,)
        shape = [len(arr)]
        
        item = arr[0]
        while not isinstance(item, (int, float)):
            shape.append(len(item))
            item = item[0]

        return tuple(shape)
        
    @classmethod
    def dot(cls, mat1, mat2):
        return sum(it.starmap(lambda x, y : x * y, zip(mat1, mat2)))
    
    @classmethod
    def det(cls, arr):
        def get_det(A):
            n = len(A)
            AM = A.copy()
        
            
            for fd in range(n):  # fd stands for focus diagonal
                if AM[fd][fd] == 0:
                    AM[fd][fd] = 1.0e-18  # Cheating by adding zero + ~zero
                for i in range(fd+1, n):  # skip row with fd in it.
                    crScaler = AM[i][fd] / AM[fd][fd]  # cr stands for "current row".
                    for j in range(n):  
                        AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
        
            
            product = 1.0
            for i in range(n):
                product *= AM[i][i] 
        
            return product
        
        row, col = arr._inner_shape
        assert row == col, "Can only get dets for square(ish) arrays"
        if len(arr.shape) == 2:
            return get_det(arr)
        detshape = arr.shape[:-2]
        dets = Array.zeros(*detshape).flatten()
        for i, mat in enumerate(arr.matiter()):
            dets[i] = get_det(mat)
        return dets.reshape(*detshape)
        
    
    @classmethod
    def _fill(cls, shape, fill_val, isfunc=False):
        if not isfunc:
            arr = cls(fill_val for _ in range(math.prod(shape)))
        else:
            arr = cls(fill_val() for _ in range(math.prod(shape)))
        arr = arr.reshape(shape)
        
        return arr

    @classmethod
    def zeros(cls, *shape):
        arr = cls._fill(shape, fill_val=0)
        
        return arr
    @classmethod
    def ones(cls, *shape):
        arr = cls._fill(shape, fill_val=1)
        
        return arr
    @classmethod
    def rand(cls, *shape):
        arr = cls._fill(shape, fill_val=random.random, isfunc = True)
        return arr
    @classmethod
    def linspace(cls, start, stop, step):
        abs_diff = abs(stop-start)
        if stop < start:
            start,stop = stop, start
        if step < 1 or step < abs_diff:
            steps = abs_diff // step
        elif step > abs_diff:
            steps = step
        steps = int(steps)
        step = abs_diff/(steps-1)
        _space  = cls(start + step *  interval for interval in range(steps))
       
        return _space
        
    @classmethod
    def arange(cls, start, stop=0, step = 1):
        
        if not isinstance(step, int):
            return cls.linspace(start, stop, step)
        if stop:
            _range = cls(range(start, stop, step))
        if not stop and step==1:
            _range = cls(range(start))
        elif not stop and step !=1:
            _range  = cls(range(0, start, step))
        
        return _range
    @classmethod
    def eye(cls, size):
        mat = cls.zeros(size, size)
        for i in range(size):
            mat[i][i] = 1
        return mat
    @classmethod
    def from_container(cls, container):
        if hasattr(container, "tolist"):
            container = container.tolist()
        arr = cls(container)
        
        arr = arr.flatten().reshape(*arr.shape)
        return arr
    @classmethod
    def sign(cls, mat, tol=1e-8):
    
        flat = mat.flatten()
        for i, item in enumerate(flat):
            if item > tol:
                flat[i] =1
            elif (not item) or (item < tol and item > 0):
                flat[i] = 0
            else:
                flat[i] = -1
        flat = flat.reshape(*mat.shape)
        return flat
    
    @classmethod
    def squeeze(cls, arr, dim = 0):
        assert arr.shape[dim] == 1, F"Can't squeeze dims > 1 shape @ dim is {arr.shape.index(dim)}"
        newshape = list(arr.shape)
        newshape.pop(dim)
        return arr.reshape(*newshape)
    @classmethod
    def unsqueeze(cls, arr, dim = 0):
        newshape = list(arr.shape)
        newshape.insert(dim, 1)
        return arr.reshape(*newshape)
    @classmethod
    def total_squeeze(cls, arr):
        def get_1_index(shape):
            return [i for i, x in enumerate(shape) if x == 1]
        proxy = arr.copy()
        if not any(x==1 for x in arr.shape):
            return proxy
        
        one_index = get_1_index(proxy.shape)
        num_is_1 = len(one_index)
        while (num_is_1 := len(one_index)):
            proxy = cls.squeeze(proxy, dim = one_index[0])
            one_index = get_1_index(proxy.shape)
        return proxy
    @classmethod
    def expand_every_dim(cls, arr, forward = False, backward = False, both = True):
        if forward:
            newshape = list(it.chain.from_iterable((x, 1) if x != 1 else (x,) for x in arr.shape ))
        elif backward:
            newshape = list(it.chain.from_iterable(( 1, x) if x != 1 else (x,) for x in arr.shape))
        elif both:
            if arr.shape[0] != 1:
                newshape = [1, ] + list(it.chain.from_iterable(( x, 1) if x != 1 else (x,) for x in arr.shape))
            else:
                newshape = list(it.chain.from_iterable(( x, 1) if x != 1 else (x,) for x in arr.shape))
                
       
        return arr.reshape(*newshape)
    @classmethod
    def diag_flat(cls, diag):
        
        mat = cls.zeros(diag.shape[0], diag.shape[0])
        for i, elem in enumerate(diag):
            mat[i][i] = elem
        return mat
    @classmethod
    def diag(cls, arr, getmat = False):
        row, col = arr._inner_shape
        assert row == col, "Can only get diagonals for square(ish) arrays"
        
        if len(arr.shape) == 2:
            diag_holder = cls.zeros(row, col) if getmat else cls.zeros(row,)
            for i in range(row):
                if getmat:
                    diag_holder[i][i] = arr[i][i]
                else:
                    diag_holder[i] = arr[i][i]
        else:
            diag_holder = Array()
            for mat in arr.matiter():
                _holder = cls.zeros(row, col) if getmat else cls.zeros(row,)
                for i in range(row):
                    if getmat:
                        _holder[i][i] = mat[i][i]
                    else:
                        _holder[i] = mat[i][i]
                diag_holder.append(_holder)
                
            
            newshape = arr.shape if getmat else (arr.shape[:-2] + (row,))
            diag_holder = diag_holder.flatten().reshape(*newshape)
            
        return diag_holder
    @classmethod
    def check_singular(cls, mat, tol=1e-5):
        if not mat.square_type:
            return True
        det = cls.det(mat)
        if abs(det) < tol:
            return True
        return False 
    @classmethod
    def inv(cls, mat, tol=1e-5):
        #got this from https://integratedmlai.com/matrixinverse/
        if not mat.twoD:
            return #(raise error)
        cls.check_singular(mat, tol)
        n = len(mat)
        AM = mat.copy()
        I = cls.eye(n)
        IM = I.copy()
     
        # Section 3: Perform row operations
        indices = list(range(n)) # to allow flexible row referencing ***
        for fd in range(n): # fd stands for focus diagonal
            fdScaler = 1.0 / AM[fd][fd]
            # FIRST: scale fd row with fd inverse. 
            for j in range(n): # Use j to indicate column looping.
                AM[fd][j] *= fdScaler
                IM[fd][j] *= fdScaler
            # SECOND: operate on all rows except fd row as follows:
            for i in indices[0:fd] + indices[fd+1:]: 
                # *** skip row with fd in it.
                crScaler = AM[i][fd] # cr stands for "current row".
                for j in range(n): 
                    # cr - crScaler * fdRow, but one element at a time.
                    AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                    IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
        return IM
    @classmethod
    def allclose(cls, m1, m2, tol = 1e-6):
        for item1, item2 in zip(m1.flatiter(), m2.flatiter()):
            if abs(item1-item2) > tol:
                return False
        return True
        
    @classmethod
    def all(cls, condition):
        for cond in condition.flatiter():
            if not cond:
                return False
        return True
    @classmethod
    def all(cls, condition):
        for cond in condition.flatiter():
            if  cond:
                return True
        return False

    @classmethod
    def where(cls, condition):
        indices = []
        
        size = condition.numel()
        
        for i in range(size):
            index = condition.unravel_index(i)
            item = condition[index]
            if item:
                indices.append(item)
        return indices 
    @classmethod
    def norm(cls, mat):
        thenorm = 0
        for item in mat.flatiter():
            thenorm += (abs(item)**2)
        return math.sqrt(thenorm)
    @classmethod
    def _col_index_2D(cls, mat, index):
        cont = cls(row[index] for row in mat.rowiter())

        return cls.from_container(cont)
            
            
        
    @classmethod
    def qr(cls, A):
        if not A.twoD:
            return A
        
        m, n = A.shape  
        Q = cls.eye(m)  # or my_eye(m) -- see below
        R = A.copy()
        if m == n:
            end = n-1
        else:
            end = n
        for i in range(0, end):
            H = Array.eye(m)

            a = cls._col_index_2D(R[i:], i)

            norm_a = Array.norm(a)
            if a[0] < 0.0:
                norm_a = -norm_a
            v = a / (a[0] + norm_a)
            v[0] = 1.0

            h = Array.eye(len(a)) 
            
            h = h - ((2 / cls.dot(v, v))  * (cls.unsqueeze(v, dim=1) @ cls.unsqueeze(v, dim=0) ))


            hflat = h.flatiter()
            for J in range(i, m):
                
                for K in range(i, m):
                    H[J][K] = next(hflat)
                    
            Q = Q @ H
           

            R = H @ R

        return Q, R


    @classmethod
    def is_upper_tri(cls, mat, tol=1e-5):
        n = len(mat)
        for i in range(n):
            for j in range(i):
                if abs(mat[i][j]) > tol:
                    return False
        return True 
    
    @classmethod
    def eig(cls, mat):
        n = len(mat)
        X = mat.copy()
        pq = cls.eye(n)
        max_ct = 100
        ct = 0
        while ct < max_ct:
            Q, R = cls.qr(X)
            pq = pq @ Q
            X = R @ Q
            ct += 1
            if cls.is_upper_tri(X, 1e-8):
                break
        evals = Array.zeros(n)
        for i in range(n):
            evals[i] = X[i][i]
        evecs = pq.copy()
        
        return evals,evecs
            

    
    
    
    
    
    
    
    
    
    
