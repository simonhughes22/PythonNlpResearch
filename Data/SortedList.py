import bisect

class SortedList(object):
    
    def __init__(self, cmp, list = None):
        if list == None:
            list = []
        self.list = list
        self.sort()
    
    def sort(self):
        l = []
        for i in range(len(self.list)):
            bisect.insort(l, self.list[i])
        self.list = l
            
    def insert(self, value):
        bisect.insort(self.list, value)
    
    def len(self):
        return len(self.list)
    
    def show(self):
        print self.list
    
    def search(self,value):
        left = bisect.bisect_left(self.list, value)
        if abs( self.list[ min( left, len(self.list) -1) ] - value) >= abs(self.list[left-1] - value):
            return self.list[left-1]
        else:
            return self.list[left]

class TopItemList(object):
    """ 
        Maintains a sorted list of fixed size
        of the top items
    """
    def __init__(self, max_size):
        self.sl = SortedList([])
        self.max_size = max_size
        
        """ Insert implementation once we are at size """
        def insert(item):
            smallest = self.sl.list[0]
            if item > smallest:
                self.sl.insert(item)
                self.sl.list.pop(0)
                return smallest
            else:
                return None
            
        """ This is an optimization so we don't have to check the size on every insert
            Once it hits max size, swap the method call to the one above
        """ 
        def init_insert(item):
            self.sl.insert(item)
            if len(self.sl.list) >= self.max_size:
                self.insert = insert
            return None
        
        """ 
            Insert an item. If the item is larger than the smallest item
            then remove and return the smallest item
        """
        self.insert = init_insert
    
    def list(self):
        return self.sl.list[:]
    
    def len(self):
        return self.sl.len()
    
    def __repr__(self):
        if self.len() > 10:
            return str(self.sl.list[0:10]) + " ...."
        return str(self.sl.list)

if __name__ == "__main__":
    
    test = range(20)
    test = test + [ 1,21,22,-1,100,99,-2]
    
    l = TopItemList(10)
    for t in test:
        item = l.insert(t)
        if item == None:
            print str(t), " Ignored"
        else:
            print str(item), " Removed"
            print str(t), " Added"
            
    print l.list()
    
    