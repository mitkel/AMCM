def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def f(x,y,z,w):
	return x*y+z*w

# A = {'x':3, 'y':4}
# B = {'z':5, 'w':1}
# C = merge_dicts(A, B)
# print C
# print f(**A, **B)