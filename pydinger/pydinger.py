class Grid:
    '''This class is a grid implementation for holding our input data'''
    def __init__(self, axes, values):
        self.axes = axes
        self.values = values
        self.dim = len(axes)

def read_file(filename, dim):
    coords = [[] for i in range(dim)]#make the empty coordinates 'array'
    values = []
    with open(filename) as f:
        for line in f:
            words = line.split(' ')
            for i in range(dim):
                coords[i].append(float(words[i]))
            values.append(float(words[dim]))#this makes a big long potentials list for higher dims
    return(Grid(coords, values))
