


def read_file(filename):
    coords = {}
    f = open(filename, "r")
    for line in f:
        if line == "NODE_COORD_SECTION\n":
            line = f.readline()
            while line != "EOF\n":
                tokens = line.split(" ")
                coords[tokens[0]] = (tokens[1], tokens[2].rstrip())
                line = f.readline()
    f.close()
    return coords

coords = read_file("att48.tsp")