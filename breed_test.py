import random


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    # geneA = int(random.random() * len(parent1))
    # geneB = int(random.random() * len(parent1))
    
    geneA = 3
    geneB = 6

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

def breed_better(parent1, parent2):
    child = []

    # geneA = random.randint(0, len(parent1))
    # geneB = random.randint(0, len(parent2))

    geneA = 3
    geneB = 6

    start = min(geneA, geneB)
    end = max(geneA, geneB)

    slice1 = parent1[start : end]
    ptr2 = 0

    for ptr1 in range(0, len(parent1)):
        if ptr1 in range(start, end):
            child.append(parent1[ptr1])
        else:
            gene2 = parent2[ptr2]
            while gene2 in slice1:
                ptr2 = ptr2 + 1
                gene2 = parent2[ptr2]
            child.append(gene2)
            ptr2 = ptr2 + 1

    return child


parent1 = [2, 4, 3, 5, 6, 1]
parent2 = [1, 3, 4, 6, 5, 2]

print(breed(parent1, parent2))

print(breed_better(parent1, parent2))