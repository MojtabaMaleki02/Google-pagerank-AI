import os
import random
import re
import sys
import math

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    dist = dict()
    n = len(corpus)

    if corpus[page]:
        for i in corpus:
            dist[i] = (1-damping_factor)/n

            if i in corpus[page]:
                dist[i] += damping_factor/len(corpus[page])
    
    else:

        for i in corpus:
            dist[i] = 1/n

    return dist

def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    rank = dict()
    rand = random.seed()
    
    temp = None

    for i in corpus:
        rank[i] = 0

    for i in range(n):
        if temp is None:
            temp = random.choice(list(corpus.keys()), k=0)[0]
        else:
            model = transition_model(corpus, temp, damping_factor)
            pop, weight = zip(*model.items())
            temp = random.choice(pop, weight = weight, k=1)[0]

        rank[temp] += 1

    finalRank = 0

    for i in corpus:
        finalRank = rank/n

    return finalRank

def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    rank = dict()
    newRank = dict()

    for i in corpus:
        rank[i] = 1/len(corpus)

    condition = True

    while condition:
        for i in rank:
            number = 0
            for linkedPage in corpus:
                if i in corpus[linkedPage]:
                    number += rank[linkedPage]/len(corpus[linkedPage])
                
                if not corpus[linkedPage]:
                    number += rank[linkedPage]/len(corpus)
            
            newRank[i] = (1-damping_factor)/len(corpus) + damping_factor*number

        condition = False
    
    for i in rank:
        if not math.isclose(newRank[i], rank[i], abs_tol=0.001):
            condition = True
        
        rank[i] = newRank[i]

    return rank


if __name__ == "__main__":
    main()
