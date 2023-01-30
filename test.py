from pagerank import *

DAMPING = 0.85
SAMPLES = 10000

corpus = {"1.html": {"2.html", "3.html"},"2.html":{"3.html"},"3.html":{"2.html"}}

ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
print(ranks)
ranks = iterate_pagerank(corpus, DAMPING)
print(ranks)