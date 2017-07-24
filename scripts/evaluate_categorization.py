import json
from web import embeddings

from web.datasets.categorization import fetch_AP, fetch_BLESS, fetch_battig, fetch_ESSLI_2c, fetch_ESSLI_2b, fetch_ESSLI_1a
from web.evaluate import evaluate_categorization

# EMBEDDINGS
word_embeddings = {}

# # GloVe
for dim in [50, 100, 200, 300]:
    word_embeddings["Glove wiki-6B {}".format(dim)] = ["fetch_GloVe", {"dim": dim, "corpus": "wiki-6B"}]

for dim in [25, 50, 100, 200]:
    word_embeddings["Glove twitter-27B {}".format(dim)] = ["fetch_GloVe", {"dim": dim, "corpus": "twitter-27B"}]

for corpus in ["common-crawl-42B", "common-crawl-840B"]:
    word_embeddings["GloVe {} {}".format(corpus, 300)] = ["fetch_GloVe", {"dim": 300, "corpus": corpus}]

# # NMT
word_embeddings["NMT FR"] = ["fetch_NMT", {"which": "FR"}]
word_embeddings["NMT DE"] = ["fetch_NMT", {"which": "DE"}]

# # PDC and HDC
for dim in [50, 100, 300]:
    word_embeddings["PDC {}".format(dim)] = ["fetch_PDC", {"dim": dim}]
    word_embeddings["HDC {}".format(dim)] = ["fetch_HDC", {"dim": dim}]

# # SG and
word_embeddings["SG_GoogleNews"] = ["fetch_SG_GoogleNews", {}]
word_embeddings["LexVec"] = ["fetch_LexVec", {}]

# DATASETS
datasets = {
           "AP": fetch_AP,
           "BLESS": fetch_BLESS,
           "Battig": fetch_battig,
           "ESSLI_2c": fetch_ESSLI_2c,
           "ESSLI_2b": fetch_ESSLI_2b,
           "ESSLI_1a": fetch_ESSLI_1a()
           }

# MODELS
models = ["kmeans", "agglomerative", "mean-shift", "spectral"]

results = {}
for e_name in word_embeddings:
    # TODO: in the future change prints to loggings
    print 'running', e_name
    fn, kwargs = word_embeddings[e_name]
    embedding = getattr(embeddings, fn)(**kwargs)  # Staszek is the smartest!
    results[e_name] = {}
    for data_name in datasets:
        print 'running', data_name
        data = datasets[data_name]()
        results[e_name][data_name] = {}
        for model in models:
            print 'running', model
            # TODO: in the future use same measure as in Brown Clustering
            results[e_name][data_name][model] = evaluate_categorization(w=embedding, X=data.X, y=data.y, method=model, seed=879)
            
with open("results", 'w') as f:
    json.dump(results, f, indent=4)