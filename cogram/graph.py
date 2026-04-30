from collections import defaultdict, Counter


def build_dbg_from_reads(reads, k):
    graph = defaultdict(Counter)

    for read in reads:
        for i in range(len(read) - k + 1):
            kmer = read[i:i+k]

            if 'N' in kmer:
                continue

            graph[kmer[:-1]][kmer[1:]] += 1

    return graph


def build_dbg_streaming(file, k, max_kmers=None):
    graph = defaultdict(Counter)
    count = 0

    with open(file, 'r') as f:
        while True:
            f.readline()
            read = f.readline().strip()
            f.readline()
            f.readline()

            if not read:
                break

            for i in range(len(read) - k + 1):
                kmer = read[i:i+k]

                if 'N' in kmer:
                    continue

                graph[kmer[:-1]][kmer[1:]] += 1
                count += 1

                if max_kmers is not None and count >= max_kmers:
                    return graph

    return graph


def filter_graph(graph, min_freq=3):
    new_graph = {}

    for u in graph:
        filtered = [v for v, c in graph[u].items() if c >= min_freq]
        if filtered:
            new_graph[u] = filtered

    return new_graph


def create_edge_dataset(graph, max_edges=None):
    edges = []

    for u in graph:
        neighbors = graph[u]

        # neighbors is list after filtering, Counter before filtering
        if isinstance(neighbors, dict):
            iterable = neighbors.keys()
        else:
            iterable = neighbors

        for v in iterable:
            edges.append((u, v))

            if max_edges and len(edges) >= max_edges:
                return edges

    return edges


def score_edges_in_batches(edges, model, batch_size=1000):
    import torch
    from utils import encode_kmer

    scores = {}

    for i in range(0, len(edges), batch_size):
        batch = edges[i:i+batch_size]

        X = [encode_kmer(u) + encode_kmer(v) for u, v in batch]
        X = torch.tensor(X, dtype=torch.float32)

        preds = model(X).detach().numpy()

        for j, e in enumerate(batch):
            scores[e] = preds[j][0]

    return scores


def extract_contigs(graph, edge_scores):
    visited_edges = set()
    contigs = []

    for start in graph:
        current = start
        path = [current]

        while current in graph:
            neighbors = graph[current]

            if not neighbors:
                break

            best = None
            best_score = -1

            for v in neighbors:
                edge = (current, v)

                if edge in visited_edges:
                    continue

                score = edge_scores.get(edge, 0)

                if score > best_score:
                    best = v
                    best_score = score

            if best is None:
                break

            visited_edges.add((current, best))
            path.append(best)
            current = best

        if len(path) > 1:
            contigs.append(path)

    return contigs