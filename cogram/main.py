import argparse
import torch

from utils import *
from graph import *
from model import EdgeClassifier
from eval import *


def run_pipeline(
    k=21,
    train_reads=100000,
    max_train_edges=200000,
    max_kmers_stream=5000000,
    reads_file="../yeast_combined.fastq",
    genome_file="../genomes/yeast.fasta",
):
    # ======================
    # TRAIN
    # ======================
    print("Training phase...")
    train_reads_data = read_fastq(reads_file, limit=train_reads)

    train_graph = build_dbg_from_reads(train_reads_data, k)
    train_edges = create_edge_dataset(train_graph, max_train_edges)

    genome = read_fasta(genome_file)
    true_edges = get_true_edges(genome, k)

    X = []
    y = []

    for u, v in train_edges:
        X.append(encode_kmer(u) + encode_kmer(v))
        y.append(1 if (u, v) in true_edges else 0)

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    model = EdgeClassifier(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(100):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # ======================
    # BUILD GRAPH
    # ======================
    print("\nBuilding full graph...")
    graph = build_dbg_streaming(reads_file, k, max_kmers_stream)
    graph = filter_graph(graph, min_freq=3)

    edges = create_edge_dataset(graph)
    print("Edges:", len(edges))

    # ======================
    # SCORE EDGES
    # ======================
    print("Scoring edges...")
    edge_scores = score_edges_in_batches(edges, model)

    # ======================
    # EXTRACT CONTIGS
    # ======================
    print("Extracting contigs...")
    paths = extract_contigs(graph, edge_scores)

    contigs = [path_to_sequence(p) for p in paths]

    print("Number of contigs:", len(contigs))

    # ======================
    # SAVE
    # ======================
    with open("contigs.fasta", "w") as f:
        for i, seq in enumerate(contigs):
            f.write(f">contig_{i}\n{seq}\n")

    print("Saved contigs.fasta")

    print("Nodes:", len(graph))
    print("Edges:", sum(len(v) for v in graph.values()))

    # ======================
    # EVALUATION
    # ======================
    lengths = [len(c) for c in contigs]

    total_contigs = len(contigs)
    total_length = sum(lengths)
    longest = max(lengths) if lengths else 0
    n50 = compute_n50(lengths) if lengths else 0
    coverage = compute_coverage(contigs, genome) if lengths else 0

    print("\n=== Evaluation ===")
    print("Total contigs:", total_contigs)
    print("Total assembled length:", total_length)
    print("Longest contig:", longest)
    print("N50:", n50)
    print("Approx coverage:", round(coverage, 4))

    return {
        "k": k,
        "total_contigs": total_contigs,
        "total_assembled_length": total_length,
        "longest_contig": longest,
        "n50": n50,
        "approx_coverage": round(coverage, 4),
        "contigs_fasta": "contigs.fasta",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run CoGram genome assembly pipeline")
    parser.add_argument("--k", type=int, default=21)
    parser.add_argument("--train_reads", type=int, default=100000)
    parser.add_argument("--max_train_edges", type=int, default=200000)
    parser.add_argument("--max_kmers_stream", type=int, default=5000000)
    parser.add_argument("--reads_file", default="../yeast_combined.fastq")
    parser.add_argument("--genome_file", default="../genomes/yeast.fasta")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        k=args.k,
        train_reads=args.train_reads,
        max_train_edges=args.max_train_edges,
        max_kmers_stream=args.max_kmers_stream,
        reads_file=args.reads_file,
        genome_file=args.genome_file,
    )
