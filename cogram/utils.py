def read_fastq(file, limit=None):
    reads = []
    with open(file, 'r') as f:
        count = 0
        while True:
            f.readline()
            seq = f.readline().strip()
            f.readline()
            f.readline()
            if not seq:
                break
            reads.append(seq)
            count += 1
            if limit and count >= limit:
                break
    return reads


def encode_kmer(kmer):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    return [mapping.get(c, 0) for c in kmer]


def read_fasta(file):
    genome = ""
    with open(file) as f:
        for line in f:
            if not line.startswith(">"):
                genome += line.strip()
    return genome


def get_true_edges(genome, k):
    true_edges = set()
    for i in range(len(genome) - k):
        kmer = genome[i:i+k]
        true_edges.add((kmer[:-1], kmer[1:]))
    return true_edges


def path_to_sequence(path):
    if not path:
        return ""
    seq = path[0]
    for node in path[1:]:
        seq += node[-1]
    return seq