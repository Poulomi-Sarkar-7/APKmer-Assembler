def compute_n50(lengths):
    lengths = sorted(lengths, reverse=True)
    total = sum(lengths)
    cumsum = 0

    for l in lengths:
        cumsum += l
        if cumsum >= total / 2:
            return l
    return 0


def compute_coverage(contigs, genome):
    covered = set()

    for contig in contigs:
        start = genome.find(contig[:20])
        if start != -1:
            for i in range(len(contig)):
                covered.add(start + i)

    return len(covered) / len(genome)