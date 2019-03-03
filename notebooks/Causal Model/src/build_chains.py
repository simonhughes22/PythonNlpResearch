from NgramGenerator import compute_ngrams

def build_chains_inner(tree, l, visited, depth=0):
    chains = []
    if l not in tree:
        return chains
    for r in tree[l]:
        if r in visited:
            continue
        visited.add(r)  # needed to prevent cycles, which cause infinite recursion
        extensions = build_chains_inner(tree, r, visited, depth + 1)
        visited.remove(r)
        for ch in extensions:
            chains.append([r] + ch)
        if not extensions:
            chains.append([r])
    return chains

def build_chains(tree):
    lhs_items = set(tree.keys())
    rhs_items = set()
    for l, rhs in tree.items():
        rhs_items.update(rhs)

    chains = []
    # starting positions of each chain are those appearing on the lhs but not the rhs
    start_codes = lhs_items - rhs_items
    for l in start_codes:
        rhs = tree[l]
        for r in rhs:
            for ch in build_chains_inner(tree, r, {l, r}, 0):
                chains.append([l, r] + ch)
    return chains

def extend_chains(chains):
    ext_chains = set()
    for tokens in chains:
        ext_chains.add(",".join(tokens))
        ngrams = compute_ngrams(tokens, max_len=None, min_len=3)
        for t in ngrams:
            ext_chains.add(",".join(t))
    return ext_chains

def get_distinct_chains(chains):
    s_chains = [",".join(ch) for ch in chains]
    distinct_chains = set()
    for ch1 in s_chains:
        found_match = False
        for ch2 in s_chains:
            if ch1 == ch2:
                continue
            if ch1 in ch2:
                found_match = True
                break
        distinct_chains.add(ch1)
    return distinct_chains