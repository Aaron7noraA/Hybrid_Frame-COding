training_pairs = {
    'nonRNN': {
        'step1': [[0], [6], [0, 3, 6], [0, 1, 3], [1, 2, 3], [3, 4, 6], [4, 5, 6]],
        # 'step1': ([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]),
        'step2': ([0, 2], [2, 4], [4, 6]),
        'step3': ([0, 3], [3, 6]),
        'step123': ([0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [0, 2], [2, 4], [4, 6], [0, 3], [3, 6]),
        'step136': ([0, 1], [2, 3], [4, 5], [0, 3], [3, 6], [0, 6]),
    },
    'RNN': {
        'step1': [[0], [0, 1], [1, 2], [2, 3], [3, 4]],
        'step2': [[0], [0, 2], [2, 4], [4, 6]],
        'step3': [[0], [0, 3], [3, 6]],
        'step123': ([[0], [0, 1], [1, 2], [2, 3], [3, 4]], [[0], [0, 2], [2, 4], [4, 6]], [[0], [0, 3], [3, 6]]),
        'step136': ([[0], [0, 1], [1, 2], [2, 3], [3, 4]], [[0], [0, 3], [3, 6]], [[0], [0, 6]]),
        'sstep1': [[0], [0, 1], [1, 2], [2, 3]],
    }
}

def getOrder(pairs):
    num_frame = max([max(p) for p in pairs]) + 1
    order = [0 for _ in range(num_frame)]

    for p in pairs:
        if len(p) == 1:
            order[p[0]] = 0
        elif len(p) == 2:
            order[p[1]] = order[p[0]] + 1
        elif len(p) == 3:
            order[p[1]] = min(order[p[0]], order[p[2]]) + 1

    return order

def get_coding_pairs(intra_period, gop_size):        
    pairs = [(0,), (intra_period,)] + \
            list(zip(range(0, intra_period - gop_size, gop_size), range(gop_size, intra_period, gop_size)))
    step = gop_size
    while step > 1:
        pairs += list(zip(range(0, intra_period - step + 1, step),
                          range(step//2, intra_period - step//2 + 1, step), 
                          range(step, intra_period + 1, step)))
        step = step // 2

    assert len(pairs) == intra_period + 1, \
        f"Number of pairs: {len(pairs)} is greater than {intra_period} + 1: {pairs}"

    target = [p[0] if len(p) == 1 else p[1] for p in pairs]
    for i in range(intra_period + 1):
        assert i in target, f"Frame: {i} does not covered by {pairs}"

    if gop_size == 1:
        pairs.pop(1)
        print("LDP mode on")


    return pairs

def get_identity_lmda(intra_lmda, pair):
    return intra_lmda