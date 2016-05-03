def frames2vec( frames, Ns ):

    import math
    import numpy as np

    # rows as frames  direction ='rows'

    # get number of frames and frame length
    [ M, Nw ] = frames.shape

    # otherwise, framing indexes were provided
    indexes = Ns
    Ns = indexes(2,1)-indexes(1,1)


    # determine signal duration
    L = max(indexes)

    window = np.hanning(L)

    # allocate storage
    vec  = np.zeros(1, L)
    wsum = np.zeros(1, L)

    # overlap-and-add syntheses, Allen & Rabiner's method
    # overlap-and-add frames
    for m in xrange(M):
        vec(indexes(m,:)) = vec(indexes(m,:)) + frames(m,:)

    # overlap-and-add window samples
    for m in xrange(M):
        wsum(indexes(m,:)) = wsum(indexes(m,:)) + window
    # for some tapered analysis windows, use:
    # wsum( wsum<1E-2 ) = 1E-2;

    # divide out summed-up analysis windows
    vec = vec./wsum
    return vec
