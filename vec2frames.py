def vec2frames( vec, Nw, Ns ):

    import math
    import numpy as np


    L = len( vec )                  # length of the input vector
    M = math.floor((L-Nw)/Ns+1)             # number of frames

    # figure out if the input vector can be divided into frames exactly
    E = (L-((M-1)*Ns+Nw))

    # see if padding is actually needed
    if( E>0 ):
        # how much padding will be needed to complete the last frame?
        P = Nw-E

        # pad with zeros
        vec[len(vec):] = [np.zeros(P)]  # pad with zeros

        # if not padding required, decrement frame count (not very elegant solution)
    else:
        M = M-1
        # increment the frame count
    M = M+1

    # compute index matrix in the direction ='rows'
    indf = Ns * list(range(0, M-1))                                # indexes for frames
    inds = list(range(0, Nw-1))                              # indexes for samples
    indexes = indf[:,np.ones(1,Nw)] + inds[np.ones(M,1),:]       # combined framing indexes

    # divide the input signal into frames using indexing
    frames = vec( indexes )


    window = np.hanning( Nw )

    frames = frames * np.diag( window )

    return frames, indexes