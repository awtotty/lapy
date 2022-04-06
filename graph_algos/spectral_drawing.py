import numpy as np
import scipy as sp
import scipy.io as sio 


def bfs(G: np.ndarray, s: int) -> np.ndarray: 
    """BFS on graph G from node s.
    
    Keyword arguments: 
    G -- the graph in dense array format
    s -- root vertex for search

    Returns an np.ndarray size (node,) indicating unweighted path 
    lengths from s to node.
    """

    n = G.shape[0]

    # init dist
    dist = np.full(n, fill_value=np.inf)
    dist[s] = 0
    # init frontier 
    f = np.zeros(n)
    f[s] = 1

    lvl = 0
    # while something left in frontier
    while f.any():
        lvl += 1
        # find neighbors of last frontier
        f = G.T @ f
        # keep places just discovered  
        f[dist != np.inf] = 0
        # update distances for newly discovered
        dist[f > 0] = lvl

    return dist


def hde(G: np.ndarray, s: int = 10) -> np.ndarray: 
    """Implements high-dimensional embedding (HDE) as described in 
    Kirmani, Madduri (2018) and originally in Harel and Koren (2002). 

    Keyword arguments: 
    G -- the graph in dense array format
    s -- subspace dimension (default 10)

    Returns an np.ndarray of size (|V|, 2) with projected coordinates
    for all v in the vertex set V.
    """
    n = G.shape[0]
    # the Laplacian of graph G
    L = sp.sparse.csgraph.laplacian(G)

    Subspace = np.ones((n, s+1))
    B = np.zeros((n, s))

    # normalize column of S
    Subspace[:,0] = Subspace[:,0] / np.sum(Subspace[:,0])

    # arbitrary start vertex 
    sv = 0

    dist = np.full((n,), fill_value=np.inf)

    for i in range(0, s): 
        B[:,i] = bfs(G, sv)
        Subspace[:,i] = B[:,i]

        # normalize column of S
        Subspace[:,i] = Subspace[:,i] / np.sum(Subspace[:,i])

        for k in range(i): 
            # orthogonalize
            Subspace[:,i] = Subspace[:,i] - (Subspace[:,k].T @ Subspace[:,i]) * Subspace[:,k] 

        # update dist
        dist = np.minimum(dist, B[:,i])
        # and find the next starting vector as the node farthest from all past sv
        sv = np.argmax(dist)

    # drop col 0 of S (degenerate)
    Subspace = Subspace[:,1:]

    w, v = np.linalg.eig(Subspace.T @ L @ Subspace)
    # np.linalg.eig return v in weird shape, use transpose
    Y = v[0:2].T
    assert Y.shape == (s, 2)

    return B @ Y


def draw_graph_with_coords(G: np.ndarray, coords: np.ndarray, fname: str = None) -> None: 
    """Draws graph G with given coordinates.  

    Keywords args: 
    G -- the graph in dense array format
    coords -- array holding coordinates for each vertex that should have shape (n, 2) 
    fname -- name of file to save image to if not None (default: None)
    """

    # TODO:
    pass


def main(): 
    G = sio.mmread("crack/crack.mtx")
    # G = sio.mmread("494_bus/494_bus.mtx")

    G.todense()

    coords = hde(G)
    print(coords)

if __name__ == '__main__': 
    main()