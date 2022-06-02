import argparse

from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw
import numpy as np
import scipy as sp
import scipy.io as sio 
import matplotlib.pyplot as plt


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


def koren(G: np.ndarray, dim: int, v1: np.ndarray, eps: float) -> np.ndarray: 
    pass


def weighted_centroid_smoothing(D: np.ndarray, A: np.ndarray, n_smooth: int, coords: np.ndarray) -> np.ndarray:  
    pass


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
        sum = np.sum(Subspace[:,i])
        if sum != 0: 
            # fix problems with NaN and inf
            Subspace = np.nan_to_num(Subspace)
            Subspace[:,i] = Subspace[:,i] / sum

        for k in range(i): 
            # orthogonalize
            Subspace[:,i] = Subspace[:,i] - (Subspace[:,k].T @ Subspace[:,i]) * Subspace[:,k] 

        # update dist
        dist = np.minimum(dist, B[:,i])
        # and find the next starting vector as the node farthest from all past sv
        sv = np.argmax(dist)

    # drop col 0 of S (degenerate)
    Subspace = Subspace[:,1:]

    try: 
        w, v = np.linalg.eig(Subspace.T @ L @ Subspace)
    except Exception as e: 
        print("Something went wrong!") 
        print(e)
        return


    # np.linalg.eig return v in weird shape, use transpose
    Y = v[0:2].T
    assert Y.shape == (s, 2)

    return B @ Y


def spectral(G: np.ndarray) -> np.ndarray: 
    pass


def draw_graph_with_coords(G: np.ndarray, coords: np.ndarray, imsize: int = 720, fname: str = None) -> None: 
    """Draws graph G with given coordinates.  

    Keywords args: 
    G -- the graph in dense array format
    coords -- array holding coordinates for each vertex that should have shape (n, 2) 
    fname -- name of file to save image to if not None (default: None)
    """

    # fix problems with NaN and inf
    coords = np.nan_to_num(coords)
    x_bounds = [int(np.min(coords[:,0])), int(np.max(coords[:,0]))]
    y_bounds = [int(np.min(coords[:,1])), int(np.max(coords[:,1]))]
    
    offset = max(np.max(np.abs(x_bounds)), np.max(np.abs(y_bounds)))

    # max_abs_coord = int(np.max(coords)+1)
    scale_factor = imsize / (2 * offset)
    # scale_factor = 10
    # imsize = scale_factor * 2 * offset 
    im = Image.new(mode="RGB", size=(imsize, imsize))
    draw = ImageDraw.Draw(im)

    def transform_coord(c): 
        # return scale_factor*(np.multiply(c, [1, -1]) + [max_abs_coord, max_abs_coord])
        return scale_factor*np.add(c, [offset, offset])

    # for all (u,v) in G draw segment from coords[u] to coords[v] 
    for u, v, val in zip(G.row, G.col, G.data):
        line = [tuple(transform_coord(coords[u])), tuple(transform_coord(coords[v]))]
        draw.line(line, width=0)
    
    # origin dot
    origin = transform_coord([0,0])
    draw.ellipse((origin[0], origin[1], origin[0]+scale_factor, origin[1]+scale_factor), fill=(0,255,0))
    
    if fname is not None: 
        im.save(fname)


def bitmap(G: np.ndarray, n_colors: int = 10, fname: str = None) -> None: 
    """Produces a bitmap/heatmap of the adjacency matrix of graph G. 

    Keyword args: 
    G -- the graph in dense array format
    n_colors -- number of colors to use in the map (default: 10)
    fname -- name of file to save image to if not None (default: None)
    """

    figsize = np.array(G.shape)//100 + np.ones(2) 
    plt.figure(figsize=figsize)

    # colormap
    viridis_arr = plt.get_cmap('viridis')(np.linspace(0, 1, 256))
    colors = viridis_arr[0::len(viridis_arr)//(n_colors-1)]
    # add white as first color
    colors = np.vstack( ([1, 1, 1, 1], colors) )
    cmap = ListedColormap(colors)

    # process graph
    G[G != 0] = 1

    plt.imshow(G, vmin=0, vmax=np.max(G), cmap=cmap)
    plt.axis('off')

    if fname is not None: 
        plt.savefig(fname)


def main(): 
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('name', metavar="Graph name", type=str, 
                        help='Name of graph')
    parser.add_argument('-bit', action='store_true',
                        help='Create a bitmap image')
    parser.add_argument('-hde', action='store_true', 
                        help='Create an HDE embedding image')

    args = parser.parse_args()

    # graph_name = "fe_4elt2"
    # graph_name = "crack"
    # graph_name = "abb313"
    # graph_name = "barth5"
    graph_name = args.name

    print(f"Reading graph {graph_name}")
    G = sio.mmread(f"graphs/{graph_name}/{graph_name}.mtx")

    # bitmap
    if args.bit: 
        print(f"Creating bitmap of {graph_name}")
        G_dense = np.array(G.todense())
        bitmap(G_dense, n_colors=10, fname=f"out/{graph_name}_bit.png", )

    # hde projection
    if args.hde: 
        print(f"Creating HDE graph of {graph_name}")
        coords = hde(G, s=50)
        # print(coords)
        draw_graph_with_coords(G, coords, fname=f"out/{graph_name}.png")


if __name__ == '__main__': 
    main()