Module tf_som
=============
This module provides a Tensorflow 2 implementation of the Self Organizing Maps (SOM).

Functions
---------

    
`normalize(df, dtype: type = numpy.float32) ‑> None`
:   Normalizes a Pandas Data Frame (DF) for using it with TF2_SOM.
    
    Arguments
    ---------
    df : pd.DataFrame
        Pandas Data Frame.
    dtype : type
        Neural network data type (default: np.float32).

    
`setup_tensorflow_for_cpus(num_threads: int = None) ‑> None`
:   Setups Tensorflow 2 for CPU parallelization.
    
    Arguments
    ---------
    num_threads : int
        Number of threads (default: multiprocessing.cpu_count())

Classes
-------

`SOM(m: int, n: int, dim: int, seed: float = None, dtype: type = numpy.float32, learning_rate: float = None, radius: float = None, sigma: float = None, epochs: int = 100, decay_function=<function _asymptotic_decay>)`
:   Tensorflow 2 implementation of the Self Organizing Maps (SOM).
    
    Initializes a Self Organizing Maps.
    
    A rule of thumb to set the size of the grid for a dimensionality
    reduction task is that it should contain 5 * sqrt(N) neurons
    where N is the number of samples in the dataset to analyze.
    
    Arguments
    ---------
    m : int
        Number of neuron rows.
    n : int
        Number of neuron columns.
    dim : int
        Dimensionality of the input data.
    seed : int
        Seed of the random generators (default: None).
    dtype : type
        Neural network data type (default: np.float32).
    learning_rate : float
        Starting value of the learning rate (default: 0.3).
    radius : float
        Starting value of the neighborhood radius (default: max(m, n) / 2.0).
    sigma : float
        Fixed standard deviation coefficient of the neighborhood function (default: 1.0).
    epochs : int
        Number of epochs to train for (default: 100).
    decay_function : function
        Function that reduces learning_rate and sigma at each iteration (default: 1.0 / (1.0 + 2.0 * epoch / epochs)).

    ### Class variables

    `BMUs`
    :   Best Matching Units

    ### Methods

    `activation_map(self, input_vectors: numpy.ndarray) ‑> numpy.ndarray`
    :   Returns a matrix containing the number of times the neuron (i,j) have been winner for the input.
        
        Parameters
        ----------
        input_vectors : np.ndarray
            Input data.

    `distance_map(self) ‑> numpy.ndarray`
    :   Returns the distance map of the neural network weights.

    `get_centroids(self) ‑> numpy.ndarray`
    :   Returns of the neural network weights (shape = [m, n, dim]).

    `get_quantization_errors(self) ‑> numpy.ndarray`
    :   Returns the quantization errors (one value per epoch).

    `get_topographic_errors(self) ‑> numpy.ndarray`
    :   Returns the topographic errors (one value per epoch).

    `get_weights(self) ‑> numpy.ndarray`
    :   Returns the neural network weights (shape = [m * n, dim]).

    `input_map(self, input_vectors: numpy.ndarray) ‑> numpy.ndarray`
    :   Returns a vector containing the coordinates (i,j) of the winner for each input.
        
        Parameters
        ----------
        input_vectors : np.ndarray
            Input data.

    `load(self, filename: str, file_format: str = 'fits') ‑> None`
    :   Loads the trained neural network from a file.
        
        Parameters
        ----------
        filename : str
            Filename.
        file_format : str
            File format (supported formats: (fits, hdf5), default: fits).

    `save(self, filename: str, file_format: str = 'fits') ‑> None`
    :   Saves the trained neural network to a file.
        
        Parameters
        ----------
        filename : str
            Filename.
        file_format : str
            File format (supported formats: (fits, hdf5), default: fits).

    `train(self, input_vectors: numpy.ndarray, progress_bar: bool = True) ‑> None`
    :   Trains the trained neural network.
        
        Parameters
        ----------
        input_vectors : np.ndarray
            Training data.
        progress_bar : bool
            Specifying whether a progress bar have to be shown (default: True).

    `winners(self, input_vectors: numpy.ndarray) ‑> tf_som.SOM.BMUs`
    :   Returns a vector of best matching unit (aka winners) locations and indices for the input.
        
        Parameters
        ----------
        input_vectors : np.ndarray
            Input data.