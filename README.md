# Require Activity 25.1

## Initial Code Base
My initial code was taken from the example Juypter notebooks passed onto us by Carleton during our live sessions. The code I used was...
- Examples/example_student.ipynb
- Examples/example_student-multi-dim.ipynb
    - A gaussian processor hyperparameter tuners based on the Upper Common Bound,
- Examples/turbo_example_solution_original.ipynb
- Examples/turbo_example_solution_with_initial_inputs.ipynb
    - example implementation of the turbo algorithms.

I chose to start here as this was working code that I could easily understand and so was able to get started right away.

## Code Modificaton
I have not kept a record of my week to week modifications as this was not made clear at the start of the project, this is mostly from memory.

I don’t like notebooks, I find them hard to debug and on the whole prefer writing command line tools and use a proper IDE instead. I worked on the initial code we were given and created a python module `capstone` where most of my code resides, and a range of command line tools that used that library.

### The main bits of code I wrapped up were...
- capstone.loadData
    - reads the sampled data of a given function and returns a trio pair of numpy arrays. Being the sample positions and the sampled values sorted by value, plus the submission order of the sorted point.
- capstone.makeGrid
    - creates an N dimensional grid of at most M points, whose positions are centred on a specific point and lie within a delta of that point on each axis.
- capstone.regressorUCB
    - runs a gaussian process regressor over a grid, return the UCB of all points on the grid.
- capstone.loneliestPoint
    - finds the point furthest from all other points within the N-dimensional convex hull of those points.  
- capstone.TurboState
    - class that manages the state of a Trust Region Baysian Optimisation hyper parameter tuner, including loading and saving states.
- capstone.turboSearch
    - runs a pass of the TuRBO algorithm.

### The command line programmes I wrote were...
- displayValue.py
    - for a specific function, prints out the samples locations and their values in submission or value order along with a few basic stats about the best two submissions
- lonely.py
    - for a specific function, computes the ‘loneliest’ point.
- random.py
    - for a specific function, computes a random position within the unit hypercube
- turbo.py
    - for a specific function, runs a pass of the turbo algorithm 
- ucb.py
    - for a specific function, run the ucb algorithm to choose a sample point. I pass in the index of the point to centre the algorithm on and the size of the volume around that point to search in. By default it samples at most 10,000 points.
- problem2.py
    - ucb parameter tuning for problem 2 trying to take into account noise.

### Notebook
I have one notebook, `multiDimDisplay.ipynb`. I use it inspect individual functions. It loads up a function’s samples and displays all the sample points by drawing a sufficient number of plots for all dimensions combinations. I used this to inspect the output of each run for each function. 

### Changes
#### Problem 3 : Dropping Dimensions
One of the first significant modifications I made was to add a `-d DIM` argument to all my command line tools, which would drop the specific dimension from all computations. I did this for problem 3, as the hint indicated that the function was independent of one of the input dimensions. Subsequently I ran all passes of my code for problem 3 dropping the 3rd dimension. This improved performance of my UCB search.

#### Seeding Samples
I ran several rounds of random samples at the start to obtain more points to seed the tuning. However I realised that these were not filling the hyper-volume at all evenly. So instead I wanted to pick the point in the unit hypercube that was furthest from all other points so far sampled. To do this I used a geometric approach, via the Delauny triangulation algorithm. 

https://en.wikipedia.org/wiki/Delaunay_triangulation

To find the optimum triangulation, this finds a set of circumcircles, each of whose circumference will pass through 3 points of the set, so forming a triangle. That circle will no other points from the set inside it. ***The centre of one of those circles will be the point furthest away from all other points in the original set***. The algorithm generalises to N dimensions. The Delauny library I used didn’t return the circum-hypersphere centres, so I used the Voronoi algorithm instead. This computes the dual of the Delauny triangulation, giving me the centre of those hyperspheres. Unfortunately the implementation I used fails for dimensions over 5, so I relied on random sampling for those values instead. This gave a better spread of seed points.

#### Function 2, Noise
The hint for problem 2 said that it was very noise. I read up on this and tried to deal with it by sampling the same point several times to gauge variance, and using a range of Gaussian Process kernels that are supposed to deal with noise. I unfortunately left the oversampling a bit late and couldn’t get the different kernels to work. I ended up falling back to the UCB tuner.

#### Function 4 : 
I basically got stuck in a local minima, I tried random sampling on lonely point sampling at the end and it really didn’t work.

#### TuRBO
I initially implemented TuRBO the week after Careleton show it to us and ran with it for several weeding on the functions over dimension 3. Unfortunately I managed to break my python install by inadvertently upgrading to numpy version 2.0, which isn’t compatible with the libraries used by my TuRBO implementation. I let too many weeks pass before fixing that properly and so only had a few TuRBO submissions.

#### Other Functions
I basically use the UBC regressor for these problems. Eyeballing the distribution of values and choosing appropriate search radius around the best performing value. 

I progressed the notebook I used to display results, also indicating the most recent point, best point, second best point, UCB suggestion, turbo and loneliest point suggestion to help with my search radius selection.

Once I implemented a version of UCB which would let me limit search to a region of the hypervolume, the only thing that seemed to be of any great benefit was adding a bit more information to my data display note book, allowing me to ‘eyeball’ the size of the region to sample from. This seemed to work best for the higher dimensional functions.

## Final Result
My approach was very ah-hoc through the competition, and I didn’t keep separate records of work for each problem. I’m surprised I did as well as I did.

My basic approach was to eyeball a specify a volume around the best point so far, on which I would perform a UCB computation. I’d occasionally hand tweak this when I saw that there were obvious gaps in the sample data. 

Nothing much changed during the final weeks of the competition. The higher dimensional values generally kept incrementally improving, and these were the ones I tweaked least. For function 4 I was obviously trapped in a local maxima and I tried more lonely/random samples to see if I could locate another point.

If there were more weeks available, for the noisy function (2) and the bumpy function (4) I would have explored more with more random/lonely samples. The noisy function was especially problematic. For the rest, it was progressing mostly slowly towards better results. I think I was too far down the ad-hoc rabbit hole to get myself out. 

I’d be much more structure for any ML/AI competition I might enter in the future. The main thing I learned was to look at research papers for better algorithms and search for implementations of those I might use.

If I were to do it all again, there are several things I would do...
- be much more structured, treating each problem individually,
- if I could seed the initial sample set, I’d use a Sobol sequence to selecting my initial random samples,
- I’d use the Optima algorithm to drive the whole process, which is an evolution of the TuRBO algorithm.
