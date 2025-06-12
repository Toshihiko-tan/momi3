
============
Tutorial
============

We will be using the `momi3` package to perform inference on demographic models. 
The following sections will guide you through the process of simulating a population structure, running inference, and interpreting results.
You can run the ipython notebook that created this tutorial at docs/tutorial.ipynb.

Simulation
============
For the sake of this tutorial, we will use a simple model with two sub populations.
We will be using 'msprime' to simulate the population structure and 'demes' to define the demographic model.
To get started, import msprime, demes, and demesdraw:
.. code-block:: python
    import msprime as msp
    import demes
    import demesdraw

Here, we will define a simple demographic model with two subpopulations that split from a common ancestor.
For simplicity, we will use a constant population size of 10000 for both subpopulations and the ancestral population, and a symmetric migration rate of 0.01.
Use demography() to create a demographic model, add_population() to add the populations, and set_symmetric_migration_rate to set symmetric migration rates among a set of populations.
.. code-block:: python
    demo = msp.Demography()
    demo.add_population(initial_size = 1e4, name = "anc")
    demo.add_population(initial_size = 1e4, name = "P0")
    demo.add_population(initial_size = 1e4, name = "P1")
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.01)

We further define the time of the split between the two subpopulations and the ancestral population to be at 1000 generations ago.
.. code-block:: python
    tmp = [f"P{i}" for i in range(2)]
    demo.add_population_split(time = 1000, derived=tmp, ancestral="anc")

We can visualize the demographic model using demesdraw.
.. code-block:: python
    g = demo.to_demes()
    demesdraw.tubes(g)

Lastly, we can simulate the ancestry of a sample of 100 individuals from each subpopulation using msprime's sim_ancestry() function.
Here, we will set the recombination rate to 1e-8 and the sequence length to 10 million base pairs.
.. code-block:: python
    sample_size = 100
    samples = {f"P{i}": sample_size for i in range(2)}
    anc = msp.sim_ancestry(samples=samples, demography=demo, recombination_rate=1e-8, sequence_length=1e7)
    ts = msp.sim_mutations(anc, rate=1e-8)

Inference using SFS based methods
============
In this section, we will use the simulated data to perform inference on the demographic model using the SFS-based methods provided by `momi3`.
Please note that if you are running this tutorial locally, 100 samples from each subpopulation will take a long time to run.
You can reduce the sample size to speed up the inference process.

To start, import the Momi3 class from the `momi3` package.
We can then create an instance of the Momi3 class using the simulated tree sequence, here, g is a demes formatted demographic model we have simulated previously.
.. code-block:: python
    from momi3.momi import Momi3
    momi_sfs_object = Momi3(g).sfs({'P0':20, 'P1':20})

.. code-block:: python
    from momi3.jsfs import JSFS
    import jax
    afs = ts.allele_frequency_spectrum(sample_sets=[ts.samples([1]), ts.samples([2])], span_normalise=False)
    jsfs = JSFS.from_dense(afs, ["P0", "P1"])
