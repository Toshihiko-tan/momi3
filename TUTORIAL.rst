========
Tutorial
========

This tutorial demonstrates how to use the ``momi3`` package to perform inference on demographic models.  
We walk through simulating a population structure, running inference using the sample frequency spectrum (SFS), and interpreting the results.  

The corresponding Jupyter notebook for this tutorial is available at ``docs/tutorial.ipynb``.

Simulation
============

We begin by simulating genetic data under a simple demographic model using the ``msprime`` and ``demes`` packages.
For simplicity, we consider a scenario with two subpopulations (P0 and P1) that split from a common ancestor.
We assume all populations have constant effective population sizes of 10,000, and the subpopulations exchange migrants at a symmetric migration rate of 0.01.

To get started, import the necessary packages:

.. code-block:: python

    import msprime as msp
    import demes
    import demesdraw

Now, define the demographic model using msprime. Below we create a demographic model with three populations: an ancestral population (anc) and two derived populations (P0 and P1).
Assuming that the ancestral population has an initial size of 10,000, we can add the populations to the demographic model using the ``add_population()`` method.
Similarly, we can set the initial size of the two derived populations (P0 and P1) to 10,000 each.
Lastly, we set the symmetric migration rate between the two subpopulations using the ``set_symmetric_migration_rate()`` method.

.. code-block:: python

    demo = msp.Demography()
    demo.add_population(initial_size=1e4, name="anc")
    demo.add_population(initial_size=1e4, name="P0")
    demo.add_population(initial_size=1e4, name="P1")
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.01)

Then, we specify the split time to be 1000 generations between the subpopulations and the ancestral population:

.. code-block:: python

    tmp = [f"P{i}" for i in range(2)]
    demo.add_population_split(time=1000, derived=tmp, ancestral="anc")

We can visualize the demographic model we created using ``demesdraw``.

.. code-block:: python

    g = demo.to_demes()
    demesdraw.tubes(g)

Lastly, we can simulate the ancestry of a sample of 100 individuals from two subpopulations using msprime's ``sim_ancestry()`` function.  
Here, we will set the recombination rate to 1e-8 and the sequence length to 10 million base pairs.

.. code-block:: python
    
    sample_size = 100
    samples = {f"P{i}": sample_size for i in range(2)}
    anc = msp.sim_ancestry(
        samples=samples,
        demography=demo,
        recombination_rate=1e-8,
        sequence_length=1e7
    )
    ts = msp.sim_mutations(anc, rate=1e-8)

Inference using SFS-based methods
============

Now we use momi3 to perform demographic inference using the SFS generated from the simulated data.

**Note**: Inference with large sample sizes may be slow. Consider reducing the number of samples when running locally.

First, import the ``Momi3`` class from the ``momi3`` package.  
We can then construct the SFS inference object by calling ``Momi3().sfs()`` with the demographic model we created earlier.
Here, ``g`` is a ``demes``-formatted demographic model we have simulated previously.

.. code-block:: python

    from momi3 import Momi3
    momi_sfs_object = Momi3(g).sfs({'P0': 20, 'P1': 20})

Compute the allele frequency spectrum (AFS) from the simulated tree sequence, then convert it into a joint SFS (JSFS):

.. code-block:: python

    from momi3.jsfs import JSFS
    import jax
    afs = ts.allele_frequency_spectrum(
        sample_sets=[ts.samples([1]), ts.samples([2])],
        span_normalise=False
    )
    jsfs = JSFS.from_dense(afs, ["P0", "P1"])

We focus inference on a single parameter: the starting population size (``start_size``) of the first epoch of the first deme.
We use ``reparameterize()`` to map the constrained parameter space into an unconstrained one:

.. code-block:: python

    from momi3 import Momi3
    import numpy as np
    params = [("demes", 0, "epochs", 0, "start_size")]
    f, x = momi_sfs_object.reparameterize(list(params))
    parameters = list(x.keys())

We now evaluate the likelihood over a range of values for ``start_size`` to visualize how the likelihood varies with this parameter.
Here, we will sweep the parameter from 5000 to 20000 in increments of 100. You can adjust the range and step size as needed.

.. code-block:: python

    from jax import vmap
    import jax.numpy as jnp
    x_values = jnp.linspace(5000, 20000, 100)

    def compute_likelihood(val):
        updated_x = x.copy()
        updated_x[parameters[0]] = val
        params = updated_x
        return momi_sfs_object.loglik(params, jsfs)

    likelihoods = vmap(compute_likelihood)(x_values)

Lastly, we can plot the log-likelihood values against the start_size parameter values to visualize the results.

.. code-block:: python

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, likelihoods, label='Likelihood')
    plt.xlabel('x (parameter values)')
    plt.ylabel('Debugger Likelihood')
    plt.title('Debugger likelihood over parameters')
    plt.legend()
    plt.grid(True)
    plt.show()

This plot should reveal a peak in the log-likelihood curve around the true value of ``start_size = 10000``, validating the accuracy of the inference pipeline.