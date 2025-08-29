========
Tutorial
========

This tutorial demonstrates how to use the ``momi3`` module included in the ``demesinfer`` package to perform inference on demographic models. 

In momi3, the main approach for demographic inference is based on the site frequency spectrum (SFS) of genetic data.

The corresponding Jupyter notebook for this tutorial is available at ``docs/tutorial.ipynb``.

We walk through simulating a population structure, running inference using the sample spectrum (SFS), and interpreting the results.  

Simulation
============

We begin by simulating genetic data under a simple demographic model using the ``msprime`` and ``demes`` packages.

To get started, import the necessary packages:

.. code-block:: python
    import msprime as msp
    import demes
    import demesdraw

For simplicity, we consider a scenario with two subpopulations (P0 and P1) that split from a common ancestor.
We assume all populations have constant effective population sizes of 5000, and the subpopulations exchange migrants at a symmetric migration rate of 0.0001.

.. code-block:: python
    demo = msp.Demography()
    demo.add_population(initial_size = 5000, name = "anc")
    demo.add_population(initial_size = 5000, name = "P0")
    demo.add_population(initial_size = 5000, name = "P1")
    demo.set_symmetric_migration_rate(populations=("P0", "P1"), rate=0.0001)
    tmp = [f"P{i}" for i in range(2)]

Then, we specify the split time to be 1000 generations between the subpopulations and the ancestral population:

.. code-block:: python
    demo.add_population_split(time = 1000, derived=tmp, ancestral="anc")

We can visualize the demographic model we created using ``demesdraw``.

.. code-block:: python
    g = demo.to_demes()
    demesdraw.tubes(g)

.. image:: _images/demo.png
   :alt: Demographic model visualization
   :align: center

Then, we can simulate the ancestry of a sample of 100 individuals from two subpopulations using msprime's ``sim_ancestry()`` function.  
Here, we will set the recombination rate to 1e-8 and the sequence length to 10 million base pairs.

.. code-block:: python
    sample_size = 10
    samples = {f"P{i}": sample_size for i in range(2)}
    anc = msp.sim_ancestry(samples=samples, demography=demo, recombination_rate=1e-8, sequence_length=1e8, random_seed = 12)
    ts = msp.sim_mutations(anc, rate=1e-8, random_seed = 13)

Lastly, we can compute the allele frequency spectrum (AFS) from the simulated data.

.. code-block:: python
    afs_samples = {f"P{i}": sample_size*2 for i in range(2)}
    afs = ts.allele_frequency_spectrum(sample_sets=[ts.samples([1]), ts.samples([2])], span_normalise=False)

Inference using SFS-based methods in momi3
============
Now we create an example of using momi3 to perform demographic inference using the SFS generated from the simulated data.

We will be inferencing the population sizes, split times, and migration rates.

**Note**: Inference with large sample sizes may be slow. Consider reducing the number of samples when running locally.

To visually inspect the how the likelihood changes and whether the method is reliable, we create a function to plot the results.

.. code-block:: python
    from jax import vmap, lax

    def plot_sfs_likelihood(demo, paths, vec_values, afs, afs_samples, theta=None, sequence_length=None):
        import matplotlib.pyplot as plt

        path_order: List[Var] = list(paths)
        esfs = ExpectedSFS(demo, num_samples=afs_samples)

        def sfs_loglik(afs, esfs, sequence_length, theta):
            afs = afs.flatten()[1:-1]
            esfs = esfs.flatten()[1:-1]
            
            if theta:
                assert(sequence_length)
                tmp = esfs * sequence_length * theta
                return jnp.sum(-tmp + xlogy(afs, tmp))
            else:
                return jnp.sum(xlogy(afs, esfs/esfs.sum()))
        
        def evaluate_at_vec(vec):
            vec_array = jnp.atleast_1d(vec)
            params = _vec_to_dict_jax(vec_array, path_order)
            e1 = esfs(params)
            return -sfs_loglik(afs, e1, sequence_length, theta)

        results = lax.map(evaluate_at_vec, vec_values)

        plt.figure(figsize=(10, 6))
        plt.plot(vec_values, results, 'r-', linewidth=2)
        plt.xlabel("vec value")
        plt.ylabel("Negative Log-Likelihood")
        plt.title("SFS Likelihood Landscape")
        plt.grid(True)
        plt.show()

        return results

We first try to inference the population size of ancestral population "anc".

With initial guess of 4000, we search for the maximum likelihood estimate over a grid of values ranging from 4000 to 6000.

.. code-block:: python
    import jax.numpy as jnp
    paths = {
        frozenset({('demes', 0, 'epochs', 0, 'end_size'),
                ('demes', 0, 'epochs', 0, 'start_size')}): 4000.,
    }
    vec_values = jnp.linspace(4000, 6000, 50)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: /images/pop_size.png
   :alt: Demographic model visualization
   :align: center

Negative Log-Likelihood is optimized at around 4600, which is pretty close to the true value of 5000.

Then, we try to inference the population size of one of the descendant populations "P0".

Again, with initial guess of 4000, we search for the maximum likelihood estimate over a grid of values ranging from 4000 to 6000.

.. code-block:: python
    import jax.numpy as jnp
    paths = {
        frozenset({('demes', 1, 'epochs', 0, 'end_size'),
                ('demes', 1, 'epochs', 0, 'start_size')}): 4000.,
    }
    vec_values = jnp.linspace(4000, 6000, 50)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: /images/pop_size2.png
   :alt: Demographic model visualization2
   :align: center

Negative Log-Likelihood is optimized at around 5500, which, again, is pretty close to the true value of 5000.

Then, we try to inference the split time between the ancestral population and the two descendant populations.

.. code-block:: python
    import jax.numpy as jnp
    paths = {
        frozenset({('demes', 0, 'epochs', 0, 'end_time'),
                ('demes', 1, 'start_time'),
                ('demes', 2, 'start_time'),
                ('migrations', 0, 'start_time'),
                ('migrations', 1, 'start_time')}): 4000.,
    }
    vec_values = jnp.linspace(500, 1500, 50)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: images/split_time.png
   :alt: Split Time
   :align: center

Negative Log-Likelihood is optimized at around 1000. Our population split is successfully inferred!

Finally, we try to inference the migration rate between the two descendant populations.

.. code-block:: python
    import jax.numpy as jnp
    paths = {
        ('migrations', 0, 'rate'): 4000.,
    }

    vec_values = jnp.linspace(0.00005, 0.0002, 10)
    result = plot_sfs_likelihood(g, paths, vec_values, afs, afs_samples)

.. image:: /images/migration_rate.png
   :alt: Migration Rate
   :align: center

Negative Log-Likelihood is optimized at around 0.00013, which, again, is pretty close to the true value of 0.0001.