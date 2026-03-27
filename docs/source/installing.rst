Installing mf6adj
=================

Install from PyPI
-----------------

Install the latest release with pip:

.. code-block:: bash

   pip install mf6adj

Install from conda-forge
------------------------

Install the latest release into a conda environment:

.. code-block:: bash

   conda install -c conda-forge mf6adj

Install from GitHub
-------------------

Install the latest development version directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/INTERA-Inc/mf6adj.git

Install a specific tag, branch, or commit by appending a ref:

.. code-block:: bash

   pip install git+https://github.com/INTERA-Inc/mf6adj.git@<ref>

For example, install from a tag:

.. code-block:: bash

   pip install git+https://github.com/INTERA-Inc/mf6adj.git@v1.0.0

Developer install from a local clone
-------------------------------------

Clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/INTERA-Inc/mf6adj.git
   cd mf6adj
   pip install -e .

To include developer dependencies:

.. code-block:: bash

   pip install -e '.[dev]'
