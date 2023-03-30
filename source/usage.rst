Usage
=====

Installation
------------

To use Napoleon, first install it using pip:

.. code-block:: console

   (.venv) $ pip install napoleonai

In code usage
-------

To use Napoleon inside your Python project:
    - import the library
    - run a script with a configuration argument

.. code-block:: python3

    from napoleonai.cli.cli import run
    run()

.. code-block:: console

    (.venv) $ python SCRIPT_FILE.py configuration=PATH_TO_YAML_CONFIGURATION.yaml

