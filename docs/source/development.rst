Development Guide
==================

This is a guide to develop this package.

Install Pipenv
~~~~~~~~~~~~~~
If you don't have `pipenv` installed, you can install it using `pip`.

.. code-block:: bash

  pip install pipenv


Linting
~~~~~~~
We use `ruff` for linting the code. To run the linter, use the following command.

.. code-block:: bash

  pipenv run lint


Auto format
~~~~~~~~~~~

We use `ruff` for formatting the code. To run the formatter, use the following command.

.. code-block:: bash

  pipenv run format


Unit test
~~~~~~~~~

We use `unittest` for testing the code. To run the unit tests, use the following command:

.. code-block:: bash
  
  pipenv run unittest

Contribution
~~~~~~~~~~~~

Once you complete your changes, please create a PR to the main branch of https://github.com/CyberAgentAILab/python-dte-adjustment.