# ASDF

Welcome to the repository of the **A**daptable **S**eismic **D**ata **F**ormat.
Please refer to the documentation for more details.

[Documentation](http://asdf.readthedocs.org)

### Building the Documentation

The documentation is created with sphinx and the read the docs theme. Install
both with pip. It furthermore requires `prov` and `pydot` to be present.

```bash
$ pip install sphinx
$ pip install sphinx_rtd_theme
$ pip install prov
$ pip install pydot
```

Then change to the `doc` directory and run

```bash
$ make html
```

to create the documentation in HTML. A number of other formats are also
available.
