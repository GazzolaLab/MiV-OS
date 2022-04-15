# Documentation

We use [`Sphinx`](https://www.sphinx-doc.org/en/master/) and [`ReadtheDocs`](https://readthedocs.org/) to organize our [documentation page]().

In addition, we utilize the following extensions to enhance the documentation :coffee:
- `numpydoc`: We favor [numpy documentation style](https://numpydoc.readthedocs.io/en/latest/format.html) for API documentation.
- `myst_parser`: We like to write documentation and guidelines in `markdown` format.

## Key things to remember

- Please DON'T IGNORE any `errors` or `warnings` during the compilation stage.
- If you are uploading any `ipynb` or notebook-style, make sure you excluded the input-output cells.
    - Currently, we don't use the `runnable` notebook directly on the documentation. If you think otherwise, please leave an issue so that we can discuss.

## Build documentation

The `sphinx` is already initialized in `docs` directory. In order to build the documentation, you will need additional packages listed in `docs/requirements.txt`.

```bash
pip install sphinx sphinx_rtd_theme myst-parser numpydoc
cd docs
make clean
make html
```

Once the documentation building is done, open `docs/_build/html/index.html` to view.

Use `make help` for other options.

## User Guide

User guidelines and tutorials are written in `.rst` or `.md` format.
These files will be managed in `docs` directory.

## API documentation

The docstring for function or modules are automatically parsed using `sphinx`+`numpydoc`.
Any inline function description, such as

```py
""" This is the form of a docstring.

... description

Attributes
----------
x : type
    x description
y : type
    y description

"""
```

will be parsed and displayed in API documentation. See `numpydoc` for more details.

## ReadtheDocs

`ReadtheDocs` runs `sphinx` internally and maintain the documentation website. We will always activate the `stable` and `latest` version, and few past-documentations will also be available for the support.

@nmnaughton and @skim449 has access to the `ReadtheDocs` account.
