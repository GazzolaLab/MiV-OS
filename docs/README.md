# Documentation

We use [`Sphinx`](https://www.sphinx-doc.org/en/master/) and [`ReadtheDocs`](https://readthedocs.org/) to organize our [documentation page](https://miv-os.readthedocs.io/en/latest/).
Most up-to-date documentation is typically collected in separate branch `doc-patch`, which can be seen [here](https://miv-os.readthedocs.io/en/doc_patch/).

We utilize the following extensions to enhance the documentation :coffee:
- `numpydoc`: We favor [numpy documentation style](https://numpydoc.readthedocs.io/en/latest/format.html) for API documentation.
- `myst_parser`: We like to write documentation and guidelines in `markdown` format.

## Key things to remember

- Please DON'T IGNORE any `errors` or `warnings` during the compilation stage.
- If you have any `.ipynb` format notebook, please change it to `.md` format using `jupytext <notebook>.ipynb --to myst`. (You can install `jupytext` with pip.)
    - Make sure you excluded the input-output cells.
    - Currently, we don't use the `runnable` notebook directly on the documentation. If you think otherwise, please leave an issue so that we can discuss.
- If you would like to include images (i.e. result, plot, or visualization), upload the image on cloud (Box or Google Drive) and add the URL link.

## Build documentation

The `sphinx` is already initialized in `docs` directory. In order to build the documentation, you will need additional package listed in extra dependencies.

```bash
poetry install -E docs
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
