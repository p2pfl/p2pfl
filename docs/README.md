![GitHub Logo](../other/logo.png)

# P2PFL - Documentation

The P2PFL documentation is created using [Sphinx](https://www.sphinx-doc.org/en/master/index.html). 


## 📥 Generating docs for new modules

To generate the documentation for new modules, you can run the following command:

```bash
rm -fr source/modules
sphinx-apidoc -M -e -f -o source/modules ../p2pfl
```

Then you can build the documentation using:

```bash
make html
```

## 📚 Documentation

The documentation is available at [https://pguijas.github.io/p2pfl](https://p2pfl.github.io/p2pfl/).
