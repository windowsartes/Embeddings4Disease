# Embeddings4Disease

## Description

TBA

## Table of Contents

- [Embeddings4Disease](#embeddings4disease)
  - [Description](#description)
  - [Table of Contents](#table-of-contents)
  - [How to install](#how-to-install)
  - [Usage](#usage)
    - [CLI for training](#cli-for-training)
    - [CLI for validation](#cli-for-validation)
    - [CLI fro preprocessing](#cli-fro-preprocessing)
  - [Additional examples](#additional-examples)
  - [Tests](#tests)

## How to install

To install a package from this repository, just write in the terminal:

```bash
pip install git+https://github.com/windowsartesEmbeddings4Disease.git
```

In the case the master branch doen't have any features, you can install this package using any other branch:

```bash
pip install git+https://github.com/windowsartes/Embeddings4Disease.git@branch
```

Where «branch» is a the name of the branch you need.

For example, to install the package version from the development branch, you need to execute:

```bash
pip install git+https://github.com/windowsartes/Embeddings4Disease.git@development
```

Also there are some optional dependencies in out package. For example, there are optional dependencies for Weights and Biases logging and for RoFormer architecture usage.
To install some optional dependencies, execute this command:

```bash
pip install "Embeddings4Disease[optional dependencies] @ git+https://github.com/windowsartes/Embeddings4Disease.git
```

Instead of «optional dependencies» you just need to list dependencies names separated by commas. In the [pyptoject](./pyproject.toml) you can find all the optional dependencies.

For example, to allow wandb intageration and get the opportunity to use RoFormer architecture, just execute:

```bash
pip install "Embeddings4Disease[wandb,roformer] @ git+https://github.com/windowsartes/Embeddings4Disease.git
```

Once you install the module in your virtual environment, you can use it like any other installed package and also use the CLI.

## Usage

Our project has a CLI within which utilities for training and validating models are implemented.
Scripts will be generated and added automatically after installing the packages in the virtual environment. The CLI works entirely based on the config you give it. You can find examples of various configs [there](./config_examples/).

### CLI for training
To train a backbone model, execute this command:

```bash
training *path-to-config*
```

Path to the config must be relative to your current working directory or it must be absolute. Also all the pathes inside this config must be relative to your current working directory or absolute. Config examples and their full descriptions you can find [there](./config_examples/train/).

### CLI for validation

Also there is a CLI for the model validation:

```bash
validation *path-to-config*
```

As during training, the path to the config file must be specified relative to the current directory or it must be an absolute path. Examples and descriptions you can find [there](./config_examples/validate/).

### CLI fro preprocessing

The training and validation CLIs expect the data to be in the specific format: one transaction on one line. To bring the data into this format and also be able to create a tokenizer using a dictionary file, we have a CLI to pre-process the data.

```bash
preprocessing *path-to-config*
```

Details for creating the config, as well as an example for preprocessing MIMIC-4, you can find [there](./config_examples/preprocess/).

## Additional examples

In [this colab-notebook](https://colab.research.google.com/drive/1xc87kcnBKP5s_thYbfkAgiiIgiNhzBY7?usp=sharing)
you can find an example of CLI usage for the model pre-training.

In [this colab-notebook](https://colab.research.google.com/drive/1UPCCeCHfk88UQ6eW-i-VtZ5sH2jmL9Jc?usp=sharing) you can find an example of CLI usage for the model validation.

## Tests

CLI was fully tested on Windows 11, Ubuntu 22.04 via WSL, Ubuntu 20.04 and inside google colab.