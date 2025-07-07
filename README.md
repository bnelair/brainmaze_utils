# BrainMaze: Brain Electrophysiology, Behavior and Dynamics Analysis Toolbox - Utils

This toolbox provides generic tools for the BrainMaze package. This tool was separated from the BrainMaze toolbox to support convenient and lightweight sharing of these tools across projects.

This project was originally developed as a part of the [BEhavioral STate Analysis Toolbox (BEST)](https://github.com/bnelair/best-toolbox) project. However, the development has transferred to the BrainMaze project.

## Documentation

Documentation is available [here](https://bnelair.github.io/brainmaze_utils).

## Installation

```bash
pip install brainmaze-utils
```

## How to contribute

The project has 2 main protected branches:
- `main` - contains official software releases
- `dev` - contains the latest feature implementations shared with developers

To implement a new feature, create a new branch from the `dev` branch with the naming pattern `developer_identifier/feature_name`.

After implementing the feature, create a pull request to merge the feature branch into the `dev` branch. Pull requests need to be reviewed by the code owners.

New releases are created by the code owners using pull requests from `dev` to `main` and by drafting a new release on GitHub.

### Documentation Guidelines

- New functions must include Sphinx-compatible docstrings
- Documentation is automatically generated from docstrings using Sphinx via `make_docs.sh`
- Documentation source is in `docs_src/`
- Generated documentation is in `docs/`
- `.doctrees` is not included in the repository

### Troubleshooting

If you encounter issues when updating the Sphinx-generated documentation (especially with many changes causing buffer hang-ups), using SSH over HTTPS is recommended. If you're using HTTPS, you can increase the buffer size with:

```bash
git config http.postBuffer 524288000
```

## License

This software is licensed under the BSD-3Clause license. For details, see the [LICENSE](https://github.com/bnelair/brainmaze_utils/blob/master/LICENSE) file in the root directory of this project.

## Acknowledgment

This code was developed and originally published by (Mivalt 2022, and Sladky 2022).
We appreciate you citing these papers when utilizing this toolbox in further research work.

- F. Mivalt et al., "Electrical brain stimulation and continuous behavioral state tracking in ambulatory humans," J. Neural Eng., vol. 19, no. 1, p. 016019, Feb. 2022, doi: [10.1088/1741-2552/ac4bfd](https://doi.org/10.1088/1741-2552/ac4bfd).
- V. Sladky et al., "Distributed brain co-processor for tracking spikes, seizures and behaviour during electrical brain stimulation," Brain Commun., vol. 4, no. 3, May 2022, doi: [10.1093/braincomms/fcac115](https://doi.org/10.1093/braincomms/fcac115).
