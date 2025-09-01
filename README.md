# UE_KL_Replication

This project is a replication of a study on Uncertainty Estimation using Kullback-Leibler (KL) Divergence in machine learning models.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Project Structure

The project is organized as follows:

```
UE_KL_Replication/
├── backend/
│   ├── data/
│   │   └── sample_data.csv
│   ├── metrics/
│   │   └── kl_divergence.py
│   └── models/
│       ├── model.py
│       └── train.py
├── configs/
│   ├── base_config.yaml
│   └── config_loader.py
├── experiments/
│   └── run_trial.py
├── notebooks/
│   └── data_exploration.ipynb
├── results/
│   └── trial_1_results.json
├── scripts/
│   ├── plot_results.py
│   └── run_all_experiments.sh
├── main.py
├── requirements.txt
└── README.md
```

- **backend/**: Contains the core logic of the project.
  - **data/**: Holds sample data for the models.
  - **metrics/**: Includes implementations of metrics like KL Divergence.
  - **models/**: Contains the model architecture, training, and evaluation scripts.
- **configs/**: Houses configuration files for experiments.
- **experiments/**: Contains scripts to run experiment trials.
- **notebooks/**: Jupyter notebooks for data exploration and analysis.
- **results/**: Stores the results from experiment trials.
- **scripts/**: Utility scripts for plotting results and running all experiments.
- **main.py**: The main entry point to run the project.
- **requirements.txt**: A list of dependencies for the project.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/anisety/AlgoverseResearch.git
    cd UE_KL_Replication
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run a single trial, you can execute the `main.py` script:

```bash
python main.py
```

This will run the default trial as specified in the `main.py` script, which trains the model and saves the results.

To run all experiments, you can use the shell script provided:

```bash
bash scripts/run_all_experiments.sh
```

## Configuration

The experiments can be configured by modifying the `.yaml` files in the `configs/` directory. The `config_loader.py` script is used to load these configurations for the experiment trials.

`base_config.yaml` contains the default parameters for the models and training process. You can create new config files for different experiments.

## Dependencies

The project's dependencies are listed in the `requirements.txt` file. The main dependencies are:

-   [PyTorch](https://pytorch.org/)
-   [NumPy](https://numpy.org/)
-   [Pandas](https://pandas.pydata.org/)

## Contributing

Contributions are welcome! Please feel free to submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## License

This project is licensed under the MIT License.
