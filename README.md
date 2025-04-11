# Jazz Music Generator Using Transformer Architecture
## Guantong Zhang, Achintya Srikanth, Aaditya Nair
A transformer model to generate jazz music (inspired by Wu et al, 2020).
This repository replicates the results of Wu, et al (https://arxiv.org/pdf/2008.01307). It combines the original repository with the metrics pipeline from the GitHub 'https://github.com/slSeanWU/MusDr/'.

The markdown 'train_infer_readme.md' contains instructions on training the Transformer-XL architecture and running the inference to generate jazz pieces.
The markdown 'metrics_readme.d' (/metrics/MusDr/) contains instructions on calculating the metrics for the generated pieces.

The repo also contains MIDI files, WAV files, REMI files and log files of our generated pieces. Model checkpoints have not been included in the repo. Batch jobs for scheduling the pipelines in SLURM are also included but not necessary.

There are two 'requirements.txt' files - one each for training/inference and metrics calculation respectively.
