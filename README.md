# PV-RNN

The simulation experiments of flexibility task using PV-RNN model.

## How to run.

```
python main.py
```

If you use job scheduler, such as PBS, use below:

```
qsub ./pbs_script.sh
```

## Hyper parameter setting

You can change learning and test conditions  by editting `config.py`. 

Don't forget to change the output directory. The setting of output directry is in `out_dir_name` in `config.py`.

## Results

```
rslt_directry
|- log.txt
|- mainrslt : Losses were saved.
|- saves : Hyper parameters and learned model parameter were saved.
|- data_preprecessed : preprocessed data (namely, softmax transformation and sequence padding) was saved.
'- epoch_size
   |- img
   |- target_generation : target generation using adaptive variables at only initial step.
   |- free_generation : closed-loop generation.
   |- error_regression : predictions using error regression.
   |- posterior_generation : target generation using adaptive variables at all steps.
   '- latent_space_traversal : generation using latent space traversal.
```
