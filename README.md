# Jet_ML_Tools

A collection of code that grew out of [1612.01551](https://arxiv.org/abs/1612.01551). Contains python code for creating jet images from an event record and running a convolutional neural net to do quark/gluon discrimination.

Python code should be run from the `python/` directory using python 3 (version 3.5 has been the most extensively tested). There are two examples, `jet_image_conv_example.py` and `image_generation_example.py`, which can be invoked as

```bash
python3 jet_image_conv_example.py
```

The `src/` directory contains event generation code. Take a look at the beginning of `Events.cc` for options. Requires Pythia8 and FastJet to be installed and pythis8-config and fastjet-config to be available in the `PATH` (for the makefile to work as is). A standard invocation after compiling would look like

```bash
for i in `seq 1 10`; do
  ./events -out gluon-event-seed$i.txt -pthatmin 160 -ptjetmin 200 -Zg -seed $i
done
```
