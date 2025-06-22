# Living Looper

This repository is the Python CLI for the Living Looper, which can be used to export [nn~](https://github.com/acids-ircam/nn_tilde) compatible Living Looper files.

For the instrument itself, see [living-looper-sc](https://github.com/victor-shepardson/living-looper-sc)

## Python scripts

Using `export.py`, you can create your own Living Looper models from an `nn~` compatible (causal) encoder-decoder model, e.g. RAVE.

- bring your own pretrained RAVE model
- clone this repo and create a Python environment with `nn_tilde` and `pytorch`
- run your RAVE model through `export.py` to produce a living looper `.ts` file

## NIME paper

This project was presented at NIME 2023. The full paper is available [here](https://www.nime.org/proc/nime2023_32/index.html)

## Models

A Living Looper instance includes a pretrained RAVE encoder-decoder and a living loop algorithm, both implemented with pytorch and exported as a TorchScript file. Plugins for music software define a real-time audio and GUI interface, but must load a `.ts` file which defines the sound and behavior. 

You can make your own living looper `.ts` models from any [nn~](https://github.com/acids-ircam/nn_tilde) compatible streaming encoder/decoder model (such as [these](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)) using the `export.py` script.

## Repository Structure

`export.py` - Python script which converts a RAVE model to a Living Looper model

`living-looper-sc/` - SuperCollider Quark with GUI

`living-looper-core/` - common components of C++ Living Looper plugins.

`living-looper-juce/` - JUCE plugin (WIP, not functional at present)

## License

Some Living Looper models incorporate `RAVE`, which uses the Creative Commons Attribution-NonCommercial 4.0 International license.

The plugin components including `living-looper-sc` are licensed under GPLv3.

As I understand it, the above affects what you can do with the source code, for example, publishing your own fork is allowed, but selling a VST based on it is not. You are free to do anything you want with the actual plugins and trained models, including making commercial art installations or music releases.
