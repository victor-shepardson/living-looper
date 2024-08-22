# Living Looper

## SuperCollider Quark

A version which runs within [SuperCollider](https://supercollider.github.io/) can be installed like so:

- Install SuperCollider
- open SuperCollider
    - on Windows, you may need to run as administrator the first time

### Automatic install

- In SuperCollider, run the line `Quarks.install("https://github.com/victor-shepardson/living-looper-sc")` 
    - (type it in a press shift+return)
- then run the line `LivingLooper.new`
    - a LivingLooper window should appear
- Select your audio devices and press the "start audio" button
    - this will prompt you to install `NN.ar` if not already installed
- See the `LivingLooper` helpfile to get started

### Manual install

- Locate the SuperCollider extension folder from the SuperCollider menu: `File -> open user support directory -> Extensions`
- Install SuperCollider plugins by placing them in this folder:
    - NN.ar: https://github.com/elgiano/nn.ar/releases/tag/v0.0.6-updated
        - download the zip file for your platform
    - Living Looper: https://github.com/victor-shepardson/living-looper-sc/releases/tag/v1.1.1
        - download the "Source Code" zip file
- In the SuperCollider menu select `Language -> reboot interpreter`
- then run the line `LivingLooper.new`

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

This project incorporates `RAVE`, which uses the Creative Commons Attribution-NonCommercial 4.0 International license.

The plugin components including `living-looper-sc` and `living-looper-juce` are licensed under GPLv3.

As I understand it, the above affects what you can do with the source code, for example, publishing your own fork is allowed, but selling a VST based on it is not. You are free to do anything you want with the actual plugins and trained models, including making commercial art installations or music releases.
