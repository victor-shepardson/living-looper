# Living Looper

A binary release which runs within [SuperCollider](https://supercollider.github.io/) on macOS (apple silicon) is available from the [Releases page](https://github.com/victor-shepardson/living-looper/releases). It also contains a model for electric guitar.

- Install SuperCollider
- Download and unzip the release
- Place the `LivingLooper` folder in the SuperCollider [extensions directory](https://doc.sccode.org/Guides/UsingExtensions.html)
- Open SuperCollider
- see the `LLGUI` helpfile

Alternatively:

- bring your own pretrained RAVE model
- clone this repo and create a Python environment
- run your RAVE model through `export.py` to produce a living looper `.ts` file
- build the SuperCollider plugin in `living-looper-sc` with `cmake`
- see `living-looper-sc/example/living-looper.scd`

## NIME paper

I presented this project in a talk at NIME 2023. The full paper will be published soon, and a preprint is available [here](https://iil.is/pdf/2023_nime_shepardson_magnusson_living_looper.pdf)

## Models

A Living Looper instance includes a pretrained RAVE encoder-decoder and a living loop algorithm, both implemented with pytorch and exported as a TorchScript file. Plugins for music software define a real-time audio and GUI interface, but must load a `.ts` file which defines the sound and behavior. 

You can make your own living looper `.ts` models from any [nn~](https://github.com/acids-ircam/nn_tilde) compatible streaming encoder/decoder model (such as [these](https://huggingface.co/Intelligent-Instruments-Lab/rave-models)) using the `export.py` script.

## Repository Structure

`export.py` - Python script which converts a RAVE model to a Living Looper model

`living-looper-core/` - common components of C++ Living Looper plugins.

`living-looper-sc/` - SuperCollider Plugin with GUI

`living-looper-juce/` - JUCE plugin (WIP, not functional at present)

## License

This project incorporates `RAVE`, which uses the Creative Commons Attribution-NonCommercial 4.0 International license.

The plugin components including `living-looper-sc` and `living-looper-juce` are licensed under GPLv3.

As I understand it, the above affects what you can do with the source code, for example, publishing your own fork is allowed, but selling a VST based on it is not. You are free to do anything you want with the actual plugins and trained models, including making commercial art or music releases.
