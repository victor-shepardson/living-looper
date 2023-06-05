# Living Looper

## NIME paper

I presented this project in a talk at NIME 2023. The full paper will be published soon, and a preprint is available [here](https://iil.is/pdf/2023_nime_shepardson_magnusson_living_looper.pdf)

## Models

A Living Looper instance includes a pretrained RAVE encoder-decoder and a living loop algorithm, both implemented with pytorch and exported as a TorchScript file. Plugins for music software define a real-time audio and GUI interface, but must load a `living-looper.ts` file which defines the sound and behavior. 

You can make your own `.ts` models using the RAVE fork (below) or download pre-trained ones from the [releases page](https://github.com/victor-shepardson/living-looper-juce/releases) (coming very soon!)

## Repository Structure

`RAVE/` - a fork of the official RAVE implementation with changes to support the Living Looper. The main implementation of the Living Looper is also here (`RAVE/export_looper.py`) as it needs to stay in sync with the RAVE modifications.

`living-looper-core/` - common components of C++ Living Looper plugins.

`living-looper-sc/` - SuperCollider Plugin

`living-looper-juce/` - JUCE plugin (WIP)

## License

This project incorporates `RAVE`, which uses the Creative Commons Attribution-NonCommercial 4.0 International license.

The plugin components including `living-looper-sc` and `living-looper-juce` are licensed under GPLv3.

As I understand it, the above affects what you can do with the source code, for example, publishing your own fork is allowed, but selling a VST based on it is not. You are free to do anything you want with the actual plugins and trained models, including making commercial art or music releases.
