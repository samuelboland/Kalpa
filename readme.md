# README

# Introduction

## Reason
The purpose of this project is to develop a deep learning neural network
to automatically generate normal maps from a given texture file. 

This process is not easily done by normal tools. Normal maps encode information
that is not present in the underlying texture; specifically, information
on the height of the texture and the response to light from various angles.

Normal maps are generally either handmade or are produced from a more
complicated 3d-mesh, and then "baked in" to texture files. 

This project aims to train a network on normal maps that were handmade, 
with the hope that it can learn how to generate plausible normal maps for
novel input data. 

## Name
The network is intended to be used to generate normal maps for textures in the 
video game The Elder Scrolls III: Morrowind. Specifically, the open-source
reimplementation and extension of the original 2002 Morrowind Engine: OpenMW. 

The lore of this universe copies the buddhist idea of a "Kalpa," a length of time
roughly equivalent to "A ton." Think eons of eons. In the Elder Scrolls, the world
is reborn at the end of each Kalpa, but retains aspects of the previous one. 

This felt like it was fitting for a cyclical deep learning neural network. 
