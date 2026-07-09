# slab

Calculates the fields in step-index slab waveguides with arbitrary numbers of layers.

## Synopsis

Below we see an input field, propagating in a vacuum, arriving at the endface of a 3 layer slab waveguide:

<!-- <div style="text-align: center;">
<img src="./images/3d_input_field_dark.png" alt="drawing"/>
</div> -->

<div style="text-align: center;">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/3d_input_field_dark.png"/>
  <img alt="desc" src="./images/3d_input_field_solarize.png"/>
</picture>
</div>

The slab package calculates the resultant field inside the guide:

<div style="text-align: center;">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./images/3d_resultant_field_dark.png"/>
  <img alt="desc" src="./images/3d_resultant_field_solarize.png"/>
</picture>
</div>

It achieves this by calculating the guided and radiation modes of the structure, determining the guided mode weights, and integrating the radiation modes over the positive real numbers to find the radiation field.

<!-- The relevant refractive index and input field information can be provided more simply in the following plot:

<img src="./images/basic_slab_RIP_and_input_dark.png" alt="drawing"/> -->

