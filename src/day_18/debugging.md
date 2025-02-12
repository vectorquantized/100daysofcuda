## Day 18 conv2D debugging
Tried to look at te kernel in detail, wanted it to compute exactlly like `torch.nn.functional.conv2d`
* Tried the following:
    * single channel version
    * single batch version
    * single channel and single batch version, this tells me that the reference might be doing something else.
    * Perhaps as a next step I look into using the deterministc cudnn backend?
    * Then Looked into the indexing logic for how the `row_in` and `row_out` were calculated.
        * They were off by `filter_radius` and that seems to have fixed it.
        * Right now everything works for batch_size = 1.
        * Debugging continues.

