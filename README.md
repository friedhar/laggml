# A GGML Derived Tensor Libra
Laggml is a GGML derived tensor library.

## GGML Architecture
### Context 
A "container" which holds computational graphs, tensors, and in some cases data.
### Graph
A computation graph. eg. a scaled dot product attention would look something like this:
![](public/sdp_attention_graph_0.png)
### Backend
Underlying accelerator device - CUDA, TPUs, CPUs, TT, MPS, AMD GPUs etc..
