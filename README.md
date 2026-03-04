##  JQUERY-JAX

### jQuery's $(selector).hide().fadeIn().css(...) — same philosophy, but for tensor math.
-Author: J. Roberto Jiménez C.  


It's a fluent, chainable, lazy wrapper over JAX. You build a pipeline of operations, nothing executes until you explicitly ask for it. 
Execution Modes

.value_of() — eager, run now, get a JAX array back
.jit() — compile the whole pipeline with XLA, then run
.vmap() — vectorise the pipeline over a batch dimension automatically
.pmap() — split work across multiple devices/GPUs

Autograd

.grad() — differentiate the entire pipeline, returns a JXTensor
.value_and_grad() — both in one pass (efficient)
.jacrev() / .jacfwd() — full Jacobian, reverse or forward mode
.hessian() — second-order derivatives

Tensor Ops (all lazy)
Activations: relu, sigmoid, tanh, gelu, softmax, log_softmax, log_sigmoid
Reductions: sum, mean, var, std, max, min, argmax, argmin
Shape: reshape, transpose, squeeze, unsqueeze, __getitem__
Math: abs, sqrt, exp, log, sin, cos
Linear algebra: matmul, dot, einsum
All arithmetic operators: +, -, *, /, **, @, and their reverse forms
Neural Network Primitives

Linear — dense layer with Xavier init, optional bias
Sequential — stack layers, collect parameters
normalize — zero-mean unit-variance, inline in the pipeline

Optimizers

sgd, adam, adamw — all wrapping optax
.step(loss_fn) — single gradient step via value_and_grad (one pass)
.train(loss_fn, steps, callback) — full loop, returns loss history

Loss Functions
mse, cross_entropy, binary_cross_entropy, l1_loss
Debugging

.debug_print(msg) — prints shape/dtype/mean mid-pipeline, passthrough
.inspect() — returns the full graph as a dict: transforms, depth, hash, devices
