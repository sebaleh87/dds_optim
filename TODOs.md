Ideas before vaccation
does mode covering property happen because time steps with small beta play such a small role?
finer integration at steps where beta is small, or use importance sampling?

### TODO
- learn sigma
- add gradients to score?
- add clipping hacks?
- implement log derivative
- mix diff sampler with adam?

### Thought
- own sigma for each direction?
- SDE Entropy can only be increased by increasing sigma in prior?
- TODO check this by computing the Entropy during Annealing?


### IDEA
### input dependent cosine embeddings
input ---> batch norm ---> cosine embedding
input will be normalized --> frequency range can be chosen a priori for example 4 sigma