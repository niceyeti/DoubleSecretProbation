## Description

A core subtask is figuring out how to implement these:

* stdin/stdout connections between containers
* others streams: file, network, grpc

I like the idea of starting with merely stdout->stdin pipe semantics, since it will help nail own the layer-one requirements and language
under which other stream models could be abstracted.

## Redirection Stdout-Stdin Pipes

Stdin of container:

* golang docker client api's appear to support stdin/stdout redirection using "attach" semantics, although some level of indirection (via some intermediate file) may be required.

## Design

Can streams (stdin/out redirection, network, and otherwise) be unified under a single model?
