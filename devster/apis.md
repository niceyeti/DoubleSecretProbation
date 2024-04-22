## Podman vs Docker

There is a question of whether podman or docker: podman is deamonless, docker daemon.
Goal: ensure that this can be abstracted under some api; hopefully the golang docker client api has some sort of daemonless bindings as well, such that I can write my code
to a domain-focused api and have it compile to whatever container backend exists.

Note:

* much of this ends up being an OCI-compliance narrative of api support, and figuring out which one supports the most while also providing the greatest stability/support.
