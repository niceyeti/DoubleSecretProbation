# Devster

## Description

This is just a preliminary prototype to help clarify reqs.
They say don't make throwaway analysis prototypes, "they" being non-developer shmarchitects.
"They" are wrong.

Most of this will be cryptic notes and design... DSL development.

## DSL Specification Proposals

### Golang infra, Python app layer

Motivation: Python has extraordinary data science and analytics workflows, while
Golang's system language properties make it far more suitable for distributed orchestration.
So, split these into two components of the language. The intuition from Dagster-evaluation
is that there are formally two languages that most workflow tools mistakenly conflate:

1) graphical DSL: coordination, graphical structure, system guarantees
2) application: pieces of application code, for example ML algorithms or ETL jobs

(1) could be the svelte systems layer, possibly with plugins for multiple targets (k8s objets or local docker compose stacks)
or external services (lambda, queues, secrets, etc).
(2) would be the application layer: data science, business reqs/rules, etc. Python or arbitrary script languages

(1):

* observables (inputs)
* graphical topology
* error layers and other orthogonals
* resiliency
* file systems, network components, services, signals
* secrets
* interesting: direct git action integration. Note per the below, (1) components
  act more like inputs, and (2) components like outputs; hence input git actions
  (repo changes, hooks, etc.) could be in (1), where output actions (commits,
  etc) are (2).
* nodes and edges: edges convey binary messages and golang-style context
* (1) must support all the requirements of possible (2) usecases: manually
  triggered processes, automatically triggered, jobs vs daemon style tasks,
  on-start/on-exit features, unit-testing on save, or possibly conditional
  execution (run unit tests only but not full deployment, on code change;
  basically, staged development and execution).

(2):

* app layer stuff
* credentials
* build actions
* business rules

First round:

* define nodes, edges, process/container tree in golang
  * Node
  * Edge
  * Observable: file (fsnotify)
* run arbitrary chunks of python and bash
  * These are completely orthogonal to (1), hence just opaque operations. Code
      chunks are like outputs, but decoupled. They could produce diagrams, write
      files, or even write files which then trigger components of (1).

Tilt-like DSL for 1+2:

```python
    
    # A few class-1 components
    # Possibly redundant: the goal is to replace Make with devster. Implement Makefile object last; seems more like legacy integration.
    make_project_a = makefile("./project_A/Makefile", default_recipe="build")
    make_project_b = makefile("./project_B/Makefile", default_recipe="build")
    # Future: just riffing, not needed now or known how secrets would be integration. But they are layer-1 for sure,
    # using an injection pattern. 
    build_secret = secret("project-c-build-secret", script="./build/get_token.sh")
    scan = task("dangerous-dast-scan", script="./build/kubesec.sh", trigger="manual") # no idea how these would trigger, but they are useful for scans, etc.
    project_c = docker_build(
        'project-c-image',
        '.',
        dockerfile="./project_C/Dockerfile",
        only=["./project_C/src"],
        secrets=[build_secret])
    # Layer 2 stuff
    arbitrary_task = task("image-scanner", build=project_c, script="./project-C/build/some_script.sh")
    # TODO: do these declare their outputs? Ie, for use as inputs to subsequent tasks, visualization, etc.
    bronze_etl = task("bronze-transformer", build)
    silver_etl = task("silver-transformer", some_build)
    visualization = task("visualize_thing", image="python:3.7")
```

Based on the above, I think we can see that the DSL may not need any python. All
builds, languages, package managers can be defined in dockerfiles or k8s yaml.

Relations divulged:

* depends_on: declare job dependencies.
* jobs vs. daemons: on-start, on-exit logic?
* inputs? outputs? how are these declared?
* entry credentials when executing?
* "trigger='manual'" parameter could instead just be the "inputs"
* image_build, secrets, and task seem to be the root objects
  * all languages encapsulated under images, dockerfiles
  * build environments (secrets, etc) provided by just some utility objects like a 'secret'
  * outputs are executable
* inputs and outputs surely exist, but I'm uncertain how they are characterized, where they live
* post-condition logic? wasn't this a primary deficiency/oversight of make?
* alternative and perhaps confounding objects: these may actually be the objects that in typical
  workflow fashion, conflate a generic workflow model with a specific implementation
  * git, test, secret
    * can each of these fit within a generic task/build model, ie layer (1) objects?

Evaluation:

* compare with pipelines
* A DSL instantiation (pipeline program) should be committable, versionable.
  * reqs are equivalent to continuous deployment
* need to incorporate testing as a first class member
