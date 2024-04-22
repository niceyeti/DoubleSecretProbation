First stab at implementing L1 requirements. See root-level readme for
raw notes of layer one reqs. This is a follow-up attempt to implement
a small layer one prototype to gather more requirements and develop
interfaces. This is an attempt to develop types.

One-sentence layer-one description: controllable and communicable processes,
a pure system decription with no application-level logic, only the capability
of supporting it.

Rules:

* stay out of the weeds: the intent is a simple mesh of containers with pluggable
  inputs/outputs and orchestration (start/restart/shutdown logic)

## Pesudocode Notepad

Note this is kind of just like graphical docker-compose with granular and pluggable
properties, like stdout/stderr.

```go

// A kitchen sink example

// Input classes: things you can listen to. These are also like "edges".
signal = Signal()
file = File()
network = Listener("0.0.0.0",8080)
db = DBStream("172.1.1.23", 1433)

// Nodes: tasks that either daemonize or run once.
parent = Container(
        inputs=[signal, file, network]
        ...
        on_exit=<Restart|Daemon|Job>
        exit_handler=...FUTURE: some other task definition?
    )

// Some relational capability: tasks may depend on one another.
child = Container(
        depends-on=task1,
        input=task1.Stdout(),
        dockerfile=""
    )

task3 = Container(

    )

// Edges: things you can listen to. These options smell like grpc...
discrete_stream = task2.Stdout()     // listen to discrete messages (get message, close source, listen again)
continuous_stream = task2.Listener() // listen to realtime (listen continuously)
poll_stream = task3.Poll() // not even sure...

```

### Examples

A local build environment: automate the tasks of building and developing locally.
A dev/build loop is not persistent, every task is discrete and exits with either
one or zero. This is a good starting model, the simplest case to support.

```go

// Watch one or more directories (these could aggregate, not sure how)
src_directory = FileSystem(
    "./src"
)

build = Container(
    inputs=[src_directory]
      dockerfile="./test/Dockerfile",
    command="dotnet build ."
    mounts="./results"
)

tests = Container(
    inputs=[src_directory]
    dockerfile="./test/Dockerfile",
    command="dotnet test ."
    mounts="./results"
)

result_directory = FileSystem(
    "./results"
)

stats = Container(
    inputs=[result_directory],
    dockerfile="./test/Dockerfile.summary_stats",
    command=""
)




```

A completely generic layer-1 language using natural domain language instead of raw system speak.

* Container -> 'Task'
* FileSystem -> 'Observer'

While this looks very procedural and potentially complexity-prone, note how each
object definition aligns with gitlab pipelines, which provide a valid
inspiration: a locally-running pipeline definition. This is a combination of
tilt and GL pipelines.

Observations:

* Task: these are like job definitions
* Observer: these are granular rules, like graph edges or triggers
  * Tasks are nodes, Observers are edges
  * Tasks are blackboxes, Observers are structured rules
  * A Task can exist in isolation, and Observer cannot
  * Per GL pipelines, Tasks are like Jobs and Observers are like the rulesets
      that GL pipelines implement. Rules in GL are poorly constrained, and are
      spilling over their file spec because they are a coherent abstraction
      (execution rules for edge triggers on a graph). Making Observers a
      first-class citizen ensures that rule-definitions and relation have a
      single abstraction; on the first go, Observer could be a facade over
      complex constructions of rules, obervers, etc, and then refactored later.
      It is possible that Observer and Ruleset might be distinct entities,
      needs implementation to figure out.
        *GL pipelines: CI_COMMIT_BRANCH == "blah"
        * Observer equivalent: some method or chain of Observers (i.e. a Git observer)
* Tasks trigger based on discrete 1/0 logic from Observers: the observer merely
  signals that its resource has changed.
* Streams can be bounded as a single task, ie the discrete-task model absorbs streams (partially) if the streamer and streamee are bounded together, ie multicontainer pods. While multicontainer pods provide an example abstractions, they could actually live separately across networks. Bounding stream/streamer as a single task may be overly assertive, but may provide useful means of integration.

```go

// Watch one or more directories (these could aggregate, not sure how)
src_directory = Observer(
    "./src"
)

build = Task(
    inputs=[src_directory]
    dockerfile="./test/Dockerfile",
    command="dotnet build ."
    mounts="./results"
)

tests = Task(
    inputs=[src_directory]
    dockerfile="./test/Dockerfile",
    command="dotnet test ."
    mounts="./results"
)

result_directory = Observer(
    "./results"
)

stats = Task(
    inputs=[result_directory],
    dockerfile="./test/Dockerfile.summary_stats",
    command=""

manual_permission = Observer(
    stdin
)

```

### Implementation

Task:

* Builds the resource or caches it
* Executes command to start
* Restarts if restartable (a daemon)
* Logs process health issues, acts like kubelet

Methods:

```go
    Task(
        inputs []<Observer>,
        dockerfile string,
        build_context string,
        command string,
        mounts string,
        mode RunningMode<Daemon|Job>
        env map<string,string>
    )

    Stderr() Observer
    Stdout() Observer
    WithStdin(observer Observer)

    Observers []Observers

    Run(context Context)
```

Observer:

* Signals via a 1/0 edge-trigger that its observed object has changed
* There could be many observer types: files, network, etc.
* Signaling is done via a channel that outputs an event when its observed item changes
* Start with a simple file observer only; other observers could be implemented later.

```go
    Observer(
        path=""
    )

    <-channel<Event> Events

    Run(context Context)
```

Critiques: there is a lot of structure when building. This is what Makefiles are for.
It is unlikely that build steps can be forced into a single dockerfile/container abstraction.
We might need a Builder object that Tasks can consume, to separate building from runtime.
A Builder could just be a facade over a Task, since running a make recipe has all of the
same requirements (a build container to run in, a command to run, some mounts/secrets).

#### A Task-Oriented Yaml DSL

This is a generic task-based yaml definition, but sans any workflow structure other
than simple discrete signals. For instance, no complex graphical tools of loop structures
are possible here, but more of a sparse layer-1 description like a docker compose file that
simply incorporates service dependencies. Recall that docker-compose does have such
mechanisms, and the gaps can be filled with scripts. This example is a mere thought
experiment in terms of starting sparse, and only adding features to a workflow language
as needed. Also, this is just a yaml foray, to start building yaml-based DSL counterparts
to turing-complete library models of devster.

A few tangents:

* this attempts to shove everything under devops object models: ie, forcing all observers into an "image",
  which is an abstraction of a deliverable, everything required to build and run a task
* the motivation here is to keep everything under/within common devops abstractions, which is a strategy
  for tool reuse, but also a consideration that devops things (images, dockerfiles, etc) are really
  consistent abstractions of recurrent domain objects that are universal (ie, since software is
  turing complete/universal, a workflow derivation built atop it should extend these base abstractions
  like makefiles, dockerfiles, etc)
* drawbacks: a turing-complete code environment like python of implementing this in a compiled
  library would be more powerful; however, declarative approach during design should yield
  more disciplined boundaries and abstractions.
* in terms of development requirements, this example makes it clear that a standard-declarative DSL
  is needed to guide development; this separates configuration from compilation, and ensures that
  coding semantics are rigid/clean.

```yaml

# Observe these, and their inputs
images:
  - name: app
    src-files:
      - ./app1/src
      - ./app1/build/Dockerfile
      - ./app1/tests
    build-cmd: 'docker build . --build-arg=abc=$ABC'
    env:
      - ABC: "some-value"
  - name: sts-puller
    src-files:
      - ./app2/src
      - ./app2/build/Dockerfile
      - ./app2/tests
    secrets: uber-secret
    build-cmd: 'make build'

tasks:
  - name: secrets-retriever
    image: sts-puller
    depends-on: null # depends on nothing
    with-stdin: manual # take input from a console
    cmd: "python app.py"
    restart: once # if exits with 0, don't restart (ie, Job semantics)
    on-error: restart  # if exits with non-zero
    volumes: /etc/ssl/certs:/etc/ssl/certs
    env:
      - FOO: BAR
  - name: sql
    image: postgres
    cmd: "./start" # ignore
    ports: 3306
    networks: localhost # localhost, some docker net, a net namespace, etc.
  - name: my-app
    image: app
    depends-on: secrets-retriever, db
    ports: 8080
    on-error: restart
    restart: always
  - name: monitor
    image: bash
    depends-on:   # notice that depends-on is probably a list: let an app declare multiple dependencies, with additional properties, such as restart logic per dependency
      - name: my-app
        if-error: restart # if parent dependency exits non-zero, restart me
        # other properties here? probably all future, but there is 
    with-stdout: my-app
    with-stderr: my-app
    cmd: "some shell to filter and colorize 'error' and 'warning' messages"
    terminal: yes  # open a terminal when this runs
      - stdin: no
```

The above implicitly implements these features (mostly just build and compose logic):

* on changes to image artifacts or input, everything downstream is rebuilt and deployed
* on exit/error,
* the above could be recursively defined! the daemon watching this file and coordinating containers
  might itself be implemented by such a file!
* only layer one stuff: files watched, stdin/stdout/stderr, restarts, depends-on

It explicitly excludes:

* git branch logic and rulesets (manual triggers, if branch=xyz, on merge, hooks, etc)
* most other structured workflow logic
* secrets aren't excluded but are a big question mark; hopefully they could be 'absorbed' into other layer one
  constructs, such as a parent task that contains all trust and builds a secure environment, then passes
  secrets through a layer such as stdin

So okay, so far I'm starting to graps the layer one requirements, at least a little clearly.
Identified layer-one concrete elements:

* stdin/stderr/stdout plumbing (could even tee these)
* restart
* image and monitored build artifacts
* depends-on: an agnostic connection between tasks by which graphical structure is built (their communication is implemented elsewhere or internall in app logic)
* ports and networks: again, these identify things to create/open, but not their internal content or structure

The above are concrete items, which should map to one or more bounded
characteristics. One way to think of layer-one is as the control layer of a
StepFunction implementation, whereas the 'cool features' of a workflow system
are all layer-two; control if failover, build stuff/images, and only system
level definitions. But hopefully these can all be generic, such that even a
'port' or pipe (stdin, stdout, etc) could be some abstraction, though this is a
bit extreme. Per the above, layer one encompasses only these characteristics:

* build artifacts and tasks: these are nodes
* resources: networks and pipes (stdin/stdout/stderr plumbing)
* structure: relationships between tasks, i.e. graphical structure
* edge-behavior: behavior defined  on edges; whereas structure defines edges,
  these are the structure system behavior defined on them. This is a restricted
  set, only system verbs.
  * Relations, in order of layering from pure process stuff to application level logic
  * depends-on: defines directed relationships and restart behavior ONLY
  * pipes: stdin/stderr/stdout
    * NOTE: notice that since these are creation-time, they impose restart reqs
      if parent dies! ie, broken-pipe
  * network resources, ie ports opened: these allow application-layer info to
        flow, without any implications to layer one behavior (except perhaps
        network and local errors, like unavailable ports, etc)
    * Internall these could implement GRPC, HTTP, REST.
* DISCRETENESS: layer one is all discrete and with only binary 1/0 edge triggers
  ("restart", "input changed / rebuild", "manual trigger")
  * no internal symbols or verbs; stick with binary logic

Critiques:

* it is safe to say that 80% of the value above could be implemented by
  makefiles, docker-compose+docker+bash-glue
  * code generation could generate makefiles, docker files, and scripts to
    implement the above
  * the task concrete items (ports, env, depends-on) map nearly 1:1 to
    docker-compose verbs of the same name
  * Rebuttal:
    * discrete trigger logic: the difference/extension is that of piping and
        depends_on restart logic, etc. The discrete edge triggers above cannot
        be implemented by compose, and are the core idea of layer-one: graphical
        structure and 1/0 edge triggers.
    * observability: the definition is also heavily intended to support, extend,
        and describe observability (ie file system changes trigger builds) and
        in the future to support other observable sources (sqs queues, streams,
        etc.)
    * extension: the ultimate goal is to extend "tasks" to serverless and
      scalable ideas: HPAs, lambda, etc.
    * for example, a structured test workflow could not be implemented in
      compose, etc: "when I save my new code, or commit it, run unit tests only
      when I say, run integration tests when I am ready" Other discrete,
      devsecops style continuous development processes could not be implemented,
      but this is where we start to combine "leaky" layer one/two things like
      secrets or git-branch conditions.

Next task: work through some layer two examples to help divide the leaky aspects
of layer one/two abstractions, and also, frankly, see if either fits cleanly
into existing tooling (docker compose, etc).

An incomplete enumeration of layer-2 features not implementable in layer-one's
language. Brainstorming: these are capabilities to distinguish layer one/two
implementation boundaries, useful catalysts for deriving better definition of
these boundaries.

* chains, forks, joins, etc:
  * layer one could not implement a job that repeats n times or based on conditions
* secrets
* git branches and condition logic: "include graph substructure only if" and "execute only if" (the latter
  includes a substructure but executes part of it only based on )
  * equivalent in most ways to gitlab rules
* dynamical topology: generate graph structure at runtime (this sounds entirely far out, future)
* custom types and triggers: these are mostly plugins of some form
  * git hooks/actions
  * k8s wrappers
  * lambda, sqs, other listeners
  * ML-flow ideas, data science stuff
  * data lake verbs: "on arbitrary job, execute job" (review Data Lake book and margin notes)
* learn and merge the StepFunction language itself into layer 2?
* MyJob().OnError(Email("<admin@fubar.com>"))
* Scalable("some-cpu-intensive-task").OnCompletion()
* Eventual consistency transactions, other important transaction patterns:
  * see Fundamentals of Software Architecture p 132, 'Distributed transactions'
  * BASE/Transactional sagas: distributed transactions have unique requirements
    that may need to be satisfied, as certain conditions are stepwise; implementing
    some known stepwise transaction pattern may be a good test case for layer-2
    acceptance.
* resiliency, rollback, versioning of the whole thing
* ACID between jobs/processes: recall how complex this is in distributed environments.
* attribution, security and other orthogonal concerns (think mil/int)
* Layer-2 should contribute a fully defined workflow for a system: the developer
  begins defining their development workflow in a layer-2 definition, and this
  definition is 1) replicable to other developers so they can build/test/dev
  2) deployable and versioned, not merely a 'development tool'
* ensure a complex transaction executes across arbitrary boundaries: a transaction that
  flows across regions, CSPs, or between unreliable technologies to end systems.
