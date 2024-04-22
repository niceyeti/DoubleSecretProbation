## Separation of Languages

During requirements analysis it was found that there are two separable languages, though this could easily change through code-write/refactoring activities to even better definition:

* **layer-one**: Layer one consists of a primarily graphical description of tasks and their relationships. This is primarily a system-level description, similar to docker compose files or gitlab-ci pipelines, but with added features for graphical definition and observability (trigger based executions).
* **layer-two**: Layer two is the application layer. In truth, layer-two could be many different languages built atop layer one: StepFunctions, local task graphs, etc.

The primary point: **layer one consists of all components and language primitives atop of which layer-two could be implemented simply through wrappers of layer-one constructs.** This provides a metric for distinguishing them. Thus layer one is paramount, which is nice anyway because it is closer to a local devops tool like make, compose, etc.

## Layer one

The primary components of layer one requirements are as follows:

* **Tasks**: these are units of work, the nodes in the graph. They are more production oriented than 'build' items, though they could implement builds. They are linked via `depends_on` relations (edges), which for now operate on 0/non-zero exit
* **Observers**: observers behave like channels in Golang, observe resources such as file systems.
* **Rules** [FUTURE]: these implement repo/branch logic (or other extensions) to modify the topology of the graph.

These three components are used to build workflow topologies, over which many extensions can be implemented in the future: daemons, different edge conditions, linux style stdin/out pipe-filter patterns, continuous streams, additional observer sources (sqs queues, etc), cloud plugins (lambdas), secrets, etc. The point is that these top-level components are sufficient to describe/implement:

* Workflows: graphically-described discrete-system designs
* Observability: changes to observers (ie file system changes) triggers a cascading restart to all downstream tasks.
* Tracking and mutation (rule-sets based on git branches)

## Component Details

Exploring the fore-mentioned components in more detail...

### Task

A Task is a unit of work, and a graphical node with input/outputs by which to build edge relations.

* Image: each Task specifies an image and command to execute
* Build: optionally, any Task may specify a 'build' section specifying build parameters, much like Gitlab tasks, with whatever build-list subkeys are needed (Dockerfile, 'script', 'volume', 'secret', 'build-image' etc). 'Build' implements the observer pattern, such that changes to its inputs (dockerfile, Makefile, code, etc) triggers a rebuild/restart of the Task to which it belongs.
* Depends_On: the primary relation by which Tasks are linked together. For now, relations have binary triggers, whereby the restart of an upstrem Task on which one depends triggers oneself to restart as well, in topological order.
* FUTURE: all of these are future.
  * stdin: plug the stdout of another Task into this one.
  * secrets: mount secrets
  * volumes: mount a volume into the container.
  * daemonize: some mechanism to allow Tasks to daemonize and restart on via their own logic, likely subkeys of the Daemonize type/key.
  * streams: some notion of long-lived streams exists here, i.e. grpc service and client. However this is a good design thought-experiment, since it highlights that layer-one may benefit from having only Task, Observers, and simple 0/1 edge (restart) relations; using only these operators, one can "up" a stream-based system which implements streams at more of an application layer, whereas layer-one is merely topological per behavior and system-oriented (basically linux-like) per implementation. Layer-one's business logic is all workflow structure; its implementation is all system-prog like.

Yaml spec:

```yaml
# A simple task
Task: # copy Kubernetes Pod constructs
    container:
        name: simple_task
        image: alpine
        commands:
        - echo hello world

# A custom task: rebuilds and restarts when its 'build' subkeys change (files, code, etc).
Task: 
    container:
        name: custom_task
        image: custom_image
        commands:
            - TOKEN=$(get_secret.sh)
            - ./run --secret $TOKEN
        build:
        image: builder:123 # is this necessary? maybe, like gitlab job images
        commands:
            - docker build -
        paths: # rebuild when any of these change
            - /app/src
  
# A test runner task. This would be a child task of some build task.
Task:
  Depends_On: build_task
  Pod:
    container:
      name: golang
      image: golang:1.20
      commands: go test .
      volumes:
      - ./app/test
```

### Observer

Observer is open to many extensions, but plan on keeping simple for now. An Observer defines the unit of change, i.e. triggers.

* Type: the type of observer, i.e. FileSystem (a file or directory path)
* other stuff [FUTURE]: since Observers have trigger logic, they could implement many modifications (max update rate, timeout, etc.). This would be useful to throttle the change rate, or to compose Observers with other observers. This extends all trigger logic from computer engineering systems.
  * Throttle: a min relaxation time between triggers
  * Parent: some compositional notion of parent-child gating between triggers, as well as AND/OR logic would be good here.
  * Image+Command: some image to run, ie a crawler or network watcher of some kind.
  * **IMPORTANT**: when implementing Observer features, think about how to leverage composition instead of special features. There are probably clever ways to compose Observers if their interfaces are flexible enough to do so. For instance, recall how linux' reduction to file abstractions makes observability easy. Same deal. Maybe all observers can be forced to implement simplifying but extensible semantics.
  
## Rules

[FUTURE] Rules can more or less match gitlab-ci rules and similar CI implementations to modify the graph based on repo/branch conditions or other aspects. It is important to remember though that the goal of rules is to implement the business value of modifying workflows based on an environment:

* business req: modify structure based on environmental info
* execute actions based on hooks: git action/hook integration
  * on_pull, on_push, on_checkout, etc.
  * These would be useful for many shift-left concerns:
    * environment initialization, dev-environments/uniformity
    * policy enforcement, security
    * devsecops
    * shift-left
    * other observability
* implementation: this can be done via git branches, env vars, and many other ways.
* [FUTURE] A cool extension that others lack would be a dynamic modifier that would add/disable Tasks based on some observer output ("when I see input X, execute workflow XYZ.123; when I see input y, executre XYZ.456)

## Reference Material, Evaluations

I recently came across Argo Workflows, which implement a highly similar graphical workflow CRD in kubernetes: <https://argo-workflows.readthedocs.io/en/latest/>

The language is a subset of that above, but includes UI, dashboards, and features such as
looping, cancellation, repeats, on-failure logic, DinD, and so forth (according to their list).
However they don't include:

* observables
* rules
* locally-runnable

Argo Workflows execute only under kubernetes environments, but kubernetes' controllers, CRDs, and
distributed state management offer many advantages for a workflow execution engine. Per the feature list above for example, saving/resuming/cancelling and other features are easily implemented because of the discrete separation of states/transitions implemented by K8s objects and controllers.

## Development Schedule

Subject to change as needs arise:

TLDR conclusion:

* implement the tasks file definition above; design/develop/implement Rules and Observables as first-class objects not dependent on Tasks.

1) Implement Task and its constrution/destruction on signals.
    * This will clarify the persistent state reqs: how to write-down and transition states.
    * Controller sidecar: observes Task yaml definition
        * sidecar may develop into a sidecar pattern or a God controller of all Tasks. A sidecar might be nice for mapping various observables.
        * Starts/kills Task
        * Possibly transforms comms and signals.
    * Builder: most likely belongs not in sidecar but in the God controller. Observes Yaml def build section and rebuilds on change.
    * Graph controller: this is the "God" controller. Communicates with Task sidecar. Builds images on change. Signals their restart. Eventually implements depends_on relations and other stuff. Logs Task state transitions.
    * NOTE: for beginnsies, keep all the above in one process, except Tasks processes themselves. They can be interfaced off later. Maintain the local-tool culture of this tool.

2) Implement Task dependencies

1) implement Tasks yaml file above
2) Rules and Observables are TBD
    * it is possible that these will factor both out and into the Tasks yaml. Recall that gitlab-ci's yaml is a terrible mess because all of the rulesets and added features have made their yaml def unreadable.
    * I suspect that Rules and Observables will backward integrate into the Tasks definition: piecemeal features of their implementation may be in Tasks, other in first-class yaml object defs.
    * However, it would be good to strive for first-class, orthogonal object defs; one could define rulesets that are completely independent of task definitions, and thus encompass independent complexity and description. Likewise with observables. Even if this totally-independent approach allows interesting possibilities (Rules that don't relate to any Tasks in the same definition, or that specify invalid graph structures, etc.), it will lead to better design; further, **a layer above this raw definition could apply language constraints to simplify usability**.
    Let Rules and Observables act as independent objects, then shrinkwrap them later with some application layer logic. In this manner, Rules and Observable will also obey a simple library form for better reuse.
    * Conclusion: start with the design of Rules and Observables per first-class objects.
3) Extend to Layer-two language.
