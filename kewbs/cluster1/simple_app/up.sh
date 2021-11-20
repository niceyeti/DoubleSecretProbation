#!/bin/bash

cluster_name="devcluster"
# Note: k3d prefixed many objects it creates 'k3d-', hence you gotta prefix to refer to them after they are created.
prefixed_cluster_name="k3d-$cluster_name"

function cleanup() {
    echo "Cleaning up ALL clusters."
    clusters=$(k3d cluster list -o json | jq -r .[].name)
    echo "Deleting $clusters"
    k3d cluster delete $clusters
    cat <<- EOF
Cluster deleted.
Note: although the cluster was deleted, you may want to manually prune images, volumes etc with:
* docker system prune -a
* docker image prune -a
But be sure you know what these commands do before executing, or you may accidentally delete desired
containers and images, such as your dev images/containers. Likewise, you'll have to re-pull many base
k3d images.
EOF
}

function create() {
    echo "Creating cluster '$cluster_name'..."
    # Create a k3d cluster with n worker nodes. See: https://www.suse.com/c/introduction-k3d-run-k3s-docker-src/
    # "k3d waits until everything is ready, pulls the Kubeconfig from the cluster and merges it with your default Kubeconfig"
    # Note that the --registry-create command is supposed to accept a name parameter, but is currently broken in my k3d version
    # FUTURE: understand and implement the --cluster-init switch to swap between etcd or sqlite; currently using defaults. HA would make a good project.
    #k3d cluster create $cluster_name --agents 1 -p 8080:30080@agent[0] --registry-create --wait
    #k3d cluster create $cluster_name --api-port 6550 -p "8081:80@loadbalancer" --agents 1 --registry-create mycluster-registry
    # Note: all ports exposed on the serverlb ("loadbalancer") will be proxied to the same ports on all server nodes in the cluster

    echo "Creating cluster. NOTE: to avoid re-pulling images use the '--pause' and '--restart' flags."
    k3d cluster create --config k3d_config.yaml
    echo "Cluster created."
}

function pause() {
    echo "Stopping $prefixed_cluster_name cluster."
    k3d cluster stop $cluster_name
    echo "Pause completed."
    echo "Use './up.sh --restart' to restart when meatspace concerns no longer prevent your intellectual development."
}

function restart() {
    k3d cluster start $cluster_name
    echo "Cluster $prefixed_cluster_name restarted."
}

function show_help() {
cat <<- EOF
Commands:
    * Create a cluster from scratch:
        ./up.sh --new
    * Restart a cluster:
        ./up.sh --restart
    * Delete old cluster, including registry:
        ./up.sh --clean
    * Pause the cluster, saving the registry as well:
        ./up.sh --pause

Explanation: it is best to use the k3d api to manage all cluster resources, instead of implementing
scripts to do so, especially as the k3d api develops. Creating a cluster can also create a registry,
which can be problematic when rebuilding the cluster from scratch every time, since the new registry
then has to re-pull k3d related images.

Thus the basic KISS usage of k3d is best: run create to make a cluster and registry, then run 'stop'
and 'start' to save or restore the cluster. This will retain the registry. IOW, let k3d manage the cluster
and registry together, don't manage the dev registry separately.
EOF
}

for arg in "$@"
do  
    if [[ $arg == "--new" || $arg == "--create" ]]; then
        create
        exit
    fi

    if [[ $arg == "--clean" || $arg == "--delete" ]]; then
        cleanup
        exit
    fi

    if [[ $arg == "--pause" || $arg == "--stop" ]]; then
        pause
        exit
    fi

    if [[ $arg == "--restart" || $arg == "--start" ]]; then
        restart
        exit
    fi

    if [[ $arg == "--help" || $arg == "-h" || $arg == "help" ]]; then
        show_help
        exit
    fi
done

show_help