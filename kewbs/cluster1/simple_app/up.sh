#!/bin/bash

# WORK IN PROGRESS. Many parameters here will be factored differently as I learn.
# Tilt will also likely do most of this stuff anyway, or much of the lifting (registry creation and management?).

cluster_name="mycluster"

function cleanup() {
    echo "Cleaning up..."
    clusters=$(k3d cluster list -o json | jq -r .[].name)
    echo "Deleting $clusters"
    k3d cluster delete $clusters
    echo ""
}

cleanup

for arg in "$@"
do
    if [ $arg == "--clean" ]; then
        echo "Clean completed"
        exit
    fi
done

echo "Creating cluster '$cluster_name'..."
# Create a k3d cluster with n worker nodes. See: https://www.suse.com/c/introduction-k3d-run-k3s-docker-src/
# "k3d waits until everything is ready, pulls the Kubeconfig from the cluster and merges it with your default Kubeconfig"
# Note that the --registry-create command is supposed to accept a name parameter, but is currently broken in my k3d version
# FUTURE: understand and implement the --cluster-init switch to swap between etcd or sqlite; currently using defaults. HA would make a good project.
#k3d cluster create $cluster_name --agents 1 -p 8080:30080@agent[0] --registry-create --wait
k3d cluster create $cluster_name --api-port 6550 -p "8081:80@loadbalancer" --agents 1 --registry-create --wait  # see  https://k3d.io/usage/guides/exposing_services/
# Note: all ports exposed on the serverlb ("loadbalancer") will be proxied to the same ports on all server nodes in the cluster
echo "Done."