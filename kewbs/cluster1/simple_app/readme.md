Helm:
* To verify chart, generate its yamls: `helm template [chart name] [chart directory]`
    * `helm template simple_app .`
    * `helm template . --debug --values Values.yaml`


Registries:
This is complicated because of the pointers to registries in the charts, and the credentials handling involved.
* See https://k3d.io/v5.0.3/usage/registries/
* Run:
    1) Run a private registry in the cluster:
    ```
    # Create a cluster and a registry
    k3d cluster create mycluster --registry-create mycluster-registry
    # Find the port on which the registry is serving
    docker ps -f name=mycluster-registry
    ```
    2) Pull an image and retag it using the image name path prefix convention to tell containerd where to get it.
    This could also just be a locally-built image, but for the sake of example:
    ```
    # some image
    docker pull alpine:latest
    # re-tag it with the registry from (1) as its prefix
    docker tag alpine:latest mycluster-registry:12345/testimage:local
    # push it to the registry in the cluster
    docker push mycluster-registry:12345/testimage:local
    # test that the image can be run in the cluster
    kubectl run --image mycluster-registry:12345/testimage:local testimage --command -- tail -f /dev/null
    ```


Tilt:
    # docker build -t companyname/frontend ./frontend
    docker_build("companyname/frontend", "frontend")

    # docker build -t companyname/frontend -f frontend/Dockerfile.dev frontend
    docker_build("companyname/frontend", "frontend", dockerfile="frontend/Dockerfile.dev")

    # docker build -t companyname/frontend --build-arg target=local frontend
    docker_build("companyname/frontend", "frontend", build_args={"target": "local"})







