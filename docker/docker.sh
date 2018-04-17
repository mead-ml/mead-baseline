usage() {
    echo "Usage: $0 [-g <GPU number>] [-n <container name>]" 1>&2; exit 1;
}

while getopts ":g:n:" x; do
    case "${x}" in
        g)
            GPU_NUM=${OPTARG}
            ;;
        n)
            CON_NAME=${OPTARG}
            ;;
        *)
            usage
            ;;

    esac
done
shift $((OPTIND-1))


if [[ -z "${GPU_NUM// }" || -z "${CON_NAME//}" ]]; then
    usage
    exit 1
fi
echo "using GPU: "${GPU_NUM}", container name: "${CON_NAME}

CON_BUILD=baseline
docker build --network=host -t ${CON_BUILD} -f Dockerfile ../
if [ $? -ne 0 ]; then
    echo "could not build container, exiting"
    exit 1
fi


NV_GPU=${GPU_NUM} nvidia-docker run -e LANG=C.UTF-8 --rm --name=${CON_NAME} --net=host -v /data/embeddings:/data/embeddings:ro -v /data/datasets:/data/datasets:ro -v /data/model-store:/data/model-store -v /data/model-checkpoints:/data/model-checkpoints -it ${CON_BUILD} bash

