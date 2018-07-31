usage() {
    echo "Usage: $0 [-g <GPU number>] [-n <container name>] [-t <container type: tf/pytorch>]" 1>&2; exit 1;
}

while getopts ":g:n:t:" x; do
    case "${x}" in
        g)
            GPU_NUM=${OPTARG}
            ;;
        n)
            CON_NAME=${OPTARG}
            ;;
        t)  
            CON_TYPE=${OPTARG}
            ;; 
        *)
            usage
            ;;

    esac
done
shift $((OPTIND-1))

if [[ -z "${GPU_NUM// }" || -z "${CON_NAME//}" || -z "${CON_TYPE//}" ]]; then
    usage
    exit 1
fi
echo "using GPU: "${GPU_NUM}", container name: "${CON_NAME}, container type: "${CON_TYPE}"

CON_BUILD=baseline-${CON_TYPE}

if [ -e $HOME/.bl-data ]; then
    CACHE_MOUNT="-v $HOME/.bl-data:$HOME/.bl-data"
else
    CACHE_MOUNT=""
fi

NV_GPU=${GPU_NUM} nvidia-docker run -e LANG=C.UTF-8 --rm --name=${CON_NAME} --net=host -v /data/embeddings:/data/embeddings:ro -v /data/datasets:/data/datasets:ro -v /data/model-store:/data/model-store -v /data/model-checkpoints:/data/model-checkpoints ${CACHE_MOUNT} -it ${CON_BUILD} bash

