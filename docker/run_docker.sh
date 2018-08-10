usage() {
    echo "Usage: $0 [-g <GPU number>] [-n <container name>] [-t <container type: tf/pytorch>] [-e <external-mount-directories>]" 1>&2; exit 1;
}

while getopts ":g:n:t:e:" x; do
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
        e)
            EXTERNAL_MOUNTS+=("$OPTARG");;
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
echo "external mount directories ${EXTERNAL_MOUNTS[@]}"

CON_BUILD=baseline-${CON_TYPE}
if [ -e $HOME/.bl-data ]; then
    CACHE_MOUNT="-v $HOME/.bl-data:$HOME/.bl-data"
else
    CACHE_MOUNT=""
fi

NUM_EXTERNAL=${#EXTERNAL_MOUNTS[@]}
for ((i=0;i<NUM_EXTERNAL;i++)); do
    BASENAME=`basename ${EXTERNAL_MOUNTS[i]}` 
    EXTERNAL_MOUNTS[i]="-v ${EXTERNAL_MOUNTS[i]}:/$BASENAME"
done

NV_GPU=${GPU_NUM} nvidia-docker run -e LANG=C.UTF-8 --rm --name=${CON_NAME} --net=host -v /data/embeddings:/data/embeddings:ro -v /data/datasets:/data/datasets:ro -v /data/model-store:/data/model-store -v /data/model-checkpoints:/data/model-checkpoints ${EXTERNAL_MOUNTS[@]} ${CACHE_MOUNT} -it ${CON_BUILD} bash

