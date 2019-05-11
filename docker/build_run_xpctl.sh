usage() {
    echo "Usage: $0
    [-u|--user|--dbuser <db user> (required)]
    [--pass|--dbpass <db password> (required)]
    [--dbhost <db host> (default=localhost)]
    [--dbport <db port> (default=27017)]
    [-b|--backend <backend> (default=mongo)]
    [-p|--port <port to run xpctl server> (default=5310)]" 1>&2; exit 1;
}

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    -u|--user|--dbuser)
    DB_USER="$2"
    shift # past argument
    shift # past value
    ;;
    --pass|--dbpass)
    DB_PASSWORD="$2"
    shift # past argument
    shift # past value
    ;;
    --dbhost)
    DB_HOST="$2"
    shift # past argument
    shift # past value
    ;;
    --dbport)
    DB_PORT="$2"
    shift # past argument
    shift # past value
    ;;
    -b|--backend)
    BACKEND="$2"
    shift # past argument
    shift # past value
    ;;
    -p|--port)
    PORT="$2"
    shift # past argument
    shift # past value
    ;;
    -h|--help)
    HELP="$2"
    #usage
    shift # past argument
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-27017}
BACK_END=${BACK_END:-mongo}
PORT=${PORT:5310}

CON_BUILD=xpctlserver
docker build \
--network=host \
--build-arg backend=${BACK_END} \
--build-arg user=${DB_USER} \
--build-arg passwd=${DB_PASSWORD} \
--build-arg dbhost=${DB_HOST} \
--build-arg dbport=${DB_PORT} \
--build-arg port=${PORT} \
-t ${CON_BUILD}-${BACK_END} \
-f Dockerfile.xpctl \
../

docker run -e LANG=C.UTF-8 --rm --name=${CON_BUILD} --network=host -it ${CON_BUILD}-${BACK_END}