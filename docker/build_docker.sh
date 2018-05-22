MEAD_CONFIG_LOC='../python/mead/config/mead-settings.json'
if [ ! -e $MEAD_CONFIG_LOC ]; then
    echo -e "{\"datacache\": \"$HOME/.bl-data\"}" > $MEAD_CONFIG_LOC
fi

CON_BUILD=baseline
docker build --network=host -t ${CON_BUILD} -f Dockerfile ../
if [ $? -ne 0 ]; then
    echo "could not build container, exiting"
    exit 1
fi
