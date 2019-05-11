# for some reason if you want to regenerate the server and client. Not recommended 
rm -rf swagger_server
rm -rf swagger_client
wget http://central.maven.org/maven2/io/swagger/swagger-codegen-cli/2.4.5/swagger-codegen-cli-2.4.5.jar -O swagger-codegen-cli.jar
JAR=swagger-codegen-cli.jar
java -jar ${JAR} generate \
  -i xpctl.yaml \
  -l python-flask \
  -o server \
  -D supportPython2=true
mv server/swagger_server . 
rm -rf server
java -jar ${JAR} generate \
  -i xpctl.yaml \
  -l python \
  -o client
mv client/swagger_client .
rm -rf client
cp swagger_static/__main__.py swagger_server/__main__.py
cp swagger_static/xpctl_controller.py swagger_server/controllers/xpctl_controller.py
cp swagger_static/configuration.py swagger_client/

