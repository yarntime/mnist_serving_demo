FROM rivernet/tomcat:8

MAINTAINER zhangwei@chinaskycloud.com

USER root

############################################################
# Setup environment variables
############################################################
ENV START_SCRIPT /root/start-up.sh

COPY ./target/mnist /opt/apache-tomcat-8.5.24/webapps/ROOT

ADD start-up.sh $START_SCRIPT

RUN chmod +x $START_SCRIPT

CMD ./$START_SCRIPT
