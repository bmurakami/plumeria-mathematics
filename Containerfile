FROM docker.io/spack/ubuntu-noble

WORKDIR /root
RUN curl -k -O https://download.swift.org/swiftly/linux/swiftly-1.0.1-$(uname -m).tar.gz
RUN tar -zxf swiftly-1.0.1-$(uname -m).tar.gz
RUN ./swiftly init --quiet-shell-followup -y
RUN rm LICENSE.txt swiftly*
RUN . "/root/.local/share/swiftly/env.sh"
RUN apt update
RUN apt-get install -y libcurl4-openssl-dev libedit2 libpython3-dev libxml2-dev libncurses-dev libz3-dev pkg-config zlib1g-dev
COPY plumath-spack/spack.yaml plumath-spack/spack.yaml
COPY spack-installations.sh spack-installations.sh
RUN apt-get install -y zsh openssh-client
RUN . /opt/spack/share/spack/setup-env.sh && . /root/spack-installations.sh
