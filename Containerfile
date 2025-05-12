FROM swift:latest

WORKDIR /root
RUN apt update
RUN apt-get -y install bzip2 ca-certificates g++ gcc gfortran git gzip lsb-release patch python3 tar unzip xz-utils zstd
RUN apt-get -y install nano
RUN git clone -c feature.manyFiles=true --depth=2 https://github.com/spack/spack.git
RUN . spack/share/spack/setup-env.sh && spack install -y openblas
