FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
# PATHの設定
ENV HOME /root
ENV PATH /usr/local/bin:$PATH
# apt-get installする時にnoninteractiveにする
ENV DEBIAN_FRONTEND noninteractive
# エラーメッセージの抑止
ENV DEBCONF_NOWARNINGS yes
# apt-get　update
RUN apt-get clean && apt-get update
# RUN apt-get upgrade -y
# localeの設定
RUN apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
# apt-get man command
RUN rm /etc/dpkg/dpkg.cfg.d/excludes \
    && apt-get install -y man
# build関連でぶち込んどいたほうがいいやつ全部のせ
# ミニマリストの方はいらないものを消してってください
RUN apt-get install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

# pyenv settings
ENV PYTHON_VERSION 3.8.16
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc

RUN pyenv install $PYTHON_VERSION
RUN pyenv global $PYTHON_VERSION
RUN pyenv rehash

# install poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH $HOME/.local/bin:$PATH
WORKDIR /home/workspace
COPY poetry.lock .
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry lock
RUN poetry install
