FROM python:3.11.4-slim

# Обновление и установка системных зависимостей
RUN apt update && apt upgrade -y
RUN apt-get install -y \
    build-essential \
    swig \
    wget \
    curl \
    git \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgdbm-dev \
    libdb5.3-dev \
    libbz2-dev \
    libexpat1-dev \
    liblzma-dev \
    tk-dev \
    libffi-dev

# Скачивание и установка Python 3.8
RUN wget https://www.python.org/ftp/python/3.8.18/Python-3.8.18.tgz
RUN tar -xzf Python-3.8.18.tgz
RUN cd Python-3.8.18 && ./configure --enable-optimizations && make altinstall

# Установка pip для Python 3.8
RUN python3.8 -m ensurepip --upgrade

# Установка pipx и pdm
RUN apt-get install -y pipx
RUN pipx ensurepath
RUN pipx install pdm[all]
RUN pipx upgrade pdm

# Проверка версий
RUN python3.8 --version && python3.11 --version