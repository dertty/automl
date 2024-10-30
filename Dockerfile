FROM python:3.11.4-slim

RUN apt update
RUN apt install pipx -y
RUN pipx ensurepath
RUN pipx install pdm[all]
RUN pipx upgrade pdm