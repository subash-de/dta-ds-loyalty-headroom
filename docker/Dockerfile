FROM python:3.7-buster
ARG TOKEN
ARG PY_PACKAGE

RUN mkdir /app
WORKDIR /app
ADD . /app/

RUN pip install ${PY_PACKAGE} --extra-index-url "https://${TOKEN}@pkgs.dev.azure.com/dta-devops/datascience-platforms/_packaging/dta-ds-libraries/pypi/simple/"

EXPOSE 5000
CMD ["python", "/app/main.py"]
