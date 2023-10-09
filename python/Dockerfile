FROM python:3.6-alpine
WORKDIR /docker
COPY . /docker
RUN pip install requests Flask gunicorn
EXPOSE 5000
ENTRYPOINT ["gunicorn"]
CMD ["-w","4","-b","0.0.0.0:5000", "API_container"]
