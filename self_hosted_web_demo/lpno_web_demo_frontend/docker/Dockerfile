FROM nginx:1.27-alpine

# Where nginx proxies the inference routes. Override at run time:
#   docker run -e LPFNO_BACKEND_UPSTREAM=http://<backend-host>:8000 ...
# An https:// upstream signed by a private certificate authority also needs that
# certificate mounted at /etc/nginx/upstream-certs/backend.crt.
ENV LPFNO_BACKEND_UPSTREAM=http://lpno-demo-backend:8000 \
    LPFNO_BACKEND_SERVER_NAME=

COPY docker/nginx.conf.template /etc/nginx/templates/default.conf.template
COPY docker/15-lpfno-backend-upstream.envsh /docker-entrypoint.d/15-lpfno-backend-upstream.envsh
RUN chmod +x /docker-entrypoint.d/15-lpfno-backend-upstream.envsh
RUN mkdir -p /etc/nginx/upstream-certs
COPY public/ /usr/share/nginx/html/

EXPOSE 80
