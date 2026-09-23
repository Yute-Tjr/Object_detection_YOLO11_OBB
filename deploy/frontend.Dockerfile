ARG NODE_BASE_IMAGE=node:22-alpine
ARG NGINX_BASE_IMAGE=nginx:1.27-alpine
FROM ${NODE_BASE_IMAGE} AS build

WORKDIR /app
COPY web_frontend/package.json web_frontend/package-lock.json ./
RUN npm ci
COPY web_frontend ./
RUN npm run build

FROM ${NGINX_BASE_IMAGE}
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY --from=build /app/dist/client /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
