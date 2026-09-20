FROM node:22-alpine AS build

WORKDIR /app
COPY web_frontend/package.json web_frontend/package-lock.json ./
RUN npm ci
COPY web_frontend ./
RUN npm run build

FROM nginx:1.27-alpine
COPY deploy/nginx.conf /etc/nginx/nginx.conf
COPY --from=build /app/dist/client /usr/share/nginx/html

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
