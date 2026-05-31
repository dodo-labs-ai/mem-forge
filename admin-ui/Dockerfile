FROM node:20-alpine AS build

WORKDIR /app

COPY admin-ui/package*.json ./
RUN npm ci

COPY admin-ui/ ./
RUN npm run build

FROM nginx:1.27-alpine

COPY admin-ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
