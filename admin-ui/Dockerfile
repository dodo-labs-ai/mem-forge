ARG DOCKERHUB_PREFIX=
FROM ${DOCKERHUB_PREFIX}node:20-alpine AS build

ARG NPM_REGISTRY=

ENV NPM_CONFIG_UPDATE_NOTIFIER=false

WORKDIR /app

COPY admin-ui/package*.json ./
RUN if [ -n "$NPM_REGISTRY" ]; then \
        npm ci --registry "$NPM_REGISTRY" --no-audit --no-fund; \
    else \
        npm ci --no-audit --no-fund; \
    fi

COPY admin-ui/ ./
RUN npm run build

FROM ${DOCKERHUB_PREFIX}nginx:1.27-alpine

COPY admin-ui/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
