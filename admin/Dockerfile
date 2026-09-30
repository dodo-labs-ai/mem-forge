ARG DOCKERHUB_PREFIX=

# Deprecated (admin-ui-v1): builds the V1 admin UI served at /. Delete this
# stage and its COPY below when V1 is removed (ADR 0044).
FROM ${DOCKERHUB_PREFIX}node:20-alpine AS build-v1

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

FROM ${DOCKERHUB_PREFIX}node:20-alpine AS build

ARG NPM_REGISTRY=

ENV NPM_CONFIG_UPDATE_NOTIFIER=false

WORKDIR /app

COPY admin/package*.json ./
RUN if [ -n "$NPM_REGISTRY" ]; then \
        npm ci --registry "$NPM_REGISTRY" --no-audit --no-fund; \
    else \
        npm ci --no-audit --no-fund; \
    fi

COPY admin/ ./
RUN npm run build

FROM ${DOCKERHUB_PREFIX}nginx:1.27-alpine

COPY admin/nginx.conf /etc/nginx/conf.d/default.conf
# Deprecated (admin-ui-v1): the V1 bundle at / (ADR 0044).
COPY --from=build-v1 /app/dist /usr/share/nginx/html
COPY --from=build /app/dist /usr/share/nginx/html/v2

EXPOSE 8080
