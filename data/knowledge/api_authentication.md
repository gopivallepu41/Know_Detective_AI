# Authentication API

The Authentication Service exposes `POST /login` and `POST /refresh`. Access tokens expire after 1 hour. Refresh tokens expire after 30 days. The service currently returns both `access_token` and `refresh_token`.

Service-to-service authentication uses JWT tokens issued by the Authentication Service.
