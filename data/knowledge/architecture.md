# Authentication Architecture

The Authentication Service signs JWT v2 tokens. Order Service and Payment Service validate JWT tokens through the shared authentication library.

The authentication service migrated from JWT v1 to JWT v2 during the platform modernization work. The current token policy is 1 hour for access tokens and 30 days for refresh tokens.
