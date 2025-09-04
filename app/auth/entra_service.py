import jwt
import msal
import httpx
import logging

from typing import Dict, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.config.settings import settings

logger = logging.getLogger(__name__)

class EntraIDService:
    """
    Azure Entra ID authentication service using MSAL.
    Handles token validation, user info retrieval, and SSO integration.
    """
    
    def __init__(self):
        # Azure AD endpoints
        self.authority = f"https://login.microsoftonline.com/{settings.AZURE_TENANT_ID}"
        self.jwks_uri = f"https://login.microsoftonline.com/{settings.AZURE_TENANT_ID}/discovery/v2.0/keys"
        
        # MSAL confidential client for server-to-server calls
        self.msal_app = msal.ConfidentialClientApplication(
            client_id=settings.AZURE_CLIENT_ID,
            client_credential=settings.AZURE_CLIENT_SECRET,
            authority=self.authority
        )
        
        # Cache for JWKS keys (Azure's public keys for token verification)
        self._jwks_cache = None
        self._jwks_cache_expiry = None
        
        logger.info("EntraIDService initialized successfully")
    
    async def get_jwks_keys(self) -> Dict[str, Any]:
        """
        Fetch and cache Azure AD's JSON Web Key Set (JWKS) for token verification.
        """
        now = datetime.utcnow()
        
        # Return cached keys if still valid (cache for 1 hour)
        if (self._jwks_cache and self._jwks_cache_expiry and 
            now < self._jwks_cache_expiry):
            return self._jwks_cache
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.jwks_uri)
                response.raise_for_status()
                
                jwks_data = response.json()
                
                # Cache the keys for 1 hour
                self._jwks_cache = jwks_data
                self._jwks_cache_expiry = now + timedelta(hours=1)
                
                logger.info("Successfully fetched JWKS keys from Azure AD")
                return jwks_data
                
        except Exception as e:
            logger.error(f"Failed to fetch JWKS keys: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to verify authentication tokens"
            )
    
    def get_public_key_from_jwks(self, token_header: Dict[str, Any], jwks: Dict[str, Any]) -> str:
        """
        Extract the public key from JWKS based on the token's 'kid' (key ID).
        """
        kid = token_header.get('kid')
        if not kid:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing 'kid' claim"
            )
        
        for key in jwks.get('keys', []):
            if key.get('kid') == kid:
                # Convert JWK to PEM format for PyJWT
                return jwt.algorithms.RSAAlgorithm.from_jwk(key)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unable to find matching key for token"
        )
    
    async def validate_access_token(self, token: str) -> Dict[str, Any]:
        """
        Validate an Azure AD access token and return the decoded claims.
        """
        try:
            token_header = jwt.get_unverified_header(token)
            jwks = await self.get_jwks_keys()
            
            public_key = self.get_public_key_from_jwks(token_header, jwks)
            
            # Verify and decode the token
            decoded_token = jwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience=settings.AZURE_CLIENT_ID,  # Token must be for our app
                issuer=f"https://login.microsoftonline.com/{settings.AZURE_TENANT_ID}/v2.0"
            )
            
            # Additional validation
            now = datetime.utcnow().timestamp()
            if decoded_token.get('exp', 0) < now:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has expired"
                )
            
            if decoded_token.get('nbf', now) > now:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token is not yet valid"
                )
            
            logger.info(f"Successfully validated token for user: {decoded_token.get('preferred_username', 'unknown')}")
            return decoded_token
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token"
            )
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token validation failed"
            )
    
    async def get_user_info(self, token_claims: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract user information from validated token claims.
        """
        return {
            "user_id": token_claims.get("oid"),
            "email": token_claims.get("preferred_username"),
            "name": token_claims.get("name"),
            "tenant_id": token_claims.get("tid")
        }
    
    async def validate_user_permissions(self, user_info: Dict[str, Any]) -> bool:
        """
        Validate if the user has permission to access the application.
        Customize this method based on your requirements.
        """
        # Check if user belongs to Grant Thornton tenant

        logger.info(f"Validating permissions for user: {user_info}")

        if user_info.get("tenant_id") != settings.AZURE_TENANT_ID:
            logger.warning(f"User from wrong tenant: {user_info.get('tenant_id')}")
            return False
        
        # Check if user email domain is allowed
        email = user_info.get("email", "")

        allowed_domains = [domain.strip() for domain in settings.ALLOWED_EMAIL_DOMAINS.split(',')]        
        if not any(email.lower().endswith(f"@{domain}") for domain in allowed_domains):
            logger.warning(f"User email not from allowed domain: {email}")
            return False

        return True


# FastAPI Security Scheme
security = HTTPBearer()

# Global service instance
entra_service = EntraIDService()


# Dependency for protected endpoints
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict[str, Any]:
    """
    FastAPI dependency to validate authentication and return current user info.
    Use this on any endpoint that requires authentication.
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token"
        )
    
    token_claims = await entra_service.validate_access_token(credentials.credentials)    
    user_info = await entra_service.get_user_info(token_claims)
    
    if not await entra_service.validate_user_permissions(user_info):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not have permission to access this application"
        )
    
    logger.info(f"Authenticated user: {user_info.get('email')} ({user_info.get('user_id')})")
    return user_info