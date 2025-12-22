"""
Domain Specialization Modules

Each module provides training examples for a specific domain.
"""

from .base import BaseDomain, DomainExample
from .sql_databases import SQLDatabasesDomain
from .ml_ai import MLAIDomain
from .data_engineering import DataEngineeringDomain
from .cloud_aws import AWSCloudDomain
from .cloud_gcp import GCPCloudDomain
from .cloud_azure import AzureCloudDomain
from .devops_iac import DevOpsIaCDomain
from .diagrams import DiagramsDomain
from .web_mobile import WebMobileDomain

# Registry of all available domains
DOMAIN_REGISTRY = {
    # SQL & Databases
    "sql": SQLDatabasesDomain,
    "postgres": lambda: SQLDatabasesDomain(flavor="postgres"),
    "mysql": lambda: SQLDatabasesDomain(flavor="mysql"),
    "oracle": lambda: SQLDatabasesDomain(flavor="oracle"),
    "sqlite": lambda: SQLDatabasesDomain(flavor="sqlite"),
    "redshift": lambda: SQLDatabasesDomain(flavor="redshift"),
    "snowflake": lambda: SQLDatabasesDomain(flavor="snowflake"),
    "sparksql": lambda: SQLDatabasesDomain(flavor="sparksql"),

    # ML/AI
    "ml": MLAIDomain,
    "ai": MLAIDomain,
    "machinelearning": MLAIDomain,

    # Data Engineering
    "data": DataEngineeringDomain,
    "bigdata": lambda: DataEngineeringDomain(focus="bigdata"),
    "pyspark": lambda: DataEngineeringDomain(focus="pyspark"),
    "glue": lambda: DataEngineeringDomain(focus="glue"),
    "etl": lambda: DataEngineeringDomain(focus="etl"),

    # Cloud Providers
    "aws": AWSCloudDomain,
    "gcp": GCPCloudDomain,
    "azure": AzureCloudDomain,

    # DevOps & IaC
    "devops": DevOpsIaCDomain,
    "terraform": lambda: DevOpsIaCDomain(focus="terraform"),
    "cloudformation": lambda: DevOpsIaCDomain(focus="cloudformation"),
    "kubernetes": lambda: DevOpsIaCDomain(focus="kubernetes"),
    "docker": lambda: DevOpsIaCDomain(focus="docker"),
    "cicd": lambda: DevOpsIaCDomain(focus="cicd"),

    # Diagrams
    "diagrams": DiagramsDomain,
    "mermaid": lambda: DiagramsDomain(format="mermaid"),
    "plantuml": lambda: DiagramsDomain(format="plantuml"),
    "drawio": lambda: DiagramsDomain(format="drawio"),

    # Web & Mobile
    "web": WebMobileDomain,
    "mobile": lambda: WebMobileDomain(focus="mobile"),
    "frontend": lambda: WebMobileDomain(focus="frontend"),
    "backend": lambda: WebMobileDomain(focus="backend"),
}

def get_domain(name: str) -> BaseDomain:
    """Get a domain instance by name."""
    if name not in DOMAIN_REGISTRY:
        available = ", ".join(sorted(DOMAIN_REGISTRY.keys()))
        raise ValueError(f"Unknown domain: {name}. Available: {available}")

    domain_or_factory = DOMAIN_REGISTRY[name]
    if callable(domain_or_factory) and not isinstance(domain_or_factory, type):
        return domain_or_factory()
    return domain_or_factory()

def list_domains() -> list:
    """List all available domain names."""
    return sorted(DOMAIN_REGISTRY.keys())

__all__ = [
    "BaseDomain",
    "DomainExample",
    "DOMAIN_REGISTRY",
    "get_domain",
    "list_domains",
    "SQLDatabasesDomain",
    "MLAIDomain",
    "DataEngineeringDomain",
    "AWSCloudDomain",
    "GCPCloudDomain",
    "AzureCloudDomain",
    "DevOpsIaCDomain",
    "DiagramsDomain",
    "WebMobileDomain",
]
