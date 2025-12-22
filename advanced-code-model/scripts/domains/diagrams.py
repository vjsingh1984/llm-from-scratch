"""
Diagrams Domain

Covers:
- Mermaid diagrams (flowcharts, sequence, ER, class, state, etc.)
- PlantUML diagrams
- Draw.io/diagrams.net XML
- ASCII diagrams
"""

from typing import List, Optional
from .base import BaseDomain, DomainExample


class DiagramsDomain(BaseDomain):
    """Diagrams and visualization training examples."""

    def __init__(self, format: Optional[str] = None):
        self.format = format  # mermaid, plantuml, drawio, or None for all

    def get_name(self) -> str:
        if self.format:
            return f"Diagrams ({self.format})"
        return "Diagrams"

    def get_description(self) -> str:
        return "Diagrams with Mermaid, PlantUML, and Draw.io"

    def get_subdomains(self) -> List[str]:
        return ["mermaid", "plantuml", "drawio", "ascii"]

    def get_examples(self) -> List[DomainExample]:
        examples = []
        if not self.format or self.format == "mermaid":
            examples.extend(self._mermaid_examples())
        if not self.format or self.format == "plantuml":
            examples.extend(self._plantuml_examples())
        if not self.format or self.format == "drawio":
            examples.extend(self._drawio_examples())
        if not self.format or self.format == "ascii":
            examples.extend(self._ascii_examples())
        return examples

    def _mermaid_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a Mermaid flowchart for user authentication flow",
                code='''```mermaid
flowchart TD
    subgraph Client["Client Application"]
        A[User] -->|Enter Credentials| B[Login Form]
        B -->|Submit| C{Validate Input}
        C -->|Invalid| D[Show Error]
        D --> B
    end

    subgraph Auth["Authentication Service"]
        C -->|Valid| E[API Gateway]
        E --> F[Auth Service]
        F -->|Check| G[(User Database)]
        G -->|User Found| H{Verify Password}
        G -->|User Not Found| I[Return 401]
        H -->|Match| J[Generate JWT]
        H -->|No Match| I
        J --> K[Generate Refresh Token]
        K --> L[Store Session]
        L --> M[Return Tokens]
    end

    subgraph Response["Response Handling"]
        I -->|Error| N[Display Login Failed]
        M -->|Success| O[Store Tokens Locally]
        O --> P[Redirect to Dashboard]
    end

    style A fill:#e1f5fe
    style J fill:#c8e6c9
    style I fill:#ffcdd2
    style P fill:#c8e6c9
```''',
                domain="diagrams",
                subdomain="mermaid",
                tags=["flowchart", "auth", "security"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Create a Mermaid sequence diagram for microservices API call",
                code='''```mermaid
sequenceDiagram
    autonumber
    participant Client
    participant Gateway as API Gateway
    participant Auth as Auth Service
    participant Orders as Order Service
    participant Inventory as Inventory Service
    participant Payment as Payment Service
    participant Queue as Message Queue
    participant Notify as Notification Service

    Client->>+Gateway: POST /orders (JWT Token)
    Gateway->>+Auth: Validate Token
    Auth-->>-Gateway: Token Valid (User ID)

    Gateway->>+Orders: Create Order Request
    Orders->>+Inventory: Check Stock
    Inventory-->>-Orders: Stock Available

    Orders->>+Payment: Process Payment
    Payment-->>-Orders: Payment Successful

    Orders->>Inventory: Reserve Items
    activate Inventory
    Inventory-->>Orders: Items Reserved
    deactivate Inventory

    Orders->>Queue: Publish OrderCreated Event
    activate Queue
    Queue-->>Orders: Event Published
    deactivate Queue

    Orders-->>-Gateway: Order Created (Order ID)
    Gateway-->>-Client: 201 Created

    Queue->>+Notify: OrderCreated Event
    Notify->>Notify: Generate Email
    Notify->>Client: Send Confirmation Email
    Notify-->>-Queue: Event Processed

    Note over Client,Notify: Async notification flow
```''',
                domain="diagrams",
                subdomain="mermaid",
                tags=["sequence", "microservices", "api"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create a Mermaid ER diagram for e-commerce database",
                code='''```mermaid
erDiagram
    CUSTOMER ||--o{ ORDER : places
    CUSTOMER {
        uuid id PK
        string email UK
        string name
        string phone
        datetime created_at
        datetime updated_at
    }

    CUSTOMER ||--o{ ADDRESS : has
    ADDRESS {
        uuid id PK
        uuid customer_id FK
        string type
        string street
        string city
        string state
        string postal_code
        string country
        boolean is_default
    }

    ORDER ||--|{ ORDER_ITEM : contains
    ORDER {
        uuid id PK
        uuid customer_id FK
        uuid shipping_address_id FK
        uuid billing_address_id FK
        string status
        decimal subtotal
        decimal tax
        decimal shipping
        decimal total
        datetime ordered_at
        datetime shipped_at
        datetime delivered_at
    }

    ORDER_ITEM {
        uuid id PK
        uuid order_id FK
        uuid product_id FK
        uuid variant_id FK
        integer quantity
        decimal unit_price
        decimal total
    }

    PRODUCT ||--o{ PRODUCT_VARIANT : has
    PRODUCT {
        uuid id PK
        uuid category_id FK
        string name
        text description
        string slug UK
        boolean is_active
        datetime created_at
    }

    PRODUCT_VARIANT {
        uuid id PK
        uuid product_id FK
        string sku UK
        string name
        decimal price
        integer stock_quantity
        json attributes
    }

    CATEGORY ||--o{ PRODUCT : contains
    CATEGORY {
        uuid id PK
        uuid parent_id FK
        string name
        string slug UK
        integer sort_order
    }

    ORDER ||--o| PAYMENT : has
    PAYMENT {
        uuid id PK
        uuid order_id FK
        string provider
        string transaction_id
        string status
        decimal amount
        string currency
        json metadata
        datetime processed_at
    }

    ORDER ||--o| SHIPMENT : has
    SHIPMENT {
        uuid id PK
        uuid order_id FK
        string carrier
        string tracking_number
        string status
        datetime shipped_at
        datetime estimated_delivery
        datetime delivered_at
    }
```''',
                domain="diagrams",
                subdomain="mermaid",
                tags=["er_diagram", "database", "ecommerce"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create a Mermaid class diagram for design patterns",
                code='''```mermaid
classDiagram
    %% Observer Pattern
    class Subject {
        <<interface>>
        +attach(observer: Observer)
        +detach(observer: Observer)
        +notify()
    }

    class Observer {
        <<interface>>
        +update(subject: Subject)
    }

    class ConcreteSubject {
        -state: any
        -observers: List~Observer~
        +getState() any
        +setState(state: any)
        +attach(observer: Observer)
        +detach(observer: Observer)
        +notify()
    }

    class ConcreteObserverA {
        -subject: ConcreteSubject
        +update(subject: Subject)
    }

    class ConcreteObserverB {
        -subject: ConcreteSubject
        +update(subject: Subject)
    }

    Subject <|.. ConcreteSubject
    Observer <|.. ConcreteObserverA
    Observer <|.. ConcreteObserverB
    ConcreteSubject --> Observer : notifies

    %% Strategy Pattern
    class Strategy {
        <<interface>>
        +execute(data: any) any
    }

    class ConcreteStrategyA {
        +execute(data: any) any
    }

    class ConcreteStrategyB {
        +execute(data: any) any
    }

    class Context {
        -strategy: Strategy
        +setStrategy(strategy: Strategy)
        +executeStrategy(data: any) any
    }

    Strategy <|.. ConcreteStrategyA
    Strategy <|.. ConcreteStrategyB
    Context o-- Strategy

    %% Factory Pattern
    class Product {
        <<interface>>
        +operation() string
    }

    class ConcreteProductA {
        +operation() string
    }

    class ConcreteProductB {
        +operation() string
    }

    class Creator {
        <<abstract>>
        +factoryMethod()* Product
        +someOperation() string
    }

    class ConcreteCreatorA {
        +factoryMethod() Product
    }

    class ConcreteCreatorB {
        +factoryMethod() Product
    }

    Product <|.. ConcreteProductA
    Product <|.. ConcreteProductB
    Creator <|-- ConcreteCreatorA
    Creator <|-- ConcreteCreatorB
    Creator ..> Product : creates
```''',
                domain="diagrams",
                subdomain="mermaid",
                tags=["class_diagram", "design_patterns", "oop"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create a Mermaid state diagram for order processing",
                code='''```mermaid
stateDiagram-v2
    [*] --> Draft: Create Order

    Draft --> Pending: Submit Order
    Draft --> Cancelled: Cancel

    Pending --> PaymentProcessing: Process Payment
    Pending --> Cancelled: Cancel

    PaymentProcessing --> PaymentFailed: Payment Declined
    PaymentProcessing --> Confirmed: Payment Successful

    PaymentFailed --> PaymentProcessing: Retry Payment
    PaymentFailed --> Cancelled: Cancel Order

    Confirmed --> Processing: Start Processing
    Confirmed --> Refunded: Request Refund

    Processing --> ReadyForShipment: Pack Items
    Processing --> PartiallyFulfilled: Some Items Unavailable

    PartiallyFulfilled --> ReadyForShipment: Continue with Available
    PartiallyFulfilled --> Cancelled: Cancel Order

    ReadyForShipment --> Shipped: Hand to Carrier

    Shipped --> InTransit: Carrier Pickup
    InTransit --> OutForDelivery: Last Mile
    OutForDelivery --> Delivered: Delivery Confirmed
    OutForDelivery --> DeliveryFailed: Delivery Attempt Failed

    DeliveryFailed --> InTransit: Reschedule Delivery
    DeliveryFailed --> Returned: Return to Sender

    Delivered --> [*]
    Delivered --> ReturnRequested: Request Return

    ReturnRequested --> ReturnApproved: Approve Return
    ReturnRequested --> ReturnDenied: Deny Return

    ReturnApproved --> ReturnInTransit: Ship Return
    ReturnInTransit --> ReturnReceived: Receive Return
    ReturnReceived --> Refunded: Process Refund

    Cancelled --> [*]
    Refunded --> [*]
    ReturnDenied --> [*]
    Returned --> [*]

    note right of Confirmed
        Payment captured
        Inventory reserved
    end note

    note right of Shipped
        Tracking number generated
        Customer notified
    end note
```''',
                domain="diagrams",
                subdomain="mermaid",
                tags=["state_diagram", "order", "workflow"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create a Mermaid C4 architecture diagram for cloud application",
                code='''```mermaid
C4Context
    title System Context Diagram for E-Commerce Platform

    Person(customer, "Customer", "A user who shops on the platform")
    Person(admin, "Admin", "Platform administrator")

    System(ecommerce, "E-Commerce Platform", "Allows customers to browse and purchase products")

    System_Ext(payment, "Payment Gateway", "Handles payment processing (Stripe)")
    System_Ext(email, "Email Service", "Sends transactional emails (SendGrid)")
    System_Ext(shipping, "Shipping Provider", "Manages shipping and tracking (FedEx API)")
    System_Ext(analytics, "Analytics", "User behavior tracking (Google Analytics)")

    Rel(customer, ecommerce, "Browses, searches, purchases")
    Rel(admin, ecommerce, "Manages products, orders, users")
    Rel(ecommerce, payment, "Processes payments")
    Rel(ecommerce, email, "Sends emails")
    Rel(ecommerce, shipping, "Creates shipments, gets tracking")
    Rel(ecommerce, analytics, "Sends events")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

```mermaid
C4Container
    title Container Diagram for E-Commerce Platform

    Person(customer, "Customer", "A user who shops on the platform")

    System_Boundary(ecommerce, "E-Commerce Platform") {
        Container(web, "Web Application", "React, TypeScript", "SPA for customer shopping experience")
        Container(mobile, "Mobile App", "React Native", "iOS and Android shopping app")
        Container(admin_ui, "Admin Dashboard", "React, TypeScript", "Admin management interface")

        Container(api_gateway, "API Gateway", "Kong", "Routes requests, handles auth, rate limiting")

        Container(product_svc, "Product Service", "Node.js, Express", "Manages product catalog")
        Container(order_svc, "Order Service", "Python, FastAPI", "Handles order processing")
        Container(user_svc, "User Service", "Go", "User management and authentication")
        Container(cart_svc, "Cart Service", "Node.js", "Shopping cart management")
        Container(search_svc, "Search Service", "Python, FastAPI", "Product search with Elasticsearch")

        ContainerDb(product_db, "Product DB", "PostgreSQL", "Product catalog data")
        ContainerDb(order_db, "Order DB", "PostgreSQL", "Order and transaction data")
        ContainerDb(user_db, "User DB", "PostgreSQL", "User accounts and profiles")
        ContainerDb(cache, "Cache", "Redis", "Session and data cache")
        ContainerDb(search_idx, "Search Index", "Elasticsearch", "Product search index")
        ContainerDb(queue, "Message Queue", "RabbitMQ", "Async event processing")
    }

    System_Ext(payment, "Payment Gateway", "Stripe")
    System_Ext(cdn, "CDN", "CloudFront")

    Rel(customer, web, "Uses", "HTTPS")
    Rel(customer, mobile, "Uses", "HTTPS")
    Rel(web, cdn, "Static assets")
    Rel(web, api_gateway, "API calls", "HTTPS/JSON")
    Rel(mobile, api_gateway, "API calls", "HTTPS/JSON")

    Rel(api_gateway, product_svc, "Routes to")
    Rel(api_gateway, order_svc, "Routes to")
    Rel(api_gateway, user_svc, "Routes to")
    Rel(api_gateway, cart_svc, "Routes to")
    Rel(api_gateway, search_svc, "Routes to")

    Rel(product_svc, product_db, "Reads/Writes")
    Rel(order_svc, order_db, "Reads/Writes")
    Rel(user_svc, user_db, "Reads/Writes")
    Rel(cart_svc, cache, "Reads/Writes")
    Rel(search_svc, search_idx, "Queries")

    Rel(order_svc, payment, "Processes payments")
    Rel(order_svc, queue, "Publishes events")

    UpdateLayoutConfig($c4ShapeInRow="4", $c4BoundaryInRow="1")
```''',
                domain="diagrams",
                subdomain="mermaid",
                tags=["c4", "architecture", "containers"],
                difficulty="advanced"
            ),
        ]

    def _plantuml_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a PlantUML sequence diagram for OAuth2 authorization flow",
                code='''@startuml OAuth2 Authorization Code Flow
!theme cerulean

title OAuth2 Authorization Code Flow with PKCE

actor User
participant "Browser" as Browser
participant "Client App" as Client
participant "Authorization\\nServer" as AuthServer
participant "Resource\\nServer" as ResourceServer

== Authorization Request ==

User -> Browser: Click "Login with OAuth"
Browser -> Client: Initiate Login

Client -> Client: Generate code_verifier\\n(random string)
Client -> Client: Create code_challenge\\n= BASE64URL(SHA256(code_verifier))

Client -> Browser: Redirect to Authorization Server
Browser -> AuthServer: GET /authorize?\\nresponse_type=code&\\nclient_id=xxx&\\nredirect_uri=xxx&\\nscope=openid profile&\\nstate=xyz&\\ncode_challenge=xxx&\\ncode_challenge_method=S256

== User Authentication ==

AuthServer -> Browser: Display Login Page
Browser -> User: Show Login Form
User -> Browser: Enter Credentials
Browser -> AuthServer: POST /login (credentials)

AuthServer -> AuthServer: Validate Credentials
AuthServer -> Browser: Display Consent Screen
Browser -> User: Show Permissions Request
User -> Browser: Grant Permissions
Browser -> AuthServer: POST /consent (approved)

== Authorization Response ==

AuthServer -> AuthServer: Generate Authorization Code
AuthServer -> Browser: 302 Redirect to redirect_uri?\\ncode=xxx&state=xyz
Browser -> Client: GET /callback?code=xxx&state=xyz

Client -> Client: Verify state matches

== Token Exchange ==

Client -> AuthServer: POST /token\\ngrant_type=authorization_code&\\ncode=xxx&\\nredirect_uri=xxx&\\nclient_id=xxx&\\ncode_verifier=xxx

AuthServer -> AuthServer: Verify code_challenge\\n= BASE64URL(SHA256(code_verifier))
AuthServer -> AuthServer: Validate authorization code

AuthServer --> Client: 200 OK\\n{\\n  "access_token": "...",\\n  "refresh_token": "...",\\n  "id_token": "...",\\n  "expires_in": 3600\\n}

== Resource Access ==

Client -> Client: Store tokens securely

Client -> ResourceServer: GET /api/userinfo\\nAuthorization: Bearer {access_token}

ResourceServer -> ResourceServer: Validate JWT
ResourceServer --> Client: 200 OK\\n{"sub": "123", "name": "John"}

Client -> Browser: Display User Profile
Browser -> User: Show "Welcome, John!"

== Token Refresh ==

note over Client: Access token expired

Client -> AuthServer: POST /token\\ngrant_type=refresh_token&\\nrefresh_token=xxx&\\nclient_id=xxx

AuthServer --> Client: 200 OK\\n{\\n  "access_token": "...",\\n  "refresh_token": "..."\\n}

@enduml''',
                domain="diagrams",
                subdomain="plantuml",
                tags=["sequence", "oauth", "security"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create a PlantUML component diagram for microservices architecture",
                code='''@startuml Microservices Architecture
!theme aws-orange

title E-Commerce Microservices Architecture

' External Actors
actor "Customer" as customer
actor "Admin" as admin

' Cloud boundary
cloud "AWS Cloud" {

    ' CDN and Edge
    node "CloudFront CDN" as cdn {
        [Static Assets]
        [API Caching]
    }

    ' Load Balancing
    node "Application Load Balancer" as alb {
        [HTTPS Termination]
        [Path-based Routing]
    }

    ' API Gateway
    rectangle "API Gateway" as gateway {
        [Kong Gateway] as kong
        [Rate Limiting]
        [Authentication]
        [Request Validation]
    }

    ' Services
    package "Microservices" {
        rectangle "User Service" as user_svc {
            [User API]
            [Auth Handler]
            database "User DB\\n(PostgreSQL)" as user_db
        }

        rectangle "Product Service" as product_svc {
            [Product API]
            [Catalog Manager]
            database "Product DB\\n(PostgreSQL)" as product_db
        }

        rectangle "Order Service" as order_svc {
            [Order API]
            [Order Processor]
            database "Order DB\\n(PostgreSQL)" as order_db
        }

        rectangle "Cart Service" as cart_svc {
            [Cart API]
            database "Cart Cache\\n(Redis)" as cart_cache
        }

        rectangle "Payment Service" as payment_svc {
            [Payment API]
            [Payment Processor]
        }

        rectangle "Notification Service" as notify_svc {
            [Email Sender]
            [Push Notifications]
        }

        rectangle "Search Service" as search_svc {
            [Search API]
            database "Elasticsearch" as es
        }
    }

    ' Message Queue
    queue "Amazon SQS" as sqs {
        [Order Events]
        [Notification Queue]
    }

    ' Event Bus
    rectangle "Amazon EventBridge" as eventbridge {
        [Event Router]
    }

    ' External Services
    rectangle "External Services" {
        [Stripe API] as stripe
        [SendGrid API] as sendgrid
        [Twilio API] as twilio
    }
}

' Relationships
customer --> cdn : HTTPS
admin --> cdn : HTTPS
cdn --> alb
alb --> kong

kong --> [User API]
kong --> [Product API]
kong --> [Order API]
kong --> [Cart API]
kong --> [Search API]

[User API] --> user_db
[Product API] --> product_db
[Order API] --> order_db
[Cart API] --> cart_cache

[Order Processor] --> [Payment API]
[Payment API] --> stripe

[Order Processor] --> eventbridge
eventbridge --> sqs
sqs --> [Email Sender]
sqs --> [Push Notifications]

[Email Sender] --> sendgrid
[Push Notifications] --> twilio

[Product API] ..> es : Sync
[Search API] --> es

' Notes
note right of kong
  - JWT Validation
  - API Key Management
  - Request/Response Transform
end note

note bottom of sqs
  Async Event Processing
  - Order Created
  - Payment Complete
  - Shipment Update
end note

@enduml''',
                domain="diagrams",
                subdomain="plantuml",
                tags=["component", "microservices", "architecture"],
                difficulty="advanced"
            ),
            DomainExample(
                prompt="Create a PlantUML activity diagram for CI/CD pipeline",
                code='''@startuml CI/CD Pipeline
!theme blueprint

title CI/CD Pipeline with Quality Gates

start

partition "Source Control" {
    :Developer pushes code;
    :Trigger webhook;
}

partition "Build Stage" {
    fork
        :Checkout code;
    fork again
        :Restore cache;
    end fork

    :Install dependencies;
    :Compile/Build;

    if (Build successful?) then (yes)
        :Create build artifacts;
    else (no)
        :Send failure notification;
        stop
    endif
}

partition "Test Stage" {
    fork
        partition "Unit Tests" {
            :Run unit tests;
            :Generate coverage report;
        }
    fork again
        partition "Static Analysis" {
            :Run linters;
            :Run SAST scan;
        }
    fork again
        partition "Security Scan" {
            :Dependency check;
            :Container scan;
        }
    end fork

    :Aggregate test results;

    if (All tests passed?) then (yes)
        if (Coverage >= 80%?) then (yes)
            :Quality gate passed;
        else (no)
            :Coverage too low;
            stop
        endif
    else (no)
        :Tests failed;
        stop
    endif
}

partition "Build Container" {
    :Build Docker image;
    :Tag image with commit SHA;
    :Push to container registry;
    :Scan container for vulnerabilities;

    if (Critical vulnerabilities?) then (yes)
        :Block deployment;
        stop
    else (no)
        :Image approved;
    endif
}

partition "Deploy to Staging" {
    :Pull image to staging cluster;
    :Run database migrations;
    :Deploy application;
    :Wait for health checks;

    if (Deployment healthy?) then (yes)
        :Run smoke tests;
    else (no)
        :Rollback deployment;
        stop
    endif

    if (Smoke tests passed?) then (yes)
        :Staging deployment complete;
    else (no)
        :Rollback deployment;
        stop
    endif
}

partition "Integration Tests" {
    :Run E2E tests;
    :Run performance tests;
    :Run API contract tests;

    if (All integration tests passed?) then (yes)
        :Ready for production;
    else (no)
        :Block production deployment;
        stop
    endif
}

partition "Deploy to Production" {
    :Request approval;

    if (Approved?) then (yes)
        :Blue-green deployment;

        fork
            :Deploy to green environment;
        fork again
            :Keep blue environment running;
        end fork

        :Run canary analysis;

        if (Canary healthy?) then (yes)
            :Shift 100% traffic to green;
            :Terminate blue environment;
        else (no)
            :Rollback to blue;
            stop
        endif
    else (no)
        :Deployment cancelled;
        stop
    endif
}

partition "Post-Deployment" {
    fork
        :Update monitoring dashboards;
    fork again
        :Send deployment notification;
    fork again
        :Tag release in Git;
    fork again
        :Update changelog;
    end fork

    :Monitor for 30 minutes;

    if (Errors detected?) then (yes)
        :Trigger automatic rollback;
        :Alert on-call engineer;
        stop
    else (no)
        #palegreen:Deployment successful!;
    endif
}

stop

@enduml''',
                domain="diagrams",
                subdomain="plantuml",
                tags=["activity", "cicd", "devops"],
                difficulty="advanced"
            ),
        ]

    def _drawio_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create a Draw.io XML for AWS architecture diagram",
                code='''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" modified="2024-01-15T10:00:00.000Z" agent="5.0">
  <diagram name="AWS Architecture" id="aws-arch">
    <mxGraphModel dx="1434" dy="780" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- AWS Cloud Boundary -->
        <mxCell id="aws-cloud" value="AWS Cloud" style="points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=12;fontStyle=0;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_aws_cloud_alt;strokeColor=#232F3E;fillColor=none;verticalAlign=top;align=left;spacingLeft=30;fontColor=#232F3E;dashed=0;" vertex="1" parent="1">
          <mxGeometry x="40" y="40" width="1080" height="700" as="geometry" />
        </mxCell>

        <!-- VPC -->
        <mxCell id="vpc" value="VPC 10.0.0.0/16" style="points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=12;fontStyle=0;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_vpc;strokeColor=#248814;fillColor=none;verticalAlign=top;align=left;spacingLeft=30;fontColor=#AAB7B8;dashed=0;" vertex="1" parent="aws-cloud">
          <mxGeometry x="60" y="60" width="960" height="580" as="geometry" />
        </mxCell>

        <!-- Public Subnet AZ1 -->
        <mxCell id="public-subnet-1" value="Public Subnet AZ1&#xa;10.0.1.0/24" style="points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=12;fontStyle=0;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_security_group;grStroke=0;strokeColor=#248814;fillColor=#E9F3E6;verticalAlign=top;align=left;spacingLeft=30;fontColor=#248814;dashed=0;" vertex="1" parent="vpc">
          <mxGeometry x="20" y="40" width="200" height="240" as="geometry" />
        </mxCell>

        <!-- Public Subnet AZ2 -->
        <mxCell id="public-subnet-2" value="Public Subnet AZ2&#xa;10.0.2.0/24" style="points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=12;fontStyle=0;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_security_group;grStroke=0;strokeColor=#248814;fillColor=#E9F3E6;verticalAlign=top;align=left;spacingLeft=30;fontColor=#248814;dashed=0;" vertex="1" parent="vpc">
          <mxGeometry x="240" y="40" width="200" height="240" as="geometry" />
        </mxCell>

        <!-- Private Subnet AZ1 -->
        <mxCell id="private-subnet-1" value="Private Subnet AZ1&#xa;10.0.3.0/24" style="points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=12;fontStyle=0;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_security_group;grStroke=0;strokeColor=#147EBA;fillColor=#E6F2F8;verticalAlign=top;align=left;spacingLeft=30;fontColor=#147EBA;dashed=0;" vertex="1" parent="vpc">
          <mxGeometry x="20" y="300" width="200" height="260" as="geometry" />
        </mxCell>

        <!-- Private Subnet AZ2 -->
        <mxCell id="private-subnet-2" value="Private Subnet AZ2&#xa;10.0.4.0/24" style="points=[[0,0],[0.25,0],[0.5,0],[0.75,0],[1,0],[1,0.25],[1,0.5],[1,0.75],[1,1],[0.75,1],[0.5,1],[0.25,1],[0,1],[0,0.75],[0,0.5],[0,0.25]];outlineConnect=0;gradientColor=none;html=1;whiteSpace=wrap;fontSize=12;fontStyle=0;container=1;pointerEvents=0;collapsible=0;recursiveResize=0;shape=mxgraph.aws4.group;grIcon=mxgraph.aws4.group_security_group;grStroke=0;strokeColor=#147EBA;fillColor=#E6F2F8;verticalAlign=top;align=left;spacingLeft=30;fontColor=#147EBA;dashed=0;" vertex="1" parent="vpc">
          <mxGeometry x="240" y="300" width="200" height="260" as="geometry" />
        </mxCell>

        <!-- ALB -->
        <mxCell id="alb" value="Application&#xa;Load Balancer" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#4D27AA;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.application_load_balancer;" vertex="1" parent="public-subnet-1">
          <mxGeometry x="70" y="80" width="60" height="60" as="geometry" />
        </mxCell>

        <!-- NAT Gateway -->
        <mxCell id="nat" value="NAT&#xa;Gateway" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#4D27AA;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.nat_gateway;" vertex="1" parent="public-subnet-2">
          <mxGeometry x="70" y="80" width="60" height="60" as="geometry" />
        </mxCell>

        <!-- ECS Tasks -->
        <mxCell id="ecs-task-1" value="ECS Task&#xa;(API)" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#D45B07;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.ecs_task;" vertex="1" parent="private-subnet-1">
          <mxGeometry x="30" y="80" width="50" height="50" as="geometry" />
        </mxCell>

        <mxCell id="ecs-task-2" value="ECS Task&#xa;(Worker)" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#D45B07;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.ecs_task;" vertex="1" parent="private-subnet-1">
          <mxGeometry x="120" y="80" width="50" height="50" as="geometry" />
        </mxCell>

        <!-- RDS -->
        <mxCell id="rds" value="RDS&#xa;PostgreSQL" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#3334AA;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.rds_postgresql_instance;" vertex="1" parent="private-subnet-2">
          <mxGeometry x="30" y="80" width="50" height="50" as="geometry" />
        </mxCell>

        <!-- ElastiCache -->
        <mxCell id="elasticache" value="ElastiCache&#xa;Redis" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#3334AA;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.elasticache_for_redis;" vertex="1" parent="private-subnet-2">
          <mxGeometry x="120" y="80" width="50" height="50" as="geometry" />
        </mxCell>

        <!-- Data Layer -->
        <mxCell id="s3" value="S3 Bucket" style="sketch=0;outlineConnect=0;fontColor=#232F3E;gradientColor=none;fillColor=#277116;strokeColor=none;dashed=0;verticalLabelPosition=bottom;verticalAlign=top;align=center;html=1;fontSize=12;fontStyle=0;aspect=fixed;pointerEvents=1;shape=mxgraph.aws4.bucket;" vertex="1" parent="vpc">
          <mxGeometry x="520" y="120" width="60" height="60" as="geometry" />
        </mxCell>

        <!-- Connections -->
        <mxCell id="conn1" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#666666;" edge="1" parent="vpc" source="alb" target="ecs-task-1">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="conn2" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#666666;" edge="1" parent="vpc" source="ecs-task-1" target="rds">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

        <mxCell id="conn3" style="edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=#666666;dashed=1;" edge="1" parent="vpc" source="ecs-task-1" target="elasticache">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>

      </root>
    </mxGraphModel>
  </diagram>
</mxfile>''',
                domain="diagrams",
                subdomain="drawio",
                tags=["aws", "architecture", "xml"],
                difficulty="advanced"
            ),
        ]

    def _ascii_examples(self) -> List[DomainExample]:
        return [
            DomainExample(
                prompt="Create an ASCII architecture diagram for microservices",
                code='''
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MICROSERVICES ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │    CLIENTS      │
                              │  (Web/Mobile)   │
                              └────────┬────────┘
                                       │
                                       ▼
                        ┌──────────────────────────┐
                        │      API GATEWAY         │
                        │  ┌────────────────────┐  │
                        │  │ • Authentication   │  │
                        │  │ • Rate Limiting    │  │
                        │  │ • Load Balancing   │  │
                        │  │ • Request Routing  │  │
                        │  └────────────────────┘  │
                        └──────────────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
           ▼                           ▼                           ▼
┌──────────────────┐        ┌──────────────────┐        ┌──────────────────┐
│   USER SERVICE   │        │  ORDER SERVICE   │        │ PRODUCT SERVICE  │
│                  │        │                  │        │                  │
│  ┌────────────┐  │        │  ┌────────────┐  │        │  ┌────────────┐  │
│  │   REST     │  │        │  │   REST     │  │        │  │   REST     │  │
│  │   API      │  │        │  │   API      │  │        │  │   API      │  │
│  └─────┬──────┘  │        │  └─────┬──────┘  │        │  └─────┬──────┘  │
│        │         │        │        │         │        │        │         │
│  ┌─────▼──────┐  │        │  ┌─────▼──────┐  │        │  ┌─────▼──────┐  │
│  │  Business  │  │        │  │  Business  │  │        │  │  Business  │  │
│  │   Logic    │  │        │  │   Logic    │  │        │  │   Logic    │  │
│  └─────┬──────┘  │        │  └─────┬──────┘  │        │  └─────┬──────┘  │
│        │         │        │        │         │        │        │         │
│  ┌─────▼──────┐  │        │  ┌─────▼──────┐  │        │  ┌─────▼──────┐  │
│  │ Repository │  │        │  │ Repository │  │        │  │ Repository │  │
│  └─────┬──────┘  │        │  └─────┬──────┘  │        │  └─────┬──────┘  │
└────────┼─────────┘        └────────┼─────────┘        └────────┼─────────┘
         │                           │                           │
         ▼                           ▼                           ▼
   ┌──────────┐                ┌──────────┐                ┌──────────┐
   │PostgreSQL│                │PostgreSQL│                │  MongoDB │
   │  (Users) │                │ (Orders) │                │(Products)│
   └──────────┘                └──────────┘                └──────────┘

                    ┌─────────────────────────────────┐
                    │         MESSAGE BROKER          │
                    │          (RabbitMQ)             │
                    └─────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
           │ Order       │ │ Notification│ │ Analytics   │
           │ Events      │ │ Events      │ │ Events      │
           └─────────────┘ └─────────────┘ └─────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│  OBSERVABILITY STACK                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Prometheus  │  │   Grafana   │  │    Loki     │  │   Jaeger    │         │
│  │  (Metrics)  │  │ (Dashboards)│  │   (Logs)    │  │  (Tracing)  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  ───▶  Synchronous call (REST/gRPC)
  ───>  Asynchronous message
  [DB]  Database
''',
                domain="diagrams",
                subdomain="ascii",
                tags=["ascii", "architecture", "microservices"],
                difficulty="intermediate"
            ),
            DomainExample(
                prompt="Create an ASCII flow diagram for data pipeline",
                code='''
╔═══════════════════════════════════════════════════════════════════════════════╗
║                         DATA PIPELINE ARCHITECTURE                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

    DATA SOURCES                    INGESTION                      PROCESSING
    ════════════                    ═════════                      ══════════

  ┌──────────────┐
  │   MySQL DB   │───┐
  │  (OLTP)      │   │
  └──────────────┘   │
                     │
  ┌──────────────┐   │         ┌─────────────────┐        ┌─────────────────┐
  │  PostgreSQL  │───┼────────▶│  Apache Kafka   │───────▶│  Apache Spark   │
  │  (Analytics) │   │         │                 │        │                 │
  └──────────────┘   │         │  ┌───────────┐  │        │  ┌───────────┐  │
                     │         │  │ Topic A   │  │        │  │  Bronze   │  │
  ┌──────────────┐   │         │  ├───────────┤  │        │  │   Layer   │  │
  │  REST APIs   │───┤         │  │ Topic B   │  │        │  ├───────────┤  │
  │  (External)  │   │         │  ├───────────┤  │        │  │  Silver   │  │
  └──────────────┘   │         │  │ Topic C   │  │        │  │   Layer   │  │
                     │         │  └───────────┘  │        │  ├───────────┤  │
  ┌──────────────┐   │         │                 │        │  │   Gold    │  │
  │  S3 Bucket   │───┤         │ Partitions: 12  │        │  │   Layer   │  │
  │  (Files)     │   │         │ Replication: 3  │        │  └───────────┘  │
  └──────────────┘   │         └─────────────────┘        └────────┬────────┘
                     │                                              │
  ┌──────────────┐   │                                              │
  │   IoT Data   │───┘                                              │
  │  (Sensors)   │                                                  │
  └──────────────┘                                                  │
                                                                    │
                                                                    ▼
    STORAGE                         SERVING                    CONSUMPTION
    ═══════                         ═══════                    ═══════════

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│   Delta Lake    │        │    Presto/      │        │    Tableau      │
│                 │───────▶│    Trino        │───────▶│   Dashboards    │
│  ┌───────────┐  │        │                 │        │                 │
│  │  Bronze   │  │        │  ┌───────────┐  │        └─────────────────┘
│  │  Tables   │  │        │  │ Query     │  │
│  ├───────────┤  │        │  │ Engine    │  │        ┌─────────────────┐
│  │  Silver   │  │        │  └───────────┘  │        │   Jupyter       │
│  │  Tables   │  │        │                 │───────▶│   Notebooks     │
│  ├───────────┤  │        │  Workers: 50    │        │                 │
│  │   Gold    │  │        │  Memory: 256GB  │        └─────────────────┘
│  │  Tables   │  │        └─────────────────┘
│  └───────────┘  │                │                  ┌─────────────────┐
│                 │                │                  │   ML Models     │
│  Storage: 10TB  │                └─────────────────▶│  (SageMaker)    │
│  Files: 1M+     │                                   │                 │
└─────────────────┘                                   └─────────────────┘


    ORCHESTRATION                  MONITORING                    GOVERNANCE
    ═════════════                  ══════════                    ══════════

┌─────────────────┐        ┌─────────────────┐        ┌─────────────────┐
│  Apache Airflow │        │    Datadog      │        │   Unity Catalog │
│                 │        │                 │        │                 │
│  DAGs: 50+      │        │  Metrics        │        │  Data Lineage   │
│  Tasks: 500+    │        │  Logs           │        │  Access Control │
│  Schedule: 24/7 │        │  Traces         │        │  Quality Rules  │
└─────────────────┘        └─────────────────┘        └─────────────────┘


═══════════════════════════════════════════════════════════════════════════════
 Data Flow:  ────▶  Sync        - - -▶  Async        ═════  Batch Process
═══════════════════════════════════════════════════════════════════════════════
''',
                domain="diagrams",
                subdomain="ascii",
                tags=["ascii", "data_pipeline", "etl"],
                difficulty="advanced"
            ),
        ]
